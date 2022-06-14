#include "yolo.h"
#include "../../module/core/async_infer.h"
#include "../../module/infer/trt_infer.h"

namespace YOLOV5{

using namespace TRT;

void set_device(int device_id) {
    if (device_id == -1)
        return;

    checkCudaRuntime(cudaSetDevice(device_id));
}

//* load_infer函数返回了一个TRTInferImpl类
std::shared_ptr<TRTInferImpl> load_infer(const std::string& file) {
    /* 实例化一个推理对象 */
    std::shared_ptr<TRTInferImpl> infer(new TRTInferImpl());
    /* 加载trt文件，并反序列化，这里包含了模型的输入输出的绑定和流的设定 */
    if (!infer->load(file))
        infer.reset();
    return infer;
}

using ThreadSafedAsyncInferImpl = ThreadSafeAsyncInfer
<
    cv::Mat,                    // input
    BoxArray,                   // output
    std::tuple<std::string, int>,         // start param
    int                // additional
>;

class YoloTRTInferImpl : public Infer, public ThreadSafedAsyncInferImpl{
public:
    virtual ~YoloTRTInferImpl(){
        stop();
    }

    virtual bool startup(const std::string& file,int gpuid,float confidence_threshold,float nms_threshold){
        confidence_threshold_ = confidence_threshold;
        nms_threshold_ = nms_threshold;
        return ThreadSafedAsyncInferImpl::startup(std::make_tuple(file,gpuid));
    }

    virtual void worker(std::promise<bool>& result) override{
        std::string file = std::get<0>(start_param_);
        int gpuid = std::get<1>(start_param_);
        set_device(gpuid);
        auto engine = load_infer(file);
        engine->print();

        int max_batch_size = engine->get_max_batch_size();
        auto input         = engine->tensor("images");
        auto output        = engine->tensor("output");
        int num_classes    = output->size(2) - 5;

        input_width_ = input->size(3);
        input_height_ = input->size(2);

        tensor_allocator_  = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
        stream_            = engine->get_stream();
        gpu_               = gpuid;

        result.set_value(true);
        input->resize_single_dim(0, max_batch_size).to_gpu();

        std::vector<Job> fetch_jobs;

        while(get_jobs_and_wait(fetch_jobs, max_batch_size)){
            /* 一旦进来说明有图片数据 ，获取图片的张数 */
            int infer_batch_size = fetch_jobs.size();
            input->resize_single_dim(0, infer_batch_size);
            /* 下面从队列取出job，把对应的仿射矩阵和预处理好的图片数据送到模型的输入 */
            /* 其中input就是engine对象的方法，该方法实际上是把预处理的数据传给engine的内部属性inputs_  */
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                auto& job  = fetch_jobs[ibatch];
                auto& mono = job.mono_tensor->data();
                // affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                job.mono_tensor->release();
            }
            /* 开始推理 */
            engine->forward(false);
            // output_array_device.to_gpu(false);
            /* 下面进行解码 */
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                
                auto& job                 = fetch_jobs[ibatch];/* 图片数据 */
                float* image_based_output = output->gpu<float>(ibatch);
                // float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                // auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                // checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                // decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, nms_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
            }

            // output_array_device.to_cpu();
            for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                // float* parray = output_array_device.cpu<float>(ibatch);
                // int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                auto& job     = fetch_jobs[ibatch];
                auto& image_based_boxes   = job.output;
                // for(int i = 0; i < count; ++i){
                //     float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                //     int label    = pbox[5];
                //     int keepflag = pbox[6];
                //     if(keepflag == 1){
                //         image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                //     }
                // }
                job.pro->set_value(image_based_boxes);
            }
            fetch_jobs.clear();
        }
        stream_ = nullptr;
        tensor_allocator_.reset();
        INFO("Engine destroy.");

    }

    virtual bool preprocess(Job& job,const cv::Mat& image) override{
        if(tensor_allocator_ == nullptr){
            INFOE("tensor_allocator_ is nullptr");
            return false;
        }

        job.mono_tensor = tensor_allocator_->query();
        if(job.mono_tensor == nullptr){
            INFOE("Tensor allocator query failed.");
            return false;
        }

        /* 配置gpu */
        AutoDevice auto_device(gpu_);
        /* 获取job里面的tensor的数据地址，第一次为nullptr */
        /* 这里需要理解的不是创建了新的tensor对象，只是把job的tensor地址拿出来使用，数据还是job指定的 */
        auto& tensor = job.mono_tensor->data(); 
        if(tensor == nullptr){
            // not init
            tensor = std::make_shared<Tensor>();
            tensor->set_workspace(std::make_shared<MixMemory>());
        }
        /* 把tensor和流绑定，后续都会使用这个流进行处理，流的创建也是在模型创建时创建 */
        tensor->set_stream(stream_);
        /* 把tensor  resize一下，此时的tensor还未填充数据 */
        tensor->resize(1, 3, input_height_, input_width_);

        size_t size_image      = input_width_ * input_height_ * 3;
        auto workspace         = tensor->get_workspace();
        float* gpu_workspace  = (float*)workspace->gpu(size_image);
        float* image_device         = gpu_workspace;

        float* cpu_workspace        = (float*)workspace->cpu(size_image);
        float* image_host           = cpu_workspace;
        
        std::vector<float> data = YOLOV5::v5prepareImage(image,input_width_,input_height_);
        memcpy(image_host, data.data(), size_image);
        // checkCudaRuntime(cudaMemcpyAsync(image_device, data.data(), size_image*sizeof(float), cudaMemcpyHostToDevice, stream_));
        // checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image*sizeof(float), cudaMemcpyHostToDevice, stream_));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));

        return true;
    }

    virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) override{
        return ThreadSafedAsyncInferImpl::commits(images);
    }

    virtual std::shared_future<BoxArray> commit(const cv::Mat& image) override{
        return ThreadSafedAsyncInferImpl::commit(image);
    }

private:
    int input_width_ = 0;
    int input_height_ = 0;
    int gpu_ = 0;
    float confidence_threshold_ = 0;
    float nms_threshold_ = 0;
    cudaStream_t stream_ = nullptr;
};

std::shared_ptr<Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold, float nms_threshold){
    /* 创建一个推理实例，该实例具备了引擎的创建、加载模型，反序列化，创建线程等一系列操作， */
    std::shared_ptr<YoloTRTInferImpl> instance(new YoloTRTInferImpl());
    if(!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold)){
        instance.reset();
    }
    return instance;
}

std::vector<float> v5prepareImage(const cv::Mat &image,const int input_w,const int input_h){

    int w, h, x, y;
    // int input_w=IMAGE_WIDTH;
    // int input_h=IMAGE_HEIGHT;
    float r_w = input_w / (image.cols*1.0);
    float r_h = input_h / (image.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * image.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * image.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(image, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(out, CV_32FC3, 1.0 / 255);
    int channels=3;
    std::vector<float> img;
    std::vector<float> data(channels* input_h * input_w);

    if (out.isContinuous())
            img.assign((float*)out.datastart, (float*)out.dataend);

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = input_h * input_w; j < hw; j++) {
            data[c * hw + j] = img[channels * j + 2 - c];
        }
    }
    return data;
}

float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > 0.5)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}



}