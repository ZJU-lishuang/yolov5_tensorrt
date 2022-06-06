#include "src/module/builder/trt_builder.h"
#include "src/module/infer/trt_infer.h"
#include "src/module/core/trt_tensor.h"
#include "src/module/common/ilogger.h"
#include "src/application/yolov5/yolo.h"
#include "src/onnxplugin/include/SiLUPlugin.h"
#include <cuda_runtime.h>

#include "src/module/core/async_infer.h"

#include <unistd.h>

using namespace TRT;

static bool exists(const std::string& path){
    return access(path.c_str(), R_OK) == 0;
}

void set_device(int device_id) {
    if (device_id == -1)
        return;

    checkCudaRuntime(cudaSetDevice(device_id));
}

static void test_tensor1(){

    size_t cpu_bytes = 1024;
    size_t gpu_bytes = 2048;

    ///////////////////////////////////////////////////////////////////
    // 封装效果，自动分配和释放
    TRT::MixMemory memory;
    void* host_ptr   = memory.cpu(cpu_bytes);
    void* device_ptr = memory.gpu(gpu_bytes);

    ///////////////////////////////////////////////////////////////////
    // 不封装效果
    // void* host_ptr   = nullptr;
    // void* device_ptr = nullptr;
    // cudaMallocHost(&host_ptr, cpu_bytes);
    // cudaMalloc(&device_ptr, gpu_bytes);

    // cudaFreeHost(&host_ptr);
    // cudaFree(&device_ptr);
    ///////////////////////////////////////////////////////////////////
}

static void test_tensor2(){

    ///////////////////////////////////////////////////////////////////
    /* 内存的自动复制，依靠head属性标记数据最新的位置
       若访问的数据不是最新的，则会自动发生复制操作 */
    TRT::Tensor tensor({1, 3, 5, 5},nullptr);
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Init，内存没有分配 */

    tensor.cpu<float>()[0] = 512;               /* 访问cpu时，分配cpu内存 */
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Host */

    float* device_ptr = tensor.gpu<float>();    /* 访问gpu时，最新数据在Host，发生复制动作并标记最新数据在Device */
    INFO("tensor.head = %s", TRT::data_head_string(tensor.head()));   /* 输出 Device */
    //INFO("device_ptr[0] = %f", device_ptr[0]);                        /* 输出 512.00000，由于gpu内存修改为cudaMalloc，这里无法直接访问 */
}

static void test_tensor3(){

    ///////////////////////////////////////////////////////////////////
    /* 计算维度的偏移量 */
    TRT::Tensor tensor({1, 3, 5, 5, 2, 5},nullptr);
    auto ptr_origin   = tensor.cpu<float>();
    auto ptr_channel2 = tensor.cpu<float>(0, 2, 3, 2, 1, 3);

    INFO("Offset = %d", ptr_channel2 - ptr_origin);                          /* 输出678 */
    INFO("Offset = %d", tensor.offset(0, 2, 3, 2, 1, 3));                    /* 输出678 */

    int offset_compute = ((((0 * 3 + 2) * 5 + 3) * 5 + 2) * 2 + 1) * 5 + 3;  
    INFO("Compute = %d", offset_compute);                                    /* 输出678 */
}

static void lesson1(){
    std::string onnx_file = "weights/yolov5n.onnx";
    std::string engine_file = "weights/yolov5n.engine";
    auto mode = Mode::FP32;
    unsigned int max_batch_size = 16;
    size_t max_workspace_size = 1<<30;
    compile(mode,max_batch_size,onnx_file,engine_file);
}

static void lesson2(){
    int gpuid = 0;
    /*  设置使用GPU */
    set_device(gpuid);

    // std::string onnx_file = "../weights/yolov5n.onnx";
    // std::string engine_file = "../weights/yolov5n.engine";
    std::string onnx_file = "../weights/yolov5n.plugin.onnx";
    std::string engine_file = "../weights/yolov5n.plugin.engine";
    if(!exists(engine_file)){
        auto mode = Mode::FP32;
        unsigned int max_batch_size = 16;
        size_t max_workspace_size = 1<<30;
        compile(mode,max_batch_size,onnx_file,engine_file);
    }

    std::shared_ptr<TRTInferImpl> infer(new TRTInferImpl());
    infer->load(engine_file);
    if(infer == nullptr){
        printf("Engine %s load failed", engine_file.c_str());
        // 解除主线程阻塞，模型加载失败
        return;
    }
    /* 打印引擎相关信息 */
    infer->print();

    /* 获取引擎的相关信息 */
    int max_batch_size = infer->get_max_batch_size();
    auto input         = infer->tensor("images");
    auto output        = infer->tensor("output");
    int num_classes    = output->size(2) - 5;

    int input_width_       = input->size(3);
    int input_height_      = input->size(2);
    CUStream stream_            = infer->get_stream();

    input->resize_single_dim(0, max_batch_size).to_gpu();
    int infer_batch_size = 1;
    input->resize_single_dim(0, infer_batch_size);

    size_t size_image      = input_width_ * input_height_ * 3;
    auto workspace         = input->get_data();
    float* image_device  = (float*)workspace->gpu(size_image);

    auto image = cv::imread("../images/coco_1.jpg");
    std::vector<float> data = YOLOV5::v5prepareImage(image,input_width_,input_height_);
    
    checkCudaRuntime(cudaMemcpyAsync(image_device, data.data(), size_image*sizeof(float), cudaMemcpyHostToDevice, stream_));

    /* 开始推理 */
    infer->forward(false);

    std::vector<YOLOV5::DetectRes> result;

    float confidence_threshold=0.5;
    int num_boxes = output->size(1);
    for(int b=0;b<infer_batch_size;b++){
        float* image_based_output = output->cpu<float>(b);
        for(int num_box=0;num_box<num_boxes;num_box++){
            float* pitem = image_based_output + (5 + num_classes) * num_box+b*num_boxes;
            float objectness = pitem[4];
            if (objectness < confidence_threshold)
                continue;
            YOLOV5::DetectRes box;
            auto max_pos=std::max_element(pitem+5,pitem+num_classes+5);
            box.classes=max_pos-pitem-5;
            box.prob=objectness;
            box.x=pitem[0];
            box.y=pitem[1];
            box.w=pitem[2];
            box.h=pitem[3];
            result.push_back(box);
        }
    }

    YOLOV5::NmsDetect(result);
    //v5prepareImage for x,y
    int w, h, x=0, y=0;
    int input_w=input_width_;
    int input_h=input_height_;
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
    //show result in image
    for (auto it: result){
        float score = it.prob;
        int xmin=it.x-it.w/2-x;
        int xmax=it.x+it.w/2-x;
        int ymin=it.y-it.h/2-y;
        int ymax=it.y+it.h/2-y;
        cv::rectangle(re, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(re, std::to_string(score), cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }

    cv::imwrite("../images/render.jpg", re);

}
struct Box{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};

typedef std::vector<Box> BoxArray;

class Infer{
public:
    virtual std::shared_future<BoxArray> commit(const cv::Mat& image) = 0;
    virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
};

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

static void lesson3(){
    std::string engine_file = "../weights/yolov5n.engine";
    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    int gpuid = 0;
    //create infer
    auto yolo = create_infer(engine_file,gpuid,confidence_threshold,nms_threshold);

    auto image = cv::imread("../images/coco_1.jpg");
    // 提交图片并获取结果
    auto objs = yolo->commit(image).get();

}

int main(){
    
    // lesson1();
    // lesson2();
    lesson3();
    // test_tensor1();
    // test_tensor2();
    // test_tensor3();
    return 0;
}