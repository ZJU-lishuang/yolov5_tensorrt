#include "src/module/builder/trt_builder.h"
#include "src/module/infer/trt_infer.h"
#include "src/module/core/trt_tensor.h"
#include "src/module/common/ilogger.h"
#include "src/application/yolov5/yolo.h"
#include "src/onnxplugin/include/SiLUPlugin.h"
#include <cuda_runtime.h>

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

static void lesson3(){
    std::string engine_file = "../weights/yolov5n.engine";
    float confidence_threshold = 0.4f;
    float nms_threshold = 0.5f;
    int gpuid = 0;
    //create infer
    auto yolo = YOLOV5::create_infer(engine_file,gpuid,confidence_threshold,nms_threshold);

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