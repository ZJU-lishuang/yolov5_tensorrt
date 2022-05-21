#include "src/module/builder/trt_builder.h"
#include "src/module/infer/trt_infer.h"
#include "src/module/core/trt_tensor.h"
#include "src/module/common/ilogger.h"
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

    std::string onnx_file = "../weights/yolov5n.onnx";
    std::string engine_file = "../weights/yolov5n.engine";
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

    // auto image = cv::imread("inference/gril.jpg");
    

}

int main(){
    
    // lesson1();
    lesson2();
    // test_tensor1();
    // test_tensor2();
    // test_tensor3();
    return 0;
}