#include "src/module/builder/trt_builder.h"
#include "src/module/core/trt_tensor.h"
#include <cuda_runtime.h>

using namespace TRT;

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

static void lesson1(){
    std::string onnx_file = "weights/yolov5n.onnx";
    std::string engine_file = "weights/yolov5n.engine";
    auto mode = Mode::FP32;
    unsigned int max_batch_size = 16;
    size_t max_workspace_size = 1<<30;
    compile(mode,max_batch_size,onnx_file,engine_file);

}

int main(){
    
    // lesson1();
    test_tensor1();
    return 0;
}