#include "src/module/builder/trt_builder.h"
using namespace TRT;
int main(){
    std::string onnx_file = "weights/yolov5n.onnx";
    std::string engine_file = "weights/yolov5n.engine";
    auto mode = Mode::FP32;
    unsigned int max_batch_size = 16;
    size_t max_workspace_size = 1<<30;
    TRT::compile(mode,max_batch_size,onnx_file,engine_file);

    return 0;
}