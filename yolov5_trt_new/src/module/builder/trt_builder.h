#ifndef TRT_BUILDER_H
#define TRT_BUILDER_H

#include <functional>
#include <string>

using namespace std;
namespace TRT{

enum class Mode:int{
    FP32,
    FP16,
    INT8
};

bool compile(
    Mode mode,
    unsigned int max_batch_size,
    const string& source_onnx,
    const string& saveto,
    size_t max_workspace_size = 1<<30
);

}// namespace TRT

#endif