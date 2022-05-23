#include "trt_builder.h"
#include "../common/ilogger.h"
#include "../common/utils.h"
#include "../common/cuda_tools.h"
#include <NvOnnxParser.h>
#include <sstream>
#include <chrono>

using namespace nvinfer1;
using namespace std;

namespace TRT{

static string join_dims(const vector<int>& dims){
    stringstream output;
    char buf[64];
    const char* fmts[] = {"%d", " x %d"};
    for(int i = 0; i < dims.size(); ++i){
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

const char* mode_string(Mode type){
    switch (type){
        case Mode::FP32:
            return "FP32";
        case Mode::FP16:
            return "FP16";
        case Mode::INT8:
            return "INT8";
        default:
            return "UknowCompileMode";
    }
}

bool compile(
    Mode mode,
    unsigned int max_batch_size,
    const string& source_onnx,
    const string& saveto,
    size_t max_workspace_size){
    
    if(mode == Mode::INT8){
        INFOE("int8process must not nullptr, when in int8 mode.");
        return false;
    }

    INFO("Compile %s %s.", mode_string(mode), source_onnx.c_str());
    shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
    if (builder == nullptr) {
        INFOE("Can not create builder.");
        return false;
    }

    shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
    if (mode == Mode::FP16) {
        if (!builder->platformHasFastFp16()) {
            INFOW("Platform not have fast fp16 support");
        }
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (mode == Mode::INT8) {
        if (!builder->platformHasFastInt8()) {
            INFOW("Platform not have fast int8 support");
        }
        config->setFlag(BuilderFlag::kINT8);
    }

    shared_ptr<INetworkDefinition> network;
    shared_ptr<nvonnxparser::IParser> onnxParser;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);

    //from onnx is not markOutput
    onnxParser.reset(nvonnxparser::createParser(*network, gLogger), destroy_nvidia_pointer<nvonnxparser::IParser>);
    if (onnxParser == nullptr) {
        INFOE("Can not create parser.");
        return false;
    }

    if (!onnxParser->parseFromFile(source_onnx.c_str(), 1)) {
        INFOE("Can not parse OnnX file: %s", source_onnx.c_str());
        return false;
    }
    
    auto inputTensor = network->getInput(0);
    auto inputDims = inputTensor->getDimensions();

    INFO("Input shape is %s", join_dims(vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
    INFO("Set max batch size = %d", max_batch_size);
    INFO("Set max workspace size = %.2f MB", max_workspace_size / 1024.0f / 1024.0f);

    int net_num_input = network->getNbInputs();
    INFO("Network has %d inputs:", net_num_input);
    vector<string> input_names(net_num_input);
    for(int i = 0; i < net_num_input; ++i){
        auto tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

        input_names[i] = tensor->getName();
    }

    int net_num_output = network->getNbOutputs();
    INFO("Network has %d outputs:", net_num_output);
    for(int i = 0; i < net_num_output; ++i){
        auto tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
    }

    int net_num_layers = network->getNbLayers();
    INFO("Network has %d layers", net_num_layers);		
    builder->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(max_workspace_size);

    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < net_num_input; ++i){
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        input_dims.d[0] = 1;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = max_batch_size;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

    INFO("Building engine...");
    auto time_start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<ICudaEngine>);
    if (engine == nullptr) {
        INFOE("engine is nullptr");
        return false;
    }

    auto time_end = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    INFO("Build done %lld ms !", time_end - time_start);
    
    // serialize the engine, then close everything down
    shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);
    return save_file(saveto, seridata->data(), seridata->size());
}

}// namespace TRT