## yolov5_tensorrt
*yolov5 deployment*

### environment
- ubuntu 20.04
- cuda 11.0
- tensorrt 8.0.3.4
- pytorch 1.7.1

### C++
```
mkdir build
cd build
cmake ..
make
```

## TENSORRT ONNX PLUGIN

### STEP1:add a plugin layer in onnx
* in project [yolov5](https://github.com/ZJU-lishuang/yolov5-v4)

`export PYTHONPATH="$PWD" && python models/export_plugin_onnx.py --weights ./weights/yolov5s.pt --img 640 --batch 1`

### STEP2:do constant folding

#### install polygraphy
* refer to [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy)

#### constant folding
`polygraphy surgeon sanitize model.onnx --fold-constants --output model_folded.onnx`


### STEP3(Optional):add the plugin layer in onnx-tensorrt
add follow code to the `builtin_op_importers.cpp` in onnx-tensorrt.
help onnx to parse the plugin layer in tensorrt.
```c++
DEFINE_BUILTIN_OP_IMPORTER(SiLU)
{
    std::vector<nvinfer1::ITensor*> inputTensors;
    std::vector<onnx2trt::ShapedWeights> weights;
    for(int i = 0; i < inputs.size(); ++i){
        auto& item = inputs.at(i);
        if(item.is_tensor()){
            nvinfer1::ITensor* input = &convertToTensor(item, ctx);
            inputTensors.push_back(input);
        }else{
            weights.push_back(item.weights());
        }
    }
    
    LOG_VERBOSE("call silu plugin: ");
    const std::string pluginName = "SiLU";
    const std::string pluginVersion = "1";

    LOG_INFO("Searching for plugin: " << pluginName << ", plugin_version: " << pluginVersion);
    printf("node.name().c_str()=",node.name().c_str());

    // Create plugin from registry
    const auto mPluginRegistry = getPluginRegistry();
    const auto pluginCreator
        = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str());

    ASSERT(pluginCreator != nullptr, ErrorCode::kINVALID_VALUE);

    std::vector<nvinfer1::PluginField> f;
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = f.size();
    fc.fields = f.data();
    
    auto plugin = pluginCreator->createPlugin(node.name().c_str(), &fc);

    ASSERT(plugin != nullptr && "NonMaxSuppression plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);

    // auto layer = ctx->network()->addPluginV2(&tensors[0], int(tensors.size()), *plugin);
    auto layer = ctx->network()->addPluginV2(inputTensors.data(), inputTensors.size(), *plugin);
    nvinfer1::ITensor* indices = layer->getOutput(0);

    RETURN_FIRST_OUTPUT(layer);

}
```

### STEP4:add the plugin layer in TensorRT
 add the plugin layer in tensorrt by using `REGISTER_TENSORRT_PLUGIN`
 example:[SiLUPlugin.h](onnxplugin/include/SiLUPlugin.h)

