## yolov5_tensorrt
*yolov5 deployment*

### environment
- ubuntu20
- cuda11.0
- tensorrt7.2.3.4
- pytorch1.7.1

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

### STEP3:add the plugin layer in TensorRT
