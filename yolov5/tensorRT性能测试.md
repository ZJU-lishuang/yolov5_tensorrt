# tensorRT性能测试
- *使用yolov5s模型，输入尺寸640x640，tensorRT模型使用yolov5自带代码生成*
|推理框架|显存占用|单张图片推理速度|
|---|---|---|
|onnx|368MB|0.035s|
|torch && tensorrt|797MB|0.018s|
|torch && onnx|776MB|0.035s|
|tensorrt|372MB|0.016s|

- onnx使用onnxruntime推理
- torch &&表示导入了torch，`import torch`
