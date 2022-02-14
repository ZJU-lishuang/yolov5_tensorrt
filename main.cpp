#include <fstream>
#include "yolov5.h"
#include "SiLUPlugin.h"

void initParams(common::params &inputparams){
    inputparams.BATCH_SIZE=1;
    inputparams.IMAGE_HEIGHT=640;
    inputparams.IMAGE_WIDTH=640;
    // inputparams.onnxPath="../yolov5_tensorrt/model/yolov5s.onnx";
    // inputparams.save_path="../yolov5_tensorrt/model/yolov5s.serialized";
   inputparams.onnxPath="../model/yolov5s_plugin.onnx";
   inputparams.save_path="../model/yolov5s_plugin.serialized";
}

int main()
{
    // auto creator = getPluginRegistry()->getPluginCreator("SiLU", "1");
    // printf("success in getPluginCreator\n");
    common::params inputparams;
    initParams(inputparams);
    YOLOv5 YOLOv5(inputparams);
    YOLOv5.v5loadEngine();
    std::vector<int> inputSize=YOLOv5.getInputSize();
    assert(inputparams.IMAGE_HEIGHT==inputSize[0]);
    assert(inputparams.IMAGE_WIDTH==inputSize[1]);
    cv::Mat image = cv::imread("../images/coco_1.jpg");
    YOLOv5.inferenceImage(image);
}
