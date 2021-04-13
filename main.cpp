#include <fstream>
#include "yolov5.h"

void initParams(common::params &inputparams){
    inputparams.BATCH_SIZE=1;
    inputparams.IMAGE_HEIGHT=640;
    inputparams.IMAGE_WIDTH=640;
    inputparams.onnxPath="../yolov5_tensorrt/model/yolov5s.onnx";
    inputparams.save_path="../yolov5_tensorrt/model/yolov5s.serialized";
}

int main()
{
    common::params inputparams;
    initParams(inputparams);
    YOLOv5 YOLOv5(inputparams);
    YOLOv5.v5loadEngine();
    std::vector<int> inputSize=YOLOv5.getInputSize();
    assert(inputparams.IMAGE_HEIGHT==inputSize[0]);
    assert(inputparams.IMAGE_WIDTH==inputSize[1]);
    cv::Mat image = cv::imread("../yolov5_tensorrt/images/coco_1.jpg");
    YOLOv5.inferenceImage(image);
}
