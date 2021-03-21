#include <fstream>
#include "yolov5.h"

int main()
{
    std::string config_file = "../yolov5/config.yaml";
    YOLOv5 YOLOv5(config_file);
    YOLOv5.v5loadEngine();
    YOLOv5.inferenceImage();
}
