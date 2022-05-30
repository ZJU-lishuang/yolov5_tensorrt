#ifndef YOLO_H
#define YOLO_H

#include <vector>
#include <opencv2/opencv.hpp>
namespace YOLOV5{

struct DetectRes{
    int classes;
    float x;
    float y;
    float w;
    float h;
    float prob;
};

std::vector<float> v5prepareImage(cv::Mat &image,const int input_w,const int input_h);

void NmsDetect(std::vector<DetectRes> &detections);

}

#endif