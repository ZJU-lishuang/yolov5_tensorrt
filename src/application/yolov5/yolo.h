#ifndef YOLO_H
#define YOLO_H

#include <vector>
#include <future>
#include <opencv2/opencv.hpp>
namespace YOLOV5{

struct Box{
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
    :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
};

typedef std::vector<Box> BoxArray;

class Infer{
public:
    virtual std::shared_future<BoxArray> commit(const cv::Mat& image) = 0;
    virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
};

std::shared_ptr<Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold, float nms_threshold);

struct DetectRes{
    int classes;
    float x;
    float y;
    float w;
    float h;
    float prob;
};

std::vector<float> v5prepareImage(const cv::Mat &image,const int input_w,const int input_h);

void NmsDetect(std::vector<DetectRes> &detections);

}

#endif