#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <fstream>



namespace common{
    struct params{
        int BATCH_SIZE;
        int IMAGE_HEIGHT;
        int IMAGE_WIDTH;
        std::string onnxPath;
        std::string save_path;
    };
}

class YOLOv5
{
    struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

public:
    //init
    YOLOv5(common::params inputparams);
    ~YOLOv5();
    //load model
    void v5loadEngine();
    //inference image
    void inferenceImage(cv::Mat image);
private:
    int BATCH_SIZE;
    int IMAGE_HEIGHT;
    int IMAGE_WIDTH;
    std::string onnxPath;
    std::string save_path;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    void onnxToTrtModel(const std::string &modelfile,
                    const std::string &filename,
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE);
    bool readTrtFile(const std::string &engineFile,nvinfer1::ICudaEngine *&engine);
    std::vector<float> v5prepareImage(cv::Mat &image);
    std::vector<float> prepareImage(cv::Mat &image);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    void NmsDetect(std::vector<DetectRes> &detections);
};



#endif //YOLOV5_TRT_H
