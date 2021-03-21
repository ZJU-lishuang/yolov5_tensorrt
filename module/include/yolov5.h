#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <fstream>



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
    YOLOv5(std::string config_file);
    ~YOLOv5();
    //load model
    void v5loadEngine();
    //inference image
    void inferenceImage();
private:
    int v5BATCH_SIZE;
    std::string onnxPath;
    std::string save_path;
    nvinfer1::ICudaEngine *engine = nullptr;
    void v5onnxToTrtModel(const std::string &modelfile,
                    const std::string &filename,
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE);
    bool v5readTrtFile(const std::string &engineFile,nvinfer1::ICudaEngine *&engine);
    std::vector<float> v5prepareImage(cv::Mat &image);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    void NmsDetect(std::vector<DetectRes> &detections);
};



#endif //YOLOV5_TRT_H
