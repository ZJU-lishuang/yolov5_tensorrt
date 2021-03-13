// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/3/10

#ifndef TENSORRT_7_UTILS_H
#define TENSORRT_7_UTILS_H

#include "common.h"

#include <NvInfer.h>
#include <dirent.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

// ==============Pre Process=============>
std::vector<float> imagePreprocess(const std::vector<cv::Mat> &images, const int &image_h, const int &image_w, bool is_padding=true, float(*pFun)(unsigned char&)=nullptr, bool HWC=true, int work=1);

std::vector<std::pair<std::string, std::string>> searchDirectory(const std::vector<std::string> &directory, const std::vector<std::string> &suffix=std::vector<std::string>{".jpg", ".png"});


// =============Post Process=============>
//! None-Maximum-Suppression
//! \param features [xmin, ymin, xmax, ymax, cid, prob]
//! \param thres
//! \return keep index
std::vector<int> nms(std::vector<common::Bbox>, float threshold);

void sigmoid(const float *input, float *output, int length, int device_id=0);

// ===============Rendering =============>
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<common::Bbox> &bboxes);

cv::Mat renderKeypoint(cv::Mat image, const std::vector<common::Keypoint> &keypoints);

cv::Mat renderPoly(cv::Mat image, const std::vector<std::vector<cv::Point>> &polygons);

cv::Mat renderSegment(cv::Mat image, const cv::Mat &mask);

cv::Mat renderRBox(cv::Mat image, const std::vector<cv::RotatedRect> &RBox);


// ===========Template Operation ==========>
template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

template <typename T>
T clip(const T &n, const T &lower, const T &upper){
    return std::max(lower, std::min(n, upper));
}

template <typename T>
void write(char*& buffer, const T& val){
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}



// ===========Activation Fun ==========>
template <typename T>
T sigmoid(const T &n){
    return 1 / (1+exp(-n));
}

#endif //TENSORRT_7_UTILS_H
