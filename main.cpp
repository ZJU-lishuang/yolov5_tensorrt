#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "common.h"
#include "logger.h"
#include <NvOnnxParser.h>
#include "NvInfer.h"
#include <fstream>
#include <numeric>

#include "buffers.h"
#include "utils.h"

int BATCH_SIZE=1;
int IMAGE_WIDTH=640;
int IMAGE_HEIGHT=640;
int INPUT_CHANNEL=3;

template <typename T>
using UniquePtr = std::unique_ptr<T, common::InferDeleter>;
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    int index = 0;
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

        //HWC TO CHW
        int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
        std::vector<cv::Mat> split_img = {
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 2)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 1)),
                cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * index)
        };
        index += 3;
        cv::split(flt_img, split_img);
    }
    return result;
}

void initInputParams(common::InputParams &inputParams){
    inputParams.ImgH = 640;
    inputParams.ImgW = 640;
    inputParams.ImgC = 3;
    inputParams.BatchSize = 1;
    // inputParams.HWC = false;
    inputParams.IsPadding = true;
    inputParams.InputTensorNames = std::vector<std::string>{"images"};
//   inputParams.OutputTensorNames = std::vector<std::string>{"output", "734", "735"};
    inputParams.OutputTensorNames = std::vector<std::string>{"output", "631", "632"}; // l
//    inputParams.pFunction = [](const unsigned char &x){return static_cast<float>(x) /255;};
    inputParams.pFunction = [](unsigned char &x){return static_cast<float>(x) /255;};
}
std::vector<common::Anchor> initAnchors(){
    std::vector<common::Anchor> anchors;
    common::Anchor anchor;
    // 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326
    anchor.width = 10;
    anchor.height = 13;
    anchors.emplace_back(anchor);
    anchor.width = 16;
    anchor.height = 30;
    anchors.emplace_back(anchor);
    anchor.width = 32;
    anchor.height = 23;
    anchors.emplace_back(anchor);
    anchor.width = 30;
    anchor.height = 61;
    anchors.emplace_back(anchor);
    anchor.width = 62;
    anchor.height = 45;
    anchors.emplace_back(anchor);
    anchor.width = 59;
    anchor.height = 119;
    anchors.emplace_back(anchor);
    anchor.width = 116;
    anchor.height = 90;
    anchors.emplace_back(anchor);
    anchor.width = 156;
    anchor.height = 198;
    anchors.emplace_back(anchor);
    anchor.width = 373;
    anchor.height = 326;
    anchors.emplace_back(anchor);
    return anchors;
}

void initDetectParams(common::DetectParams &yoloParams){
    yoloParams.Strides = std::vector<int> {8, 16, 32};
    yoloParams.Anchors = initAnchors();
    yoloParams.AnchorPerScale = 3;
    yoloParams.NumClass = 80;
    yoloParams.NMSThreshold = 0.5;
    yoloParams.PostThreshold = 0.6;
}

std::vector<float> preProcess(const std::vector<cv::Mat> &images) {
    common::InputParams mInputParams;
    initInputParams(mInputParams);
    const unsigned long len_img = mInputParams.ImgH * mInputParams.ImgW * 3;
    std::vector<float> files_data(len_img * images.size());
    for(auto img_count=0; img_count<images.size(); ++img_count){
        auto image = images[img_count];
        cv::Mat image_processed(mInputParams.ImgH, mInputParams.ImgW, CV_8UC3);
        int dh = 0;
        int dw = 0;
        if(mInputParams.IsPadding){
            int ih = image.rows;
            int iw = image.cols;
            float scale = std::min(static_cast<float>(mInputParams.ImgW) / static_cast<float>(iw), static_cast<float>(mInputParams.ImgH) / static_cast<float>(ih));
            int nh = static_cast<int>(scale * static_cast<float>(ih));
            int nw = static_cast<int>(scale * static_cast<float>(iw));
            dh = (mInputParams.ImgH - nh) / 2;
            dw = (mInputParams.ImgW - nw) / 2;
            cv::Mat image_resized = cv::Mat(nh, nw, CV_8UC3);
            cv::resize(image, image_resized, cv::Size(nw, nh));
            //this->resize_bilinear_c3(image.data, image.cols, image.rows, image_resized.data, nw, nh);
            cv::copyMakeBorder(image_resized, image_processed, dh, mInputParams.ImgH-nh-dh, dw, mInputParams.ImgW-nw-dw, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
        }else{
//            this->resize_bilinear_c3(image.data, image.cols, image.rows, image_processed.data, mInputParams.ImgW, mInputParams.ImgH);
            cv::Mat image_resized = cv::Mat(mInputParams.ImgH, mInputParams.ImgW, CV_8UC3);
            cv::resize(image, image_resized, cv::Size(mInputParams.ImgW, mInputParams.ImgH));
        }
        std::vector<unsigned char> file_data = image_processed.reshape(1, 1);
        //this->pixel_convert(file_data.data(), files_data.data() + img_count * len_img);
        long image_h=mInputParams.ImgH;
        long image_w=mInputParams.ImgW;
        auto *pFunc = mInputParams.pFunction;
        bool HWC=true;
        if(HWC){
            // HWC and BRG=>RGB
            for (int h=0; h<image_h; ++h){
                for (int w=0; w<image_w; ++w){
                    files_data[img_count * len_img + h * image_w * 3 + w * 3 + 0] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 2]);
                    files_data[img_count * len_img + h * image_w * 3 + w * 3 + 1] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 1]);
                    files_data[img_count * len_img + h * image_w * 3 + w * 3 + 2] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 0]);
                }
            }
        }else{
            // CHW and BRG=>RGB
            for (int h=0; h<image_h; ++h){
                for (int w=0; w<image_w; ++w) {
                    files_data[img_count * len_img + 0 * image_h * image_w + h * image_w + w] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 2]);
                    files_data[img_count * len_img + 1 * image_h * image_w + h * image_w + w] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 1]);
                    files_data[img_count * len_img + 2 * image_h * image_w + h * image_w + w] =
                            (*pFunc)(file_data[h * image_w * 3 + w * 3 + 0]);
                }
            }
        }
    }

    return files_data;
}

void safePushBack(std::vector<common::Bbox> *bboxes, common::Bbox *bbox) {
    // std::lock_guard<std::mutex> guard(mMutex);
    (*bboxes).emplace_back((*bbox));
}

void nms_cpu(std::vector<common::Bbox> &bboxes, float threshold) {
    if (bboxes.empty()){
        return ;
    }
    // 1.之前需要按照score排序
    std::sort(bboxes.begin(), bboxes.end(), [&](common::Bbox b1, common::Bbox b2){return b1.score>b2.score;});
    // 2.先求出所有bbox自己的大小
    std::vector<float> area(bboxes.size());
    for (int i=0; i<bboxes.size(); ++i){
        area[i] = (bboxes[i].xmax - bboxes[i].xmin + 1) * (bboxes[i].ymax - bboxes[i].ymin + 1);
    }
    // 3.循环
    for (int i=0; i<bboxes.size(); ++i){
        for (int j=i+1; j<bboxes.size(); ){
            float left = std::max(bboxes[i].xmin, bboxes[j].xmin);
            float right = std::min(bboxes[i].xmax, bboxes[j].xmax);
            float top = std::max(bboxes[i].ymin, bboxes[j].ymin);
            float bottom = std::min(bboxes[i].ymax, bboxes[j].ymax);
            float width = std::max(right - left + 1, 0.f);
            float height = std::max(bottom - top + 1, 0.f);
            float u_area = height * width;
            float iou = (u_area) / (area[i] + area[j] - u_area);
            if (iou>=threshold){
                bboxes.erase(bboxes.begin()+j);
                area.erase(area.begin()+j);
            }else{
                ++j;
            }
        }
    }
}

void transform(const int &ih, const int &iw, const int &oh, const int &ow, std::vector<common::Bbox> &bboxes,
                          bool is_padding) {
    if(is_padding){
        float scale = std::min(static_cast<float>(ow) / static_cast<float>(iw), static_cast<float>(oh) / static_cast<float>(ih));
        int nh = static_cast<int>(scale * static_cast<float>(ih));
        int nw = static_cast<int>(scale * static_cast<float>(iw));
        int dh = (oh - nh) / 2;
        int dw = (ow - nw) / 2;
        for (auto &bbox : bboxes){
            bbox.xmin = (bbox.xmin - dw) / scale;
            bbox.ymin = (bbox.ymin - dh) / scale;
            bbox.xmax = (bbox.xmax - dw) / scale;
            bbox.ymax = (bbox.ymax - dh) / scale;
        }
    }else{
        for (auto &bbox : bboxes){
            bbox.xmin = bbox.xmin * iw / ow;
            bbox.ymin = bbox.ymin * ih / oh;
            bbox.xmax = bbox.xmax * iw / ow;
            bbox.ymax = bbox.ymax * ih / oh;
        }
    }
}

cv::Mat renderBoundingBox(cv::Mat image, const std::vector<common::Bbox> &bboxes){
    for (auto it: bboxes){
        float score = it.score;
        cv::rectangle(image, cv::Point(it.xmin, it.ymin), cv::Point(it.xmax, it.ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(it.xmin, it.ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}

void onnxToTrtModel(const std::string &modelfile,
                    const std::string &filename,
                    nvinfer1::ICudaEngine *&engine)
{
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelfile.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    auto config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1<<20);
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "start building engine" << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;
    assert(engine);

    parser->destroy();
    //save engine
    nvinfer1::IHostMemory *data=engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    // then close everything down
    network->destroy();
    builder->destroy();

}

bool readTrtFile(const std::string &engineFile,nvinfer1::ICudaEngine *&engine)
{
    std::string cached_engine;
    std::fstream file;
    std::cout<<"loading filename from:"<<engineFile<<std::endl;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engineFile,std::ios::binary|std::ios::in);

    if(!file.is_open()){
        std::cout<<"read file error:"<<engineFile<<std::endl;
        cached_engine="";
    }

    while(file.peek()!=EOF){
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
    std::cout << "deserialize done" << std::endl;

    return true;
}

struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };
template <typename T>
T sigmoid_new(const T &n){
    return 1 / (1+exp(-n));
}


float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

void NmsDetect(std::vector<DetectRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const DetectRes &left, const DetectRes &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].classes == detections[j].classes)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > 0.5)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const DetectRes &det)
    { return det.prob == 0; }), detections.end());
}


int main()
{
    std::string onnxPath = "../yolov5_tensorrt/model/yolov5s.onnx";
    std::string save_path="../yolov5_tensorrt/model/yolov5s.serialized";
    nvinfer1::ICudaEngine *engine = nullptr;

    std::fstream existEngine;
    existEngine.open(save_path,std::ios::in);
    if(existEngine)
    {
        readTrtFile(save_path,engine);
        assert(engine != nullptr);
    }
    else
    {
        onnxToTrtModel(onnxPath,save_path,engine);
        assert(engine != nullptr);
    }



    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    std::cout<< "Preparing data..." << std::endl;
    cv::Mat image = cv::imread("../yolov5_tensorrt/images/coco_1.jpg");
    auto dims = engine->getBindingDimensions(0);
    std::vector<int> inputSize={dims.d[2],dims.d[3]};
    cv::resize(image, image, cv::Size(inputSize[1], inputSize[0]));
    std::cout<<float(image.at<cv::Vec3b>(0, 0)[0])<<std::endl;
    std::cout<<float(image.at<cv::Vec3b>(0, 0)[1])<<std::endl;
    std::cout<<float(image.at<cv::Vec3b>(0, 0)[2])<<std::endl;
    cv::Mat pixels;
    image.convertTo(pixels,CV_32FC3,1.0/255,0);

    int channels=3;
    std::vector<float> img;
    std::vector<float> data(channels* inputSize[0] * inputSize[1]);

    if(pixels.isContinuous())
        img.assign((float*)pixels.datastart,(float*)pixels.dataend);
    else{
        std::cout<<"Error reading image"<<std::endl;
        return -1;
    }

//    std::vector<float> mean {0.485, 0.456, 0.406};
//    std::vector<float> std {0.229, 0.224, 0.225};
    //HWC TO CHW BGR=>RGB
    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
//            data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
            data[c * hw + j] = img[channels * j + 2 - c];
        }
    }

    //get buffers
    assert(engine->getNbBindings() == 4);
    //buffers num == engine->getNbBindings()
    void *buffers_new[4];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers_new[i], totalSize);
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::cout << "host2device" << std::endl;
    cudaMemcpyAsync(buffers_new[0], data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    std::cout << "execute" << std::endl;
    context->execute(BATCH_SIZE,buffers_new);
//    context->enqueue(1,buffers_new,stream,nullptr);
    int outSize1 = bufferSize[1] / sizeof(float) / BATCH_SIZE;
    auto *out1 = new float[outSize1 * BATCH_SIZE];
    cudaMemcpyAsync(out1, buffers_new[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    int outSize2 = bufferSize[2] / sizeof(float) / BATCH_SIZE;
    auto *out2 = new float[outSize2 * BATCH_SIZE];
    cudaMemcpyAsync(out2, buffers_new[2], bufferSize[2], cudaMemcpyDeviceToHost, stream);
    int outSize3 = bufferSize[3] / sizeof(float) / BATCH_SIZE;
    auto *out3 = new float[outSize3 * BATCH_SIZE];
    cudaMemcpyAsync(out3, buffers_new[3], bufferSize[3], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<DetectRes> result;
    float *output1=out1;
    float *output2=out2;
    float *output3=out3;
    std::vector<float *> output={out1,out2,out3};
    
    int ratio=1;
    std::vector<int> stride = std::vector<int> {8, 16, 32};
    std::vector<std::vector<int>> grids = {
                {3, int(IMAGE_WIDTH / stride[0]), int(IMAGE_HEIGHT / stride[0])},
                {3, int(IMAGE_WIDTH / stride[1]), int(IMAGE_HEIGHT / stride[1])},
                {3, int(IMAGE_WIDTH / stride[2]), int(IMAGE_HEIGHT / stride[2])},
        };
    std::vector<std::vector<int>> anchors={{10,13}, {16,30}, {33,23}, {30,61}, {62,45}, {59,119}, {116,90}, {156,198}, {373,326}};
    for(int n=0;n<(int)grids.size();n++)
    {
//        if (n>0)
//            break;
        int position=0;
        for(int c=0;c<grids[n][0];c++)
        {
            std::vector<int> anchor=anchors[n*grids[n][0]+c];
            for(int h=0;h<grids[n][1];h++)
            {
                for(int w=0;w<grids[n][2];w++)
                {
                    // float *row=output1+position*(80+5);
                    float *row=output[n]+position*(80+5);
                    position++;
                    DetectRes box;
                    auto max_pos=std::max_element(row+5,row+80+5);
                    box.prob=Logist(row[4])*Logist(row[max_pos-row]);
                    if (box.prob < 0.5)
                        continue;
                    box.classes=max_pos-row-5;
                    box.x = (Logist(row[0]) * 2 - 0.5 + w) / grids[n][1] * IMAGE_WIDTH * ratio;
                    box.y = (Logist(row[1]) * 2 - 0.5 + h) / grids[n][2] * IMAGE_HEIGHT * ratio;
//                    std::cout<< pow(Logist(row[2]) * 2.f, 2)<<std::endl;
//                    std::cout<< anchor[0]<<std::endl;
                    box.w = pow(Logist(row[2]) * 2.f, 2) * anchor[0] * ratio;
                    box.h = pow(Logist(row[3]) * 2.f, 2) * anchor[1] * ratio;
                    result.push_back(box);
                }
            }
        }

    }
    NmsDetect(result);
    for (auto it: result){
        float score = it.prob;
        int xmin=it.x-it.w/2;
        int xmax=it.x+it.w/2;
        int ymin=it.y-it.h/2;
        int ymax=it.y+it.h/2;
        cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(score), cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }

    cv::imwrite("../yolov5_tensorrt/images/render.jpg", image);






    


}
