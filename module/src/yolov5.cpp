#include "yolov5.h"
#include "logger.h"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <numeric>

__device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

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



YOLOv5::YOLOv5(std::string config_file){
    onnxPath = "../yolov5_tensorrt/model/yolov5s.onnx";
    save_path="../yolov5_tensorrt/model/yolov5s.serialized";
    v5BATCH_SIZE=1;

}

YOLOv5::~YOLOv5() = default;

void YOLOv5::v5onnxToTrtModel(const std::string &modelfile,
                    const std::string &filename,
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE)
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

bool YOLOv5::v5readTrtFile(const std::string &engineFile,nvinfer1::ICudaEngine *&engine)
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


void YOLOv5::v5loadEngine(){
    std::fstream existEngine;
    existEngine.open(save_path,std::ios::in);
    if(existEngine)
    {
        v5readTrtFile(save_path,engine);
        assert(engine != nullptr);
    }
    else
    {
        v5onnxToTrtModel(onnxPath,save_path,engine,v5BATCH_SIZE);
        assert(engine != nullptr);
    }
}

std::vector<float> YOLOv5::v5prepareImage(cv::Mat &image){
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
    return data;
    
}

void YOLOv5::inferenceImage()
{
    cv::Mat image = cv::imread("../yolov5_tensorrt/images/coco_1.jpg");
    std::vector<float> data=v5prepareImage(image);

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

    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    context->execute(v5BATCH_SIZE,buffers_new);

    int outSize1 = bufferSize[1] / sizeof(float) / v5BATCH_SIZE;
    auto *out1 = new float[outSize1 * v5BATCH_SIZE];
    cudaMemcpyAsync(out1, buffers_new[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    int outSize2 = bufferSize[2] / sizeof(float) / v5BATCH_SIZE;
    auto *out2 = new float[outSize2 * v5BATCH_SIZE];
    cudaMemcpyAsync(out2, buffers_new[2], bufferSize[2], cudaMemcpyDeviceToHost, stream);
    int outSize3 = bufferSize[3] / sizeof(float) / v5BATCH_SIZE;
    auto *out3 = new float[outSize3 * v5BATCH_SIZE];
    cudaMemcpyAsync(out3, buffers_new[3], bufferSize[3], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<DetectRes> result;
    std::vector<float *> output={out1,out2,out3};
    
    int ratio=1;
    int IMAGE_WIDTH=640;
    int IMAGE_HEIGHT=640;
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


float YOLOv5::IOUCalculate(const DetectRes &det_a, const DetectRes &det_b) {
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

void YOLOv5::NmsDetect(std::vector<DetectRes> &detections) {
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
