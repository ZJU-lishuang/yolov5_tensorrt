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

template<typename _T>
static std::string join_dims(const std::vector<_T>& dims){
    std::stringstream output;
    char buf[64];
    const char* fmts[] = {"%d", " x %d"};
    for(int i = 0; i < dims.size(); ++i){
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

YOLOv5::YOLOv5(common::params inputparams){
    onnxPath = inputparams.onnxPath;
    save_path=inputparams.save_path;
    BATCH_SIZE=inputparams.BATCH_SIZE;
    IMAGE_WIDTH=inputparams.IMAGE_WIDTH;
    IMAGE_HEIGHT=inputparams.IMAGE_HEIGHT;

}

YOLOv5::~YOLOv5(){
    if (context) context->destroy();
    if (engine) engine->destroy();
}

void YOLOv5::onnxToTrtModel(const std::string &modelfile,
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
    config->setMaxWorkspaceSize(1<<30);

    int net_num_input = network->getNbInputs();
    printf("Network has %d inputs:", net_num_input);
    std::vector<std::string> input_names(net_num_input);
    for(int i = 0; i < net_num_input; ++i){
        auto tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d+dims.nbDims));
        printf("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

        input_names[i] = tensor->getName();
    }

    int net_num_output = network->getNbOutputs();
    printf("Network has %d outputs:", net_num_output);
    for(int i = 0; i < net_num_output; ++i){
        auto tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d+dims.nbDims));
        printf("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
    }

    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < net_num_input; ++i){
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        input_dims.d[0] = 1;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = BATCH_SIZE;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

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

bool YOLOv5::readTrtFile(const std::string &engineFile,nvinfer1::ICudaEngine *&engine)
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
        readTrtFile(save_path,engine);
        assert(engine != nullptr);
    }
    else
    {
        onnxToTrtModel(onnxPath,save_path,engine,BATCH_SIZE);
        assert(engine != nullptr);
    }
}

std::vector<float> YOLOv5::v5prepareImage(cv::Mat &image){

    int w, h, x, y;
    int input_w=IMAGE_WIDTH;
    int input_h=IMAGE_HEIGHT;
    float r_w = input_w / (image.cols*1.0);
    float r_h = input_h / (image.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * image.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * image.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(image, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(out, CV_32FC3, 1.0 / 255);
    int channels=3;
    std::vector<float> img;
    std::vector<float> data(channels* input_h * input_w);

    if (out.isContinuous())
            img.assign((float*)out.datastart, (float*)out.dataend);

    for (int c = 0; c < channels; c++) {
        for (int j = 0, hw = input_h * input_w; j < hw; j++) {
            data[c * hw + j] = img[channels * j + 2 - c];
        }
    }
    return data;
}


std::vector<int> YOLOv5::getInputSize() {
    auto dims = engine->getBindingDimensions(0);
    return {dims.d[2], dims.d[3]};
}

void YOLOv5::inferenceImage(cv::Mat image)
{
    std::vector<float> data=v5prepareImage(image);

    //get buffers
    int nbBindings=2;
    assert(engine->getNbBindings() == nbBindings);
    void *buffers_new[nbBindings];
    std::vector<int64_t> bufferSize;
    // int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        // std::cout << "binding" << i << ": " << totalSize << std::endl;
        cudaMalloc(&buffers_new[i], totalSize);
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // std::cout << "host2device" << std::endl;
    cudaMemcpyAsync(buffers_new[0], data.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    context = engine->createExecutionContext();
    assert(context != nullptr);

    context->execute(BATCH_SIZE,buffers_new);

    int outSize1 = bufferSize[1] / sizeof(float) / BATCH_SIZE;
    auto *out1 = new float[outSize1 * BATCH_SIZE];
    cudaMemcpyAsync(out1, buffers_new[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    nvinfer1::Dims outdims = engine->getBindingDimensions(1);
    int batches= outdims.d[0];
    int num_boxes = outdims.d[1];
    int num_classes = outdims.d[2]-5;
    std::vector<DetectRes> result;
    std::vector<float *> output={out1};

    float confidence_threshold=0.5;
    for(int b=0;b<batches;b++){
        for(int num_box=0;num_box<num_boxes;num_box++){
            float* pitem = out1 + (5 + num_classes) * num_box+b*num_boxes;
            float objectness = pitem[4];
            if (objectness < confidence_threshold)
                continue;
            DetectRes box;
            auto max_pos=std::max_element(pitem+5,pitem+num_classes+5);
            box.classes=max_pos-pitem-5;
            box.prob=objectness;
            box.x=pitem[0];
            box.y=pitem[1];
            box.w=pitem[2];
            box.h=pitem[3];
            result.push_back(box);
        }
    }
    
    NmsDetect(result);

    //v5prepareImage for x,y
    int w, h, x=0, y=0;
    int input_w=IMAGE_WIDTH;
    int input_h=IMAGE_HEIGHT;
    float r_w = input_w / (image.cols*1.0);
    float r_h = input_h / (image.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * image.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * image.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(image, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //show result in image
    for (auto it: result){
        float score = it.prob;
        int xmin=it.x-it.w/2-x;
        int xmax=it.x+it.w/2-x;
        int ymin=it.y-it.h/2-y;
        int ymax=it.y+it.h/2-y;
        cv::rectangle(re, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 204,0), 3);
        cv::putText(re, std::to_string(score), cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }

    cv::imwrite("../images/render.jpg", re);
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
