#ifndef TRT_INFER_H
#define TRT_INFER_H

#include "../core/trt_tensor.h"
#include "../common/cuda_tools.h"
// #include <NvInfer.h>

// using namespace nvinfer1;

namespace TRT
{

class EngineContext{
public:
    virtual ~EngineContext() {destroy();}

    void set_stream(CUStream stream){
        if(owner_stream_){
            if(stream_){
                cudaStreamDestroy(stream_);
            }
            owner_stream_ = false;
        }
        stream_ = stream;
    }

    bool build_model(const void* pdata,size_t size){
        destroy();

        if(pdata == nullptr || size == 0)
            return false;

        owner_stream_ = true;
        checkCudaRuntime(cudaStreamCreate(&stream_));
        if(stream_ == nullptr)
            return false;
        
        runtime_ = std::shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
        if (runtime_ == nullptr)
            return false;

        engine_ = std::shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr), destroy_nvidia_pointer<ICudaEngine>);
        if (engine_ == nullptr)
            return false;

        context_ = std::shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
        return context_ != nullptr;
    }

    CUStream stream_ = nullptr;
    bool owner_stream_ = false;
    std::shared_ptr<IExecutionContext> context_;
    std::shared_ptr<ICudaEngine> engine_;
    std::shared_ptr<IRuntime> runtime_ = nullptr;

private:
    void destroy(){
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if(owner_stream_){
            if(stream_){
                cudaStreamDestroy(stream_);
            } 
        }
        stream_ = nullptr;
    }
};

class TRTInferImpl{
public:
    virtual ~TRTInferImpl();

    bool load(const std::string& file);
    bool load_from_memory(const void* pdata,size_t size);
    void destroy();
    void forward(bool sync);
    int get_max_batch_size();
    CUStream get_stream();
    void set_stream(CUStream stream);
    void synchronize();
    size_t get_device_memory_size();
    std::shared_ptr<MixMemory> get_workspace();
    std::shared_ptr<Tensor> input(int index = 0);
    std::string get_input_name(int index = 0);
    std::shared_ptr<Tensor> output(int index = 0);
    std::string get_output_name(int index = 0);
    std::shared_ptr<Tensor> tensor(const std::string& name);
    bool is_output_name(const std::string& name);
    bool is_input_name(const std::string& name);
    void set_input(int index,std::shared_ptr<Tensor> tensor);
    void set_output(int index,std::shared_ptr<Tensor> tensor);
    std::shared_ptr<std::vector<uint8_t>> serial_engine();

    void print();

    int num_output();
    int num_input();
    int device();

private:
    void build_engine_input_and_outputs_mapper();

    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::vector<int> inputs_map_to_ordered_index_;
    std::vector<int> outputs_map_to_ordered_index_;
    std::vector<std::string> inputs_name_;
    std::vector<std::string> outputs_name_;
    std::vector<std::shared_ptr<Tensor>> orderdBlobs_;
    std::map<std::string, int> blobsNameMapper_;
    std::shared_ptr<EngineContext> context_;
    std::vector<void*> bindingsPtr_;
    std::shared_ptr<MixMemory> workspace_;
    int device_ = 0;


};
    
} // namespace TRT


#endif