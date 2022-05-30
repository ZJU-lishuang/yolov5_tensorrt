// #include <cstring>
#include <string.h>
#include "trt_tensor.h"
#include "../common/cuda_tools.h"

using namespace std;
namespace TRT{

inline static int get_device(int device_id){
    if(device_id != CURRENT_DEVICE_ID){
        check_device_id(device_id);
        return device_id;
    }

    checkCudaRuntime(cudaGetDevice(&device_id));
    return device_id;
}

MixMemory::MixMemory(int device_id){
    device_id_ = get_device(device_id);
}

MixMemory::MixMemory(void* cpu,size_t cpu_size,void* gpu,size_t gpu_size){
    reference_data(cpu,cpu_size,gpu,gpu_size);
}

void MixMemory::reference_data(void* cpu,size_t cpu_size,void* gpu,size_t gpu_size){
    release_all();

    if(cpu == nullptr || cpu_size == 0){
        cpu = nullptr;
        cpu_size = 0;
    }

    if(gpu ==nullptr || gpu_size == 0){
        gpu = nullptr;
        gpu_size = 0;
    }

    this->cpu_ = cpu;
    this->cpu_size_ = cpu_size;
    this->gpu_ = gpu;
    this->gpu_size_ = gpu_size;

    // 判断内存块是否属于MixMemory管理
    this->owner_cpu_ = !(cpu && cpu_size > 0);
    this->owner_gpu_ = !(gpu && gpu_size > 0);
    checkCudaRuntime(cudaGetDevice(&device_id_));
}

MixMemory::~MixMemory(){
    release_all();
}

void* MixMemory::gpu(size_t size){
    if(gpu_size_ < size){
        release_gpu();

        gpu_size_ = size;
        AutoDevice auto_device_exchange(device_id_);
        checkCudaRuntime(cudaMalloc(&gpu_,size));
        checkCudaRuntime(cudaMemset(gpu_,0,size));
    }
    return gpu_;
}

void* MixMemory::cpu(size_t size){
    if(cpu_size_ < size){
        release_cpu();

        cpu_size_ = size;
        AutoDevice auto_device_exchange(device_id_);
        // 锁页内存
        checkCudaRuntime(cudaMallocHost(&cpu_,size));
        Assert(cpu_ != nullptr);
        memset(cpu_,0,size);
    }
    return cpu_;
}

void MixMemory::release_cpu(){
    if(cpu_){
        if(owner_cpu_){
            AutoDevice auto_device_exchange(device_id_);
            checkCudaRuntime(cudaFreeHost(cpu_));
        }
        cpu_ = nullptr;
    }
    cpu_size_ = 0;
}

void MixMemory::release_gpu(){
    if(gpu_){
        if(owner_gpu_){
            AutoDevice auto_device_exchange(device_id_);
            checkCudaRuntime(cudaFree(gpu_));
        }
        gpu_ = nullptr;
    }
    gpu_size_ = 0;
}

void MixMemory::release_all(){
    release_cpu();
    release_gpu();
}

const char* data_head_string(DataHead dh){
    switch(dh){
        case DataHead::Init: return "Init";
        case DataHead::Device: return "Device";
        case DataHead::Host: return "Host";
        default: return "Unknow";
    }
}

Tensor::Tensor(int n,int c,int h,int w,shared_ptr<MixMemory> data,int device_id){
    this->device_id_ = get_device(device_id);
    descriptor_string_[0]=0;
    setup_data(data);
    resize(n,c,h,w);
}

Tensor::Tensor(const vector<int>& dims,shared_ptr<MixMemory> data,int device_id){
    this->device_id_ = get_device(device_id);
    descriptor_string_[0]=0;
    setup_data(data);
    resize(dims);
}

Tensor::Tensor(int ndims,const int* dims,shared_ptr<MixMemory> data,int device_id){
    this->device_id_ = get_device(device_id);
    descriptor_string_[0] = 0;
    setup_data(data);
    resize(ndims, dims);
}

Tensor::Tensor(shared_ptr<MixMemory> data,int device_id){
    shape_string_[0]=0;
    descriptor_string_[0]=0;
    this->device_id_=get_device(device_id);
    setup_data(data);
}

Tensor::~Tensor(){
    release();
}

const char* Tensor::descriptor() const{
    char* descriptor_ptr = (char*)descriptor_string_;
    int device_id = device();
    snprintf(descriptor_ptr,sizeof(descriptor_string_),
    "Tensor:%p, %s, CUDA:%d",
    data_.get(),
    shape_string_,
    device_id
    );
    return descriptor_ptr;
}

Tensor& Tensor::compute_shape_string(){
    shape_string_[0] = 0;
    char* buffer = shape_string_;
    size_t buffer_size = sizeof(shape_string_);
    for(int i=0;i<shape_.size();++i){
        int size = 0;
        if(i < shape_.size()-1)
            size = snprintf(buffer,buffer_size,"%d x ",shape_[i]);
        else
            size = snprintf(buffer,buffer_size,"%d",shape_[i]);
        
        buffer += size;
        buffer_size -= size;
    }
    return *this;
}

void Tensor::setup_data(shared_ptr<MixMemory> data){
    data_ = data;
    if(data_ ==nullptr){
        data_ = make_shared<MixMemory>(device_id_);
    }else{
        device_id_ = data_->device_id();
    }

    head_ = DataHead::Init;
    if(data_->cpu()){
        head_ = DataHead::Host;
    }

    if(data_->gpu()){
        head_ = DataHead::Device;
    }
}

Tensor& Tensor::copy_from_gpu(size_t offset,const void* src,size_t num_element,int device_id){
    if(head_ == DataHead::Init)
        to_gpu(false);
    
    size_t offset_location = offset * element_size();
    if(offset_location >= bytes_){
        INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
        return *this;
    }

    size_t copyed_bytes = num_element * element_size();
    size_t remain_bytes = bytes_ - offset_location;
    if(copyed_bytes > remain_bytes){
        INFOE("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
        return *this;
    }

    if(head_ == DataHead::Device){
        int current_device_id = get_device(device_id);
        int gpu_device_id = device();
        if(current_device_id != gpu_device_id){
            checkCudaRuntime(cudaMemcpyPeerAsync(gpu<unsigned char>() + offset_location, gpu_device_id, src, current_device_id, copyed_bytes, stream_));
        }
        else{
            checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
        }
    }else if(head_ == DataHead::Host){
        AutoDevice auto_device_exchange(this->device());
        checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
    }else{
        INFOE("Unsupport head type %d", head_);
    }
    return *this;
}

Tensor& Tensor::release(){
    data_->release_all();
    shape_.clear();
    bytes_ = 0;
    head_ = DataHead::Init;
    return *this;
}

bool Tensor::empty() const{
    return data_->cpu() == nullptr && data_->gpu() == nullptr;
}

int Tensor::count(int start_axis) const{
    if(start_axis >=0 && start_axis < shape_.size()){
        int size = 1;
        for(int i=start_axis;i<shape_.size();++i){
            size *= shape_[i];
        }
        return size;
    }else{
        return 0;
    }
}

Tensor& Tensor::resize(const vector<int>& dims){
    return resize(dims.size(),dims.data());
}

int Tensor::numel() const{
    int value = shape_.empty() ? 0 : 1;
    for(int i=0;i<shape_.size();++i){
        value *= shape_[i];
    }
    return value;
}

Tensor& Tensor::resize_single_dim(int idim,int size){
    Assert(idim >= 0 && idim < shape_.size());

    auto new_shape = shape_;
    new_shape[idim] = size;
    return resize(new_shape);
}

Tensor& Tensor::resize(int ndims,const int* dims){
    vector<int> setup_dims(ndims);
    for(int i=0;i<ndims;++i){
        int dim = dims[i];
        if(dim == -1){
            Assert(ndims == shape_.size());
            dim = shape_[i];
        }
        setup_dims[i]=dim;
    }
    this->shape_ = setup_dims;
    this->adajust_memory_by_update_dims_or_type();
    this->compute_shape_string();
    return *this;
}

Tensor& Tensor::adajust_memory_by_update_dims_or_type(){
    int needed_size = this->numel() * element_size();
    if(needed_size > this->bytes_){
        head_ = DataHead::Init;
    }
    this->bytes_ = needed_size;
    return *this;
}

Tensor& Tensor::synchronize(){
    AutoDevice auto_device_exchange(this->device());
    checkCudaRuntime(cudaStreamSynchronize(stream_));
    return *this;
}

Tensor& Tensor::to_gpu(bool copy){
    if(head_ == DataHead::Device)
        return *this;
    head_ = DataHead::Device;
    data_->gpu(bytes_);
    if (copy && data_->cpu() != nullptr){
        AutoDevice auto_device_exchange(this->device());
        checkCudaRuntime(cudaMemcpyAsync(data_->gpu(),data_->cpu(),bytes_,cudaMemcpyHostToDevice,stream_));
    }
    return *this;
}

Tensor& Tensor::to_cpu(bool copy){
    if(head_ == DataHead::Host)
        return *this;
    
    head_ = DataHead::Host;
    data_->cpu(bytes_);
    if(copy && data_->gpu() != nullptr){
        AutoDevice auto_device_exchange(this->device());
        checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
        checkCudaRuntime(cudaStreamSynchronize(stream_));
    }
    return *this;
}

int Tensor::offset_array(size_t size,const int* index_array) const{
    Assert(size <= shape_.size());
    int value = 0;
    for(int i=0;i<shape_.size();++i){
        if(i<size)
            value += index_array[i];

        if(i+1 < shape_.size())
            value *= shape_[i+1];
    }
    return value;
}

int Tensor::offset_array(const vector<int>& index_array) const{
    return offset_array(index_array.size(),index_array.data());
}

}