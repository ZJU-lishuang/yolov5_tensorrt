#ifndef TRT_TENSOR_H
#define TRT_TENSOR_H

#include <string>
#include <memory>

#define CURRENT_DEVICE_ID   -1

struct CUstream_st;
typedef CUstream_st CUStreamRaw;

typedef CUStreamRaw* CUStream;
namespace TRT{

enum class DataHead:int {
    Init = 0,
    Device = 1,
    Host = 2
};

// cpu和gpu混合内存管理
class MixMemory{
public:
    MixMemory(int device_id = CURRENT_DEVICE_ID);
    MixMemory(void* cpu,size_t cpu_size,void* gpu,size_t gpu_size);
    // 虚函数
    virtual ~MixMemory();

    void* gpu(size_t size);
    void* cpu(size_t size);
    void release_gpu();
    void release_cpu();
    void release_all();

    inline bool owner_gpu() const{return owner_gpu_;}
    inline bool owner_cpu() const{return owner_cpu_;}
    inline size_t cpu_size() const{return cpu_size_;}
    inline size_t gpu_size() const{return gpu_size_;}
    inline int device_id() const{return device_id_;}
    inline void* gpu() const{return gpu_;}
    inline void* cpu() const{return cpu_;}

    void reference_data(void* cpu,size_t cpu_size,void* gpu,size_t gpu_size);

private:
    void* cpu_ = nullptr;
    size_t cpu_size_ = 0;
    bool owner_cpu_ = true;
    
    int device_id_ = 0;
    void* gpu_ = nullptr;
    size_t gpu_size_ = 0;
    bool owner_gpu_ = true;
};


}

#endif