#include "cuda_tools.h"

bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

bool check_device_id(int device_id){
    int device_count = -1;
    checkCudaRuntime(cudaGetDeviceCount(&device_count));
    if(device_id < 0 || device_id >= device_count){
        INFOE("Invalid device id: %d, count = %d", device_id, device_count);
        return false;
    }
    return true;
}

AutoDevice::AutoDevice(int device_id){

    cudaGetDevice(&old_);
    checkCudaRuntime(cudaSetDevice(device_id));
}

AutoDevice::~AutoDevice(){
    checkCudaRuntime(cudaSetDevice(old_));
}