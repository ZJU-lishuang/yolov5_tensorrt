#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include "ilogger.h"
#include <cuda_runtime.h>

#define Assert(op)					 \
    do{                              \
        bool cond = !(!(op));        \
        if(!cond){                   \
            INFOF("Assert failed, " #op);  \
        }                                  \
    }while(false)

#define checkCudaRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)

bool check_runtime(cudaError_t e, const char* call, int iLine, const char *szFile);

bool check_device_id(int device_id);

/* 构造时设置当前gpuid，析构时修改为原来的gpuid */
class AutoDevice{
public:
    AutoDevice(int device_id = 0);
    virtual ~AutoDevice();

private:
    int old_ = -1;
};

#endif