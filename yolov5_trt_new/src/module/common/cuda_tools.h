#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include "ilogger.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <stdlib.h>

using namespace nvinfer1;

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

class Logger : public ILogger {
public:
    virtual void log(Severity severity, const char* msg) noexcept override {

        if (severity == Severity::kINTERNAL_ERROR) {
            INFOE("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }else if (severity == Severity::kERROR) {
            INFOE("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kWARNING) {
            INFOW("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kINFO) {
            INFOD("NVInfer: %s", msg);
        }
        else {
            INFOD("%s", msg);
        }
    }
};

static Logger gLogger;

template<typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
    if (ptr) ptr->destroy();
}

/* 构造时设置当前gpuid，析构时修改为原来的gpuid */
class AutoDevice{
public:
    AutoDevice(int device_id = 0);
    virtual ~AutoDevice();

private:
    int old_ = -1;
};

#endif