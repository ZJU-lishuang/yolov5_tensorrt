/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_SILU_PLUGIN_H
#define TRT_SILU_PLUGIN_H
#include "NvInfer.h"
#include "kernel.h"
// #include "plugin.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class SILU : public IPluginV2IOExt
{
public:
    SILU();

    SILU(const void* buffer, size_t length);

    ~SILU() override = default;

    int getNbOutputs() const TRT_NOEXCEPT override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

    int initialize() TRT_NOEXCEPT override;

    void terminate() TRT_NOEXCEPT override;

    size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override;

    int enqueue(
        int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

    size_t getSerializationSize() const TRT_NOEXCEPT override;

    void serialize(void* buffer) const TRT_NOEXCEPT override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override;

    const char* getPluginType() const TRT_NOEXCEPT override;

    const char* getPluginVersion() const TRT_NOEXCEPT override;

    void destroy() TRT_NOEXCEPT override;

    IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

    const char* getPluginNamespace() const TRT_NOEXCEPT override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

    void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

    void detachFromContext() TRT_NOEXCEPT override;

    int input_size_;

private:
    const char* mPluginNamespace;
    // int mBatchDim;
    pluginStatus_t SiLUInference_cpu(const int n, const float* input, float* output);

};

class SiLUPluginCreator : public IPluginCreator
{
public:
    SiLUPluginCreator();

    ~SiLUPluginCreator() override = default;

    const char* getPluginName() const TRT_NOEXCEPT override;

    const char* getPluginVersion() const TRT_NOEXCEPT override;

    const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

    void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const TRT_NOEXCEPT override
    {
        return mNamespace.c_str();
    }


private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(SiLUPluginCreator);

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SILU_PLUGIN_H
