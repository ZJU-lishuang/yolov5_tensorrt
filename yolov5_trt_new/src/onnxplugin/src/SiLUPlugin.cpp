
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

#include "../include/SiLUPlugin.h"
#include "../include/checkMacrosPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::SiLUPluginCreator;
using nvinfer1::plugin::SILU;

static const char* SILU_PLUGIN_VERSION{"1"};
static const char* SILU_PLUGIN_NAME{"SiLU"};
PluginFieldCollection SiLUPluginCreator::mFC{};
std::vector<PluginField> SiLUPluginCreator::mPluginAttributes;

// LeakyReLU {{{
SILU::SILU()
{
}

SILU::SILU(const void* buffer, size_t length)
{
    // const char *d = reinterpret_cast<const char*>(buffer), *a = d;
    // mBatchDim = read<int>(d);
    // ASSERT(d == a + length);
    assert(length==sizeof(input_size_));
    input_size_ = *reinterpret_cast<const int*>(buffer);

}

int SILU::getNbOutputs() const TRT_NOEXCEPT 
{
    return 1;
}

// Dims SILU::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
// {
//     ASSERT(nbInputDims == 1);
//     ASSERT(index == 0);
//     input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
//     return inputs[0];
// }

DimsExprs SILU::getOutputDimensions(
int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) TRT_NOEXCEPT
{
    ASSERT(nbInputs == 1);
    ASSERT(outputIndex == 0);
    // input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    input_size_ = exprBuilder.operation(DimensionOperation::kPROD,
                                *exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *inputs[0].d[2]),
                                *inputs[0].d[3])->getConstantValue();
    // [batch,channel,height,width] batch is not Constant
    // printf("inputs[0].d[0]=%d\n",inputs[0].d[0]->getConstantValue());
    // printf("inputs[0].d[1]=%d\n",inputs[0].d[1]->getConstantValue());
    // printf("inputs[0].d[2]=%d\n",inputs[0].d[2]->getConstantValue());
    // printf("inputs[0].d[3]=%d\n",inputs[0].d[3]->getConstantValue());
    // printf("input_size_=%d\n",input_size_);                        
    return inputs[0];
}

__device__ float Logist_kernel_cpu(float data) { return 1.0f / (1.0f + expf(-data)); };


pluginStatus_t SILU::SiLUInference_cpu(const int n, const float* input, float* output)
{
    printf("SiLUInference_cpu start\n");
    for (int i =0; i < n; i += 1)
    {
        printf("SiLUInference_cpu id=%d\n",i);
        output[i] = input[i] * Logist_kernel_cpu(input[i]);
    }
    return STATUS_SUCCESS;
}

int32_t SILU::enqueue(const PluginTensorDesc* inputDesc,
const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
cudaStream_t stream) TRT_NOEXCEPT
{
    const int batchSize = inputDesc[0].dims.d[0];
    inputDesc[0].dims;
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    pluginStatus_t status = SiLUInference(stream, batchSize*input_size_, inputData, outputData);
    // pluginStatus_t status = SiLUInference_cpu(batchSize*input_size_, (const float*) inputData, (float*) outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

// int SILU::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
// {
//     const void* inputData = inputs[0];
//     void* outputData = outputs[0];
//     pluginStatus_t status = SiLUInference(stream, batchSize*input_size_, inputData, outputData);
//     // pluginStatus_t status = SiLUInference_cpu(batchSize*input_size_, (const float*) inputData, (float*) outputData);
//     ASSERT(status == STATUS_SUCCESS);
//     return status;
// }

size_t SILU::getSerializationSize() const TRT_NOEXCEPT
{
    // mNegSlope, mBatchDim
    // return sizeof(float) + sizeof(int);
    // return sizeof(int);
    return sizeof(input_size_);
}

// Set plugin namespace
void SILU::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
    mPluginNamespace = pluginNamespace;
}

const char* SILU::getPluginNamespace() const TRT_NOEXCEPT
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType SILU::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
{
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
// bool SILU::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
// {
//     return false;
// }

// Return true if plugin can use input that is broadcast across batch without replication.
// bool SILU::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
// {
//     return false;
// }

void SILU::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) TRT_NOEXCEPT
{
    // gLogVerbose << "SiLU configurePlugin\n";
}

// void SILU::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
// {
//     // ASSERT(mBatchDim == 1);
//     // for (int i = 0; i <in->dims.nbDims; ++i)
//     // {
//     //     mBatchDim *= in->dims.d[i];
//     // }
// }

void SILU::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
{
}

// Detach the plugin object from its execution context.
void SILU::detachFromContext() TRT_NOEXCEPT {}

void SILU::serialize(void* buffer) const TRT_NOEXCEPT
{
    // char *d = reinterpret_cast<char*>(buffer), *a = d;
    // write(d, mBatchDim);
    // ASSERT(d == a + getSerializationSize());
    *reinterpret_cast<int*>(buffer)=input_size_;
}

// void SILU::configureWithFormat(
//     const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
// {
//     ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
//     ASSERT(mBatchDim == 1);
//     ASSERT(nbOutputs == 1);
//     for (int i = 0; i < inputDims[0].nbDims; ++i)
//     {
//         mBatchDim *= inputDims[0].d[i];
//     }
// }

// bool SILU::supportsFormat(DataType type, PluginFormat format) const
// {
//     return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
// }

// bool SILU::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT
// {
// //    ASSERT(mBatchDim == 1);
// //    for (int i = 0; i <inOut->dims.nbDims; ++i)
// //    {
// //        mBatchDim *= inOut->dims.d[i];
// //    }
//     return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
// }

bool SILU::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) TRT_NOEXCEPT
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

int SILU::initialize() TRT_NOEXCEPT
{
    return 0;
}

void SILU::terminate() TRT_NOEXCEPT {}

// size_t SILU::getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT
// {
//     return 0;
// }

size_t SILU::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs,
    const PluginTensorDesc* outputs, int32_t nbOutputs) const TRT_NOEXCEPT
{
    return 0;
}

const char* SILU::getPluginType() const TRT_NOEXCEPT
{
    return SILU_PLUGIN_NAME;
}

const char* SILU::getPluginVersion() const TRT_NOEXCEPT
{
    return SILU_PLUGIN_VERSION;
}

void SILU::destroy() TRT_NOEXCEPT
{
    delete this;
}

IPluginV2DynamicExt* SILU::clone() const TRT_NOEXCEPT
{
    SILU* plugin = new SILU();
    plugin->input_size_ = input_size_;
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

SiLUPluginCreator::SiLUPluginCreator()
{
    // mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SiLUPluginCreator::getPluginName() const TRT_NOEXCEPT
{
    return SILU_PLUGIN_NAME;
}

const char* SiLUPluginCreator::getPluginVersion() const TRT_NOEXCEPT
{
    return SILU_PLUGIN_VERSION;
}

const PluginFieldCollection* SiLUPluginCreator::getFieldNames() TRT_NOEXCEPT
{
    return &mFC;
}

IPluginV2DynamicExt* SiLUPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
{
    // const PluginField* fields = fc->fields;
    // ASSERT(fc->nbFields == 1);
    // ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    // negSlope = *(static_cast<const float*>(fields[0].data));

    // return new SILU();
    SILU* obj = new SILU();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* SiLUPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
{
    // This object will be deleted when the network is destroyed, which will
    // call LReluPlugin::destroy()
    // return new SILU(serialData, serialLength);
    SILU* obj = new SILU(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
// LeakReLU }}}
