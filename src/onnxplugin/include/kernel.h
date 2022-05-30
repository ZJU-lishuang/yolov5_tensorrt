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

#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include "cublas_v2.h"
#include "plugin.h"
#include <algorithm>
#include <cassert>
#include <cstdio>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
#define DEBUG_ENABLE 0

#ifndef TRT_RPNLAYER_H
typedef enum
{
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

pluginStatus_t SiLUInference(cudaStream_t stream, int n, const void* input, void* output);

#endif // TRT_RPNLAYER_H
#endif
