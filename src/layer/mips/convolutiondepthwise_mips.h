// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTIONDEPTHWISE_MIPS_H
#define LAYER_CONVOLUTIONDEPTHWISE_MIPS_H

#include "convolutiondepthwise.h"

namespace ncnn {

class ConvolutionDepthWise_mips : virtual public ConvolutionDepthWise
{
public:
    ConvolutionDepthWise_mips();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int create_group_ops(const Option& opt);

public:
    Layer* activation;
    std::vector<ncnn::Layer*> group_ops;

    // packing
    Mat weight_data_packed;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTIONDEPTHWISE_MIPS_H
