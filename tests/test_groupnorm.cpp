// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/groupnorm.h"
#include "testutil.h"

static int test_groupnorm(const ncnn::Mat& a, int group, float eps)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, channels);
    pd.set(2, eps);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    int ret = test_layer<ncnn::GroupNorm>("GroupNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_groupnorm failed a.dims=%d a=(%d %d %d) group=%d eps=%f\n", a.dims, a.w, a.h, a.c, group, eps);
    }

    return ret;
}

static int test_groupnorm_0()
{
    return 0
           || test_groupnorm(RandomMat(6, 4, 2), 1, 0.01f)
           || test_groupnorm(RandomMat(3, 3, 8), 2, 0.002f)
           || test_groupnorm(RandomMat(4, 5, 6), 3, 0.01f)
           || test_groupnorm(RandomMat(5, 6, 12), 4, 0.02f)
           || test_groupnorm(RandomMat(6, 7, 24), 2, 0.001f)
           || test_groupnorm(RandomMat(8, 9, 24), 3, 0.0001f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_groupnorm_0();
}
