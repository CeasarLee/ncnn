
// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void pooling2x2s2_max_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;

            for (; j < outw; j++)
            {
                __m128 _r00 = _mm_loadu_ps(r0);
                __m128 _r01 = _mm_loadu_ps(r0 + 4);
                __m128 _r10 = _mm_loadu_ps(r1);
                __m128 _r11 = _mm_loadu_ps(r1 + 4);

                __m128 _max0 = _mm_max_ps(_r00, _r01);
                __m128 _max1 = _mm_max_ps(_r10, _r11);
                __m128 _max = _mm_max_ps(_max0, _max1);

                _mm_storeu_ps(outptr, _max);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
