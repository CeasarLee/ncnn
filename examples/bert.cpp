//
// Created by kairuli on 2021/12/28.
//

// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "net.h"
#include "layer.h"
#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <iostream>



static std::vector<std::vector<int> > parse_calibartion_txt(const char *s)
{
    std::vector<std::vector<int> > cab;
//    std::map<std::string, int> lll;
    FILE * fp = fopen(s, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", s);
    }
    std::vector<int> seq;
    char line[1024];
    while (!feof(fp))
    {
        char* ss = fgets(line, 1024, fp);
        if (!ss)
        {
            break;
        }

        char* word = strtok(ss, " ");
        while (word != NULL)
        {
//            printf("%s ", word);
            seq.push_back(std::atoi(word));
            word = strtok(NULL, " ");
        }
        cab.push_back(seq);
        seq.clear();
    }
    fclose(fp);
    return cab;
}

static int bert_infer(std::vector<float>& cls_scores)
{

//    char* vocal_file = "test.txt";
    std::vector<std::vector<int>> cab_set;
    cab_set = parse_calibartion_txt("calibration.txt");
//    printf("cab size %s ", cab_set.size());
    ncnn::Net bert;

    bert.opt.use_vulkan_compute = false;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    bert.load_param("new_bert.ncnn.param");
    bert.load_model("new_bert.ncnn.bin");

    std::vector<int> setence = cab_set[0];

    ncnn::Mat input_seq = ncnn::Mat((int)setence.size());
    for (int i=0; i<input_seq.total(); i++){
        input_seq[i] = setence[i];
    }

    float *ptr = (float*) input_seq;
    for (int i=0; i<input_seq.cstep; i++){
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;

    ncnn::Extractor ex = bert.create_extractor();
    ex.input("in0", input_seq);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::cout<< out.dims << " " << out.w << " " << out.h << std::endl;

//    std::vector<char, int> vocab;
//    vocab = read_vocab("vocab.txt");

//    std::cout << input_seq.w << std::endl;
//    ncnn::Mat in = ncnn::Mat::fro
//    m_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 223, 223);

//    const float mean_vals[3] = {104.f, 117.f, 123.f};
//    in.substract_mean_normalize(mean_vals, 0);
//
//    ncnn::Extractor ex = bert.create_extractor();
////
//    ex.input("in0", input_seq);
////
//    ncnn::Mat out;
//    ex.extract("out0", out);
//    std::cout << out.shape << std::endl;
//
//    cls_scores.resize(out.w);
//    for (int j = 0; j < out.w; j++)
//    {
//        cls_scores[j] = out[j];
//    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
//    if (argc != 2)
//    {
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        return -1;
//    }

//    const char* imagepath = argv[1];

//    cv::Mat m = cv::imread(imagepath, 1);
//    if (m.empty())
//    {
//        fprintf(stderr, "cv::imread %s failed\n", imagepath);
//        return -1;
//    }

//    ncnn::Mat input_sentence = ncnn::Mat(int w=1, int h=5)
    std::vector<float> cls_scores;
    bert_infer(cls_scores);

//    print_topk(cls_scores, 3);

    return 0;
}
//
// Created by kairuli on 2021/12/29.
//

