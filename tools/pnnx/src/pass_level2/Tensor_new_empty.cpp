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

#include "pass_level2.h"

namespace pnnx {

class Tensor_new_empty : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0             0 1 input
pnnx.Input              input_1             0 1 size
prim::Constant          op_0                0 1 dtype value=*
prim::Constant          op_1                0 1 layout value=*
prim::Constant          op_2                0 1 device value=*
prim::Constant          op_3                0 1 pin_memory value=*
aten::new_empty         op_4                6 1 input size dtype layout device pin_memory out
pnnx.Output             output              1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.new_empty";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_new_empty, 10)

} // namespace pnnx
