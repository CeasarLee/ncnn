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

#include "eliminate_output.h"

namespace pnnx {

namespace ncnn {

void eliminate_output(Graph& graph)
{
    for (;;)
    {
        bool need_eliminate = false;

        for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
        {
            Operator* op = graph.ops[i];

            if (op->type != "pnnx.Output")
                continue;

            need_eliminate = true;

            // canonicalize output name
            for (int j = 0; j < (int)op->inputs.size(); j++)
            {
                op->inputs[j]->name = std::string("out") + std::to_string(j);
            }

            for (Operand* r : op->inputs)
            {
                r->remove_consumer(op);
            }

            op->inputs.clear();

            for (Operand* r : op->outputs)
            {
                r->producer = 0;
            }

            op->outputs.clear();

            graph.ops.erase(graph.ops.begin() + i);
            delete op;

            break;
        }

        if (!need_eliminate)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
