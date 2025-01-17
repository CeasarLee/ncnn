# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.gn_0 = nn.GroupNorm(num_groups=4, num_channels=12)
        self.gn_1 = nn.GroupNorm(num_groups=12, num_channels=12, eps=1e-2, affine=True)
        self.gn_2 = nn.GroupNorm(num_groups=1, num_channels=12, eps=1e-4, affine=True)

    def forward(self, x):
        x = self.gn_0(x)
        x = self.gn_1(x)
        x = self.gn_2(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 24, 64)

    a0 = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_GroupNorm.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_GroupNorm.pt inputshape=[1,12,24,64]")

    # ncnn inference
    import test_nn_GroupNorm_ncnn
    b0 = test_nn_GroupNorm_ncnn.test_inference()

    return torch.allclose(a0, b0, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
