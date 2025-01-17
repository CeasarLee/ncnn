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

        self.pool_0 = nn.AvgPool1d(kernel_size=3)
        self.pool_1 = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.pool_2 = nn.AvgPool1d(kernel_size=3, stride=1, padding=(0), ceil_mode=False, count_include_pad=True)
        self.pool_3 = nn.AvgPool1d(kernel_size=5, stride=2, padding=(2), ceil_mode=True, count_include_pad=False)
        self.pool_4 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False, count_include_pad=True)
        self.pool_5 = nn.AvgPool1d(kernel_size=2, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.pool_6 = nn.AvgPool1d(kernel_size=4, stride=1, padding=2, ceil_mode=False, count_include_pad=False)

    def forward(self, x):
        x = self.pool_0(x)
        x = self.pool_1(x)
        x = self.pool_2(x)
        x = self.pool_3(x)
        x = self.pool_4(x)
        x = self.pool_5(x)
        x = self.pool_6(x)
        return x

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12, 128)

    a = net(x)

    # export torchscript
    mod = torch.jit.trace(net, x)
    mod.save("test_nn_AvgPool1d.pt")

    # torchscript to pnnx
    import os
    os.system("../../src/pnnx test_nn_AvgPool1d.pt inputshape=[1,12,128]")

    # ncnn inference
    import test_nn_AvgPool1d_ncnn
    b = test_nn_AvgPool1d_ncnn.test_inference()

    return torch.allclose(a, b, 1e-4, 1e-4)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
