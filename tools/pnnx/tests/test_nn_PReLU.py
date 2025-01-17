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

        self.prelu_0 = nn.PReLU(num_parameters=12)
        self.prelu_1 = nn.PReLU(num_parameters=1, init=0.12)

    def forward(self, x, y, z, w):
        x = self.prelu_0(x)
        x = self.prelu_1(x)

        y = self.prelu_0(y)
        y = self.prelu_1(y)

        z = self.prelu_0(z)
        z = self.prelu_1(z)

        w = self.prelu_0(w)
        w = self.prelu_1(w)
        return x, y, z, w

def test():
    net = Model()
    net.eval()

    torch.manual_seed(0)
    x = torch.rand(1, 12)
    y = torch.rand(1, 12, 64)
    z = torch.rand(1, 12, 24, 64)
    w = torch.rand(1, 12, 24, 32, 64)

    a0, a1, a2, a3 = net(x, y, z, w)

    # export torchscript
    mod = torch.jit.trace(net, (x, y, z, w))
    mod.save("test_nn_PReLU.pt")

    # torchscript to pnnx
    import os
    os.system("../src/pnnx test_nn_PReLU.pt inputshape=[1,12],[1,12,64],[1,12,24,64],[1,12,24,32,64]")

    # pnnx inference
    import test_nn_PReLU_pnnx
    b0, b1, b2, b3 = test_nn_PReLU_pnnx.test_inference()

    return torch.equal(a0, b0) and torch.equal(a1, b1) and torch.equal(a2, b2) and torch.equal(a3, b3)

if __name__ == "__main__":
    if test():
        exit(0)
    else:
        exit(1)
