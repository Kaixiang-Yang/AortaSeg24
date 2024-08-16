#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class MultipleOutputLoss2_MS(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2_MS, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y, class_for_batch=torch.arange(10)):#class index接入口
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        class_for_batch = maybe_to_torch(class_for_batch)
        if torch.cuda.is_available():
            class_for_batch = to_cuda(class_for_batch)

        L = 0

        for batch in range(x[0].shape[0]):#x[0]:[BCDHW]ori_res
            l = weights[0] * self.loss(x[0][batch].unsqueeze(dim=0), y[0][batch].unsqueeze(dim=0), class_for_batch[batch])
            for i in range(1, len(weights)):
                if weights[i] != 0:
                    l += weights[i] * self.loss(x[i][batch].unsqueeze(dim=0), y[i][batch].unsqueeze(dim=0), class_for_batch[batch])

            L += l
        return L/x[0].shape[0]
