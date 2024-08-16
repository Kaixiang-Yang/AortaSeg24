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


import imp
import torch
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
import warnings

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class Masked_SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(Masked_SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None, class_for_batch=None, Miss=True):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if Miss:#Miss=True, label-missing, don't calculate the bg_dice
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        else:#All labeled samples
            if not self.do_bg:
                if self.batch_dice:
                    dc = dc[1:]
                else:
                    dc = dc[:, 1:]

        loss = 1 - dc

        if class_for_batch is None:
            warnings.warn("\nThe batch seem to be label-missing, but don't get class_for_batch!\nso we just dc.mean()")
            loss = loss.mean()
        else:
            if not Miss:
                loss = loss.mean()
            else:
                ToTal = loss.shape[0] * loss.shape[1]
                for batch in range(dc.shape[0]):
                    for c in range(dc.shape[1]):
                        if c+1 in class_for_batch:#[0,1,2,3,4,5,6,7,8,9]
                            continue
                        else:#label-missing, this is masked loss, so we don't neet to think too much, just let the channel be 0
                            loss[batch, c] = 0
                            ToTal -= 1
                loss = loss.sum() / ToTal if ToTal > 0 else loss.sum()

        return loss

class Masked_CrossEntropy(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
        CE_Loss = -1*gt*log(pt)
        Masked_CELoss = Masked * CE_Loss, Masked = [1,1,...,0,1,...], if channel c is missing, then index of c is 0
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
    #深监督函数处已经定义每次送入self.loss的只有一个sample，即b=1

    def __init__(self, apply_nonlin=None, smooth=1e-5, size_average=True):
        super(Masked_CrossEntropy, self).__init__()
        self.apply_nonlin = apply_nonlin #here is softmax,dim=1
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, class_for_batch, Miss=True):#体素水平下的计算
        # 1.softmax
        # 2.log
        # 3.gt赋权重
        # MEloss 在one_hot_key处让某一列为0即可，同时logit某一列加到某一列即可
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:# 这么做的目的是为了保证和target的体素一一对应, 设置m是在实际情况中不知道具体维度
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1) #logit: [N, C, d1*d2*...]
            logit = logit.permute(0, 2, 1).contiguous() #logit: [N, d1*d2*..., C]
            logit = logit.view(-1, logit.size(-1)) #logit: [N*d1*d2*..., C]
        target = torch.squeeze(target, 1) #target: [N, 1, d1, d2,...] -> [N, d1, d2,...]
        target = target.view(-1, 1) #target: [N*d1*d2*..., 1]

        alpha = torch.ones(num_class, 1)
        
        if Miss:
            for ch in range(1, num_class):#10，其实这里没有必要，因为缺失的标签c根本不存在于target中，所以用不到alpha，只需背景的alpha=0即可
                if ch in class_for_batch:
                    continue
                else:
                    alpha[ch, 0] = 0
            alpha[0, 0] = 0 #Miss=True, 标签缺失，所以背景的alpha直接为0
        else:
            pass#alpha=ones

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long() #target: [N*d1*d2*..., 1]

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_() #one_hot_key: [N*d1*d2*..., numclass] 零矩阵
        one_hot_key = one_hot_key.scatter_(1, idx, 1) #按列填充，相应位置类别设为1，one-hot
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            # one_hot_key = torch.clamp(
            #     one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
            logit = torch.clamp(logit, self.smooth, 1.0)
        
        logpt = logit.log()#logpt已经softmax。[N*d1*d2*..., C]

        CELoss = (one_hot_key * logpt).sum(1) + self.smooth #按行求和，得到每个像素点的概率值，[N*d1*d2*..., 1]
        
        alpha = alpha[idx] #alpha: [N*d1*d2*..., 1], 属于哪一类，则相应位置为该类的alpha值，如idx[0]=9,则alpha[0]为第9类的归一化alpha值
        # 这里可能存在一个小bug
        # alpha使用的idx是target，也就是说缺失的label一定不会被赋0(相应位置的alpha值)，因为它们被“错误地”归为了背景
        # 所以这里的背景alpha=0既是为了不计算错误的背景，也为了使相应缺失的channel=0
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * CELoss


        if self.size_average:
            if Miss:
                loss = loss.sum()/(target > 0).sum() if (target > 0).sum() > 0 else loss.mean()
                # loss = loss.mean()#不考虑背景的loss<=>不改变背景相关参数，即默认背景预测正确
            else:
                loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DC_and_CE_loss_Masked(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss_Masked, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        self.Masked_ce = Masked_CrossEntropy(apply_nonlin=softmax_helper, **ce_kwargs)

        self.Masked_dc = Masked_SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.ignore_label = ignore_label


    def forward(self, net_output, target, class_for_batch):
        """
        target must be b, c, x, y(, z) with c=1, in label-missing situation, we put b=1,c=1,d,h,w in self.loss
        :param net_output:
        :param target:
        :param index of classes for batch: [10,]
        :return:

        the basic idea is that, all-label img use normal loss, and label-missing img use masked or ME loss
        here, we first make masked loss as the file name :-)20221103
        """

        if len(class_for_batch.shape) > 1: #[1,10]
            class_for_batch = class_for_batch[0]
            if class_for_batch[0] < 0:#不知道有的class_for_batch第一个数为什么是-1(在data_loading已经解决)
                class_for_batch = class_for_batch[1:]
            if len(class_for_batch[0]) > 9:#全标签
                MISS = False
            else:#标签缺失
                MISS = True
        else:#[10,]
            if class_for_batch[0] < 0:#不知道第一个数为什么是-1
                class_for_batch = class_for_batch[1:]
            if len(class_for_batch) > 9:#全标签
                MISS = False
            else:#标签缺失
                MISS = True

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.Masked_dc(net_output, target, loss_mask=mask, class_for_batch=class_for_batch, Miss=MISS) if self.weight_dice != 0 else 0

        ce_loss = self.Masked_ce(net_output, target, class_for_batch=class_for_batch, Miss=MISS) if self.weight_ce != 0 else 0

        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result
