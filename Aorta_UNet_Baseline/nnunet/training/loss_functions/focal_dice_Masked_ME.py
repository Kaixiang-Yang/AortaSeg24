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

'''
V3 20221121
不使用if等条件判断，以免影响loss计算
'''
import torch
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np


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

    def forward(self, x, y, loss_mask=None, class_weight=None):
        shp_x = x.shape
        C = shp_x[1]
        
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

        if not self.do_bg:
            class_weight = class_weight[1:]
            if self.batch_dice:#MS: have set batch_sice=False
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        loss = 1 - dc

        if class_weight is not None:
            loss *= class_weight
            loss = loss.sum()
            loss /= class_weight.sum()
        else:
            loss = loss.sum()
            loss /= C
        
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

    def forward(self, logit, target, class_weight=None):#体素水平下的计算
        # 1.softmax
        # 2.log
        # 3.gt赋权重

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

        
        idx = target.cpu().long() #target: [N*d1*d2*..., 1]

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_() #one_hot_key: [N*d1*d2*..., numclass] 零矩阵
        one_hot_key = one_hot_key.scatter_(1, idx, 1) #按列填充，相应位置类别设为1，one-hot
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            logit = torch.clamp(logit, self.smooth, 1.0)
        
        logpt = logit.log()

        CELoss = (one_hot_key * logpt).sum(1) + self.smooth #按行求和，得到每个像素点的概率值，[N*d1*d2*..., 1]
        
        if class_weight is None:
            class_weight = torch.ones(num_class)
        class_weight = class_weight.to(logit.device)
        class_weight = class_weight[idx] #alpha: [N*d1*d2*..., 1], 属于哪一类，则相应位置为该类的alpha值，如idx[0]=9,则alpha[0]为第9类的归一化alpha值
        
        class_weight = torch.squeeze(class_weight)
        loss = -1 * class_weight * CELoss


        if self.size_average:
            loss = loss.sum()/((loss>0).sum()+1) #softmax后，交叉熵不太可能为0，除非被class_weight赋值0
        else:
            loss = loss.sum()

        return loss

class Masked_TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, alpha_Tver=0.5, weight_tver=None, smooth=1.):
        super(Masked_TverskyLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.alpha = alpha_Tver
        self.smooth = smooth
        self.weight_tver = weight_tver

    def forward(self, x,y, loss_mask=None, class_weight=None):
        shp_x = x.shape
        C = shp_x[1]
        tver_weight = self.weight_tver

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        if tver_weight is None:
            tver_weight = torch.ones(C)
        elif isinstance(tver_weight, (list, np.ndarray)):
            assert len(tver_weight) == C
            tver_weight = torch.FloatTensor(tver_weight).view(C)
        elif isinstance(tver_weight, float):
            tver_weight = torch.ones(C)
            tver_weight = tver_weight * (1 - self.weight_tver)
            tver_weight[0] = self.weight_tver
        else:
            raise TypeError('Not support alpha type')
                    

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x) # softmax here

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth

        Tversky = nominator / (denominator + 1e-8)

        if not self.do_bg:
            tver_weight = tver_weight[1:]
            class_weight = class_weight[1:]
            if self.batch_dice:
                Tversky = Tversky[1:]
            else:
                Tversky = Tversky[:, 1:]
        
        loss = 1 - Tversky

        if tver_weight.device != loss.device:
            tver_weight = tver_weight.to(loss.device)

        if class_weight is not None:
            loss *= class_weight
            loss *= tver_weight
            loss = loss.sum()
            loss /= class_weight.sum()
        else:
            loss *= self.weight_tver
            loss = loss.sum()
            loss /= C

        return loss

class Masked_FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
        Masked_Focal_Loss = -1*class_weight*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
    #深监督函数处已经定义每次送入self.loss的只有一个sample，即b=1

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(Masked_FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin #here is softmax,dim=1
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, class_weight=None):#体素水平下的计算
        # 1.softmax
        # 2.log
        # 3.gt赋权重

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

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha #/ alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long() #target: [N*d1*d2*..., 1]

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_() #one_hot_key: [N*d1*d2*..., numclass] 零矩阵
        one_hot_key = one_hot_key.scatter_(1, idx, 1) #按列填充，相应位置类别设为1，one-hot
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            logit = torch.clamp(logit, self.smooth, 1.0)
        
        #Focal_Loss= -1*alpha*(1-pt)*log(pt)
        #Masked_Focal_Loss = -1*class_weight*alpha*(1-pt)^2*log(pt)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        
        if class_weight is None:
            class_weight = torch.ones(num_class)
        class_weight = class_weight.to(logit.device)
        class_weight = class_weight[idx] #alpha: [N*d1*d2*..., 1], 属于哪一类，则相应位置为该类的alpha值，如idx[0]=9,则alpha[0]为第9类的归一化alpha值
        class_weight = torch.squeeze(class_weight)

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)

        loss = -1 * class_weight * alpha * torch.pow((1-pt), gamma) * logpt


        if self.size_average:
            loss = loss.sum()/((loss>0).sum()+1) #softmax后，交叉熵不太可能为0，除非被class_weight赋值0
        else:
            loss = loss.sum()

        return loss

class Masked_TverskyLoss_Ex(nn.Module):#Exclution Tver
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, alpha_Tver=0.5, weight_tver=None, smooth=1.):
        super(Masked_TverskyLoss_Ex, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.alpha = alpha_Tver
        self.smooth = smooth
        self.weight_tver = weight_tver

    def forward(self, x, y, loss_mask=None, class_weight=None):
        shp_x = x.shape
        C = shp_x[1]
        tver_weight = self.weight_tver

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        
        if tver_weight is None:
            tver_weight = torch.ones(C)
        elif isinstance(tver_weight, (list, np.ndarray)):
            assert len(tver_weight) == C
            tver_weight = torch.FloatTensor(tver_weight).view(C)
        elif isinstance(tver_weight, float):
            tver_weight = torch.ones(C)
            tver_weight = tver_weight * (1 - self.weight_tver)
            tver_weight[0] = self.weight_tver
        else:
            raise TypeError('Not support alpha type')
                    

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x) # softmax here

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = tp + self.smooth
        denominator = tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth

        Tversky = nominator / (denominator + 1e-8)

        if not self.do_bg:
            tver_weight = tver_weight[1:]
            class_weight = class_weight[1:]
            if self.batch_dice:
                Tversky = Tversky[1:]
            else:
                Tversky = Tversky[:, 1:]
        
        loss = Tversky

        if tver_weight.device != loss.device:
            tver_weight = tver_weight.to(loss.device)

        if class_weight is not None:
            loss *= class_weight
            loss *= tver_weight
            loss = loss.sum()
            loss /= class_weight.sum()
        else:
            loss *= self.weight_tver
            loss = loss.sum()
            loss /= C

        return loss

class Masked_FocalLoss_Ex(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
        Masked_Focal_Loss = -1*class_weight*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
    #深监督函数处已经定义每次送入self.loss的只有一个sample，即b=1

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(Masked_FocalLoss_Ex, self).__init__()
        self.apply_nonlin = apply_nonlin #here is softmax,dim=1
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target, class_weight=None):#体素水平下的计算
        # 1.softmax
        # 2.log
        # 3.gt赋权重

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

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha #/ alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long() #target: [N*d1*d2*..., 1]

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_() #one_hot_key: [N*d1*d2*..., numclass] 零矩阵
        one_hot_key = one_hot_key.scatter_(1, idx, 1) #按列填充，相应位置类别设为1，one-hot
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            logit = torch.clamp(logit, self.smooth, 1.0)
        
        #Focal_Loss= -1*alpha*(1-pt)*log(pt)
        #Masked_Focal_Loss = -1*class_weight*alpha*(1-pt)^2*log(pt)
        #Exclution_Focal_Loss = 1*class_weight*alpha*(pt)^2*log(pt+1)

        pt = (one_hot_key * logit).sum(1) + self.smooth + 1
        logpt = pt.log()
        
        if class_weight is None:
            class_weight = torch.ones(num_class)
        class_weight = class_weight.to(logit.device)
        class_weight = class_weight[idx] #alpha: [N*d1*d2*..., 1], 属于哪一类，则相应位置为该类的alpha值，如idx[0]=9,则alpha[0]为第9类的归一化alpha值
        class_weight = torch.squeeze(class_weight)

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)

        loss = 1 * class_weight * alpha * torch.pow(pt, gamma) * logpt


        if self.size_average:
            loss = loss.sum()/((loss>0).sum()+1) #softmax后，交叉熵不太可能为0，除非被class_weight赋值0
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

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None
        
        class_weight = torch.zeros(net_output.shape[1]).to(net_output.device)
        for i in range(len(class_weight)):
            if i in class_for_batch:
                class_weight[i] = 1
        if len(class_for_batch) < net_output.size(1):
            class_weight[0] = 0

        dc_loss = self.Masked_dc(net_output, target, loss_mask=mask, class_weight=class_weight) if self.weight_dice != 0 else 0

        ce_loss = self.Masked_ce(net_output, target, class_weight=class_weight) if self.weight_ce != 0 else 0

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

class Focal_and_Tversky_loss_ME(nn.Module):
    def __init__(self, focal_loss_kwargs, soft_tversky_kwargs, aggregate="sum", square_dice=False, weight_focal=1, weight_dice=1,
                 log_dice=False, ignore_label=None, Masked=True, ME=False, weight_ex=2):
        """
        CAREFUL. Weights for Focal and Tversky do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_focal:
        :param weight_dice:
        """
        super(Focal_and_Tversky_loss_ME, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            focal_loss_kwargs['reduction'] = 'none'
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.aggregate = aggregate
        self.log_dice = log_dice
        self.weight_ex = weight_ex

        self.fc = Masked_FocalLoss(apply_nonlin=nn.Softmax(dim=1), **focal_loss_kwargs)
        self.tver = Masked_TverskyLoss(apply_nonlin=nn.Softmax(dim=1), **soft_tversky_kwargs)

        self.ex_fc = Masked_FocalLoss_Ex(apply_nonlin=nn.Softmax(dim=1), **focal_loss_kwargs)
        self.ex_tver = Masked_TverskyLoss_Ex(apply_nonlin=nn.Softmax(dim=1), **soft_tversky_kwargs)

        self.ignore_label = ignore_label

        self.MaskedLoss = Masked

        self.ME = ME

    def forward(self, net_output0, target, class_for_batch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        net_output = net_output0.clone()#ME
        net_output_ex = net_output0.clone()
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.MaskedLoss:
            class_weight = torch.zeros(net_output.shape[1]).to(net_output.device)
            for i in range(len(class_weight)):
                if i in class_for_batch:
                    class_weight[i] = 1
            if len(class_for_batch) < net_output.size(1):
                class_weight[0] = 0

                # # marginal loss and exclusion loss
                if self.ME:
                    class_weight[0] = 1
                    # marginal loss: just add other missing channel to bg, don't need other things
                    for lex in range(1, len(class_weight)):
                        if class_weight[lex] < 1:
                            net_output[:,0,...] += net_output[:,lex,...]
                            # net_output[:,lex,...] -= net_output[:,lex,...]
                    # for lex in range(1, len(class_weight)):
                    #     if class_weight[lex] < 1:
                    #         net_output[:,lex,...] += net_output[:,0,...]

                    # exclusion loss: use marginal probablity to cal not ori output
                    net_output_ex = net_output.clone()
                    for lex in range(1, len(class_weight)):
                        if class_weight[lex] < 1: #missing organ's channel
                            net_output_ex[:,lex,...] = torch.ones_like(net_output_ex[:,lex,...], requires_grad=True) - net_output_ex[:,0,...]#1-p(0)
                        else: #exist organ's channel
                            net_output_ex[:,lex,...] = torch.ones_like(net_output_ex[:,lex,...], requires_grad=True) - net_output_ex[:,lex,...]
                    net_output_ex[:,0,...] = torch.ones_like(net_output_ex[:,0,...], requires_grad=True) - net_output_ex[:,0,...]

        else:
            class_weight = torch.ones(net_output.shape[1]).to(net_output.device)


        tver_loss = self.tver(net_output, target, loss_mask=mask, class_weight=class_weight) if self.weight_dice != 0 else 0
        if self.log_dice:
            tver_loss = -torch.log(-tver_loss)

        fc_loss = self.fc(net_output, target, class_weight=class_weight) if self.weight_focal != 0 else 0
        if self.ignore_label is not None:
            fc_loss *= mask[:, 0]
            fc_loss = fc_loss.sum() / mask.sum()

        if self.ME:
            ex_tver_loss = self.ex_tver(net_output_ex, target, loss_mask=mask, class_weight=class_weight) if self.weight_dice != 0 else 0
            ex_focal_loss = self.ex_fc(net_output_ex, target, class_weight=class_weight) if self.weight_focal != 0 else 0

        if self.aggregate == "sum":
            if not self.ME:
                result = self.weight_focal * fc_loss + self.weight_dice * tver_loss
            else:
                result = self.weight_focal * fc_loss + self.weight_dice * tver_loss + self.weight_ex * (ex_tver_loss + ex_focal_loss)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DC_and_FC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, fc_kwargs, aggregate="sum", square_dice=False, weight_fc=1, weight_dice=1,
                 log_dice=False, ignore_label=None, Masked=True):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_FC_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            fc_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_fc = weight_fc
        self.aggregate = aggregate
        self.fc = Masked_FocalLoss(apply_nonlin=nn.Softmax(dim=1), **fc_kwargs)
        self.dc = Masked_SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

        self.ignore_label = ignore_label

        self.MaskedLoss = Masked


    def forward(self, net_output, target, class_for_batch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.MaskedLoss:
            class_weight = torch.zeros(net_output.shape[1]).to(net_output.device)
            for i in range(len(class_weight)):
                if i in class_for_batch:
                    class_weight[i] = 1
            if len(class_for_batch) < net_output.size(1):
                class_weight[0] = 0
        else:
            class_weight = torch.ones(net_output.shape[1]).to(net_output.device)

        dc_loss = self.dc(net_output, target, loss_mask=mask, class_weight=class_weight) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        fc_loss = self.fc(net_output, target, class_weight=class_weight) if self.weight_fc != 0 else 0
        if self.ignore_label is not None:
            fc_loss *= mask[:, 0]
            fc_loss = fc_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_fc * fc_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class TVER_and_CE_loss(nn.Module):
    def __init__(self, tver_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_tver=1,
                 log_dice=False, ignore_label=None, Masked=True):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(TVER_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_tver = weight_tver
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        self.ce = Masked_CrossEntropy(apply_nonlin=softmax_helper, **ce_kwargs)
        self.tver = Masked_TverskyLoss(apply_nonlin=nn.Softmax(dim=1), **tver_kwargs)

        self.ignore_label = ignore_label

        self.MaskedLoss = Masked


    def forward(self, net_output, target, class_for_batch):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.MaskedLoss:
            class_weight = torch.zeros(net_output.shape[1]).to(net_output.device)
            for i in range(len(class_weight)):
                if i in class_for_batch:
                    class_weight[i] = 1
            if len(class_for_batch) < net_output.size(1):
                class_weight[0] = 0
        else:
            class_weight = torch.ones(net_output.shape[1]).to(net_output.device)

        tver_loss = self.tver(net_output, target, loss_mask=mask, class_weight=class_weight) if self.weight_tver != 0 else 0
        if self.log_dice:
            tver_loss = -torch.log(-tver_loss)

        ce_loss = self.ce(net_output, target, class_weight=class_weight) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_tver * tver_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result
