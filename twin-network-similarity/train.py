#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(y,distance,batch_size):
    '''定义对比损失函数'''
    tmp=y*torch.square(distance)
    tmp2=(1-y)*torch.square(torch.max((1-distance),0))
    res=torch.sum(tmp+tmp2)/batch_size/2
    return res



















