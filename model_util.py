# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:38:03 2022

@author: knight
"""

# 专门存放模型和网络结构与模块相关的类与函数

import os
import torch
from torch import nn
import torchvision
import cv2
import numpy as np
import math

# 用于特征向量l2归一化的函数
def l2_normalize(x):
    l2 = torch.norm(x, 2, dim=1)
    return torch.transpose(torch.transpose(x,0,1) / l2,0,1)

# 3.26为了triplet的损失函数训练而重新编写的具有通用性的四个模型版本
class L2Net(nn.Module):
    def __init__(self, kernel_channels, kernel_size1, kernel_size2, kernel_size3, BN_affine=False, BN_track=True):
        super(L2Net, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN1 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(kernel_channels, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN2 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(kernel_channels, kernel_channels*2, kernel_size1, padding=kernel_size1//2, stride=2)
        self.BN3 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(kernel_channels*2, kernel_channels*2, kernel_size2, padding=kernel_size2//2)
        self.BN4 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(kernel_channels*2, kernel_channels*4, kernel_size2, padding=kernel_size2//2, stride=2)
        self.BN5 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(kernel_channels*4, kernel_channels*4, kernel_size3, padding=kernel_size3//2)
        self.BN6 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.BN7 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.out_channels = kernel_channels * 4
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(out1)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.BN7(self.conv7(out0))
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1, self.out_channels))
        #return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

class L2AttentionNetv1(nn.Module):
    def __init__(self, kernel_channels, kernel_size1, kernel_size2, kernel_size3, BN_affine=False, BN_track=True, channel_reduction_ratio=8):
        super(L2AttentionNetv1, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1_1 = CBAM(kernel_channels, 7, self.channel_reduction_ratio)
        self.CBAM1 = CBAM(kernel_channels, 7, self.channel_reduction_ratio)
        self.CBAM2_1 = CBAM(kernel_channels*2, 3, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(kernel_channels*2, 3, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(kernel_channels*4, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(kernel_channels*4, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(kernel_channels*4, kernel_channels*4, 1), 
                                         nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN1 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(kernel_channels, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN2 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(kernel_channels, kernel_channels*2, kernel_size1, padding=kernel_size1//2, stride=2)
        self.BN3 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(kernel_channels*2, kernel_channels*2, kernel_size2, padding=kernel_size2//2)
        self.BN4 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(kernel_channels*2, kernel_channels*4, kernel_size2, padding=kernel_size2//2, stride=2)
        self.BN5 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(kernel_channels*4, kernel_channels*4, kernel_size3, padding=kernel_size3//2)
        self.BN6 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.BN7 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.out_channels = kernel_channels*4
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.CBAM1_1(self.relu(out1))
        x = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.CBAM2_1(self.relu(self.BN3(self.conv3(x))))
        x = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.relu(self.BN7(self.conv7(out0)))
        #out2 = self.transformer((self.channel_attention7(out2)) * out2)
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1, self.out_channels))
        #return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

class L2FusionNetv3(nn.Module):
    def __init__(self, kernel_channels, kernel_size1, kernel_size2, kernel_size3, out_channels, BN_affine=False, BN_track=True, channel_reduction_ratio=8):
        super(L2FusionNetv3, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN1 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(kernel_channels, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN2 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(kernel_channels, kernel_channels*2, kernel_size1, padding=kernel_size1//2, stride=2)
        self.BN3 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(kernel_channels*2, kernel_channels*2, kernel_size2, padding=kernel_size2//2)
        self.BN4 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(kernel_channels*2, kernel_channels*4, kernel_size2, padding=kernel_size2//2, stride=2)
        self.BN5 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(kernel_channels*4, kernel_channels*4, kernel_size3, padding=kernel_size3//2)
        self.BN6 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.BN7 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.feature_compress = nn.Sequential(nn.Conv2d(kernel_channels*11, out_channels, 1),
                                              nn.BatchNorm2d(out_channels, affine=BN_affine, track_running_stats=BN_track))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.downsampling1 = nn.Conv2d(kernel_channels, kernel_channels, 32)
        self.downsampling2 = nn.Conv2d(kernel_channels*2, kernel_channels*2, 16)
        self.downsampling3 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.channel_attention = ChannelAttentionModule(kernel_channels*11, channel_reduction_ratio)
        self.input_size = 32
        self.out_channels = out_channels
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(y2)))
        y3 = self.relu(self.BN6(self.conv6(x)))
        y4 = self.relu(self.BN7(self.conv7(y3)))
        y = torch.cat([y4, self.downsampling3(y3), self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        #y = torch.cat([y4, self.avg_pool(y3), self.avg_pool(y2), self.avg_pool(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y)) * y)
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1, self.out_channels))
        #return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

class DenseAttentionNetv2(nn.Module):
    def __init__(self, kernel_channels, kernel_size1, kernel_size2, kernel_size3, out_channels, BN_affine=False, BN_track=True, channel_reduction_ratio=8):
        super(DenseAttentionNetv2, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1_1 = CBAM(kernel_channels, 7, self.channel_reduction_ratio)
        self.CBAM1 = CBAM(kernel_channels, 7, self.channel_reduction_ratio)
        self.CBAM2_1 = CBAM(kernel_channels*2, 3, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(kernel_channels*2, 3, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(kernel_channels*4, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(kernel_channels*4, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(kernel_channels*4, kernel_channels*4, 1), 
                                         nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN1 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(kernel_channels, kernel_channels, kernel_size1, padding=kernel_size1//2)
        self.BN2 = nn.BatchNorm2d(kernel_channels, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(kernel_channels, kernel_channels*2, kernel_size1, padding=kernel_size1//2, stride=2)
        self.BN3 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(kernel_channels*2, kernel_channels*2, kernel_size2, padding=kernel_size2//2)
        self.BN4 = nn.BatchNorm2d(kernel_channels*2, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(kernel_channels*2, kernel_channels*4, kernel_size2, padding=kernel_size2//2, stride=2)
        self.BN5 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(kernel_channels*4, kernel_channels*4, kernel_size3, padding=kernel_size3//2)
        self.BN6 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.BN7 = nn.BatchNorm2d(kernel_channels*4, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.feature_compress = nn.Sequential(nn.Conv2d(kernel_channels*11, out_channels, 1),
                                              nn.BatchNorm2d(out_channels, affine=BN_affine, track_running_stats=BN_track))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.downsampling1 = nn.Conv2d(kernel_channels, kernel_channels, 32)
        self.downsampling2 = nn.Conv2d(kernel_channels*2, kernel_channels*2, 16)
        self.downsampling3 = nn.Conv2d(kernel_channels*4, kernel_channels*4, 8)
        self.channel_attention = ChannelAttentionModule(kernel_channels*11, channel_reduction_ratio)
        self.input_size = 32
        self.out_channels = out_channels
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.CBAM1_1(self.relu(out1))
        y1 = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.CBAM2_1(self.relu(self.BN3(self.conv3(y1))))
        y2 = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(y2)))
        y3 = self.relu(self.BN6(self.conv6(x)))
        y4 = self.relu(self.BN7(self.conv7(y3)))
        y = torch.cat([y4, self.downsampling3(y3), self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        #y = torch.cat([y4, self.avg_pool(y3), self.avg_pool(y2), self.avg_pool(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y)) * y)
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1, self.out_channels))
        #return y3, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1, self.out_channels)), l2_normalize(out3.view(-1, self.out_channels))

class L2Net128(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(L2Net128, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.BN7 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        x = self.relu(self.BN1(self.conv1(x)))
        out1 = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(out1)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.BN7(self.conv7(out0))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

# feature fusion v1
class L2FusionNetv1128(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(L2FusionNetv1128, self).__init__()
        self.relu = nn.ReLU()
        self.input_size = 32
        self.block1_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track))
        self.block1_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, 8),
            nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track))
        self.lrn = nn.LocalResponseNorm(5)
    
    def forward(self, x):
        out1 = self.block1_1(x)
        x1_1 = self.relu(out1)
        x1_2 = self.block1_2(x1_1)
        y1 = torch.cat([x1_2, x1_1], dim=1)
        x2_1 = self.downsampling1(y1)
        x2_2 = self.block2(x2_1)
        y2 = torch.cat([x2_2, x2_1], dim=1)
        x3_1 = self.downsampling2(y2)
        out2 = self.block3(x3_1)
        out3 = self.lrn(self.relu(out2))
        return x3_1, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

# feature fusion v2
class L2FusionNetv2256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(L2FusionNetv2256, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.BN7 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.feature_compress = nn.Conv2d(32+64+128+128, 256, 1)
        self.BN8 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.relu = nn.ReLU()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.BN7(self.conv7(out0))
        out2 = torch.cat([self.relu(y3), self.maxpooling(out0), self.maxpooling(y2), self.maxpooling(y1)], dim=1)
        out2 = self.BN8(self.feature_compress(out2))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# feature fusion v3
class L2FusionNetv3256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True, channel_reduction_ratio=8):
        super(L2FusionNetv3256, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.BN7 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.feature_compress = nn.Conv2d(32+64+128, 256, 1)
        self.BN8 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.relu = nn.ReLU()
        self.downsampling1 = nn.Conv2d(32, 32, 32)
        self.downsampling2 = nn.Conv2d(64, 64, 16)
        self.channel_attention = ChannelAttentionModule(32+64+128, channel_reduction_ratio)
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.BN7(self.conv7(out0))
        y = torch.cat([self.relu(y3), self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = self.channel_attention(y) * y
        out2 = self.BN8(self.feature_compress(out2))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# v3192，用于消融实验
class L2FusionNetv3192(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio=8):
        super(L2FusionNetv3192, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 5, padding=2)
        self.BN1 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(48, 48, 5, padding=2)
        self.BN2 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(48, 96, 5, padding=2, stride=2)
        self.BN3 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(96, 192, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(192, 192, 8)
        self.BN7 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.feature_compress = nn.Conv2d(48+96+192, 192, 1)
        self.BN8 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.relu = nn.ReLU()
        self.downsampling1 = nn.Conv2d(48, 48, 32)
        self.downsampling2 = nn.Conv2d(96, 96, 16)
        self.channel_attention = ChannelAttentionModule(48+96+192, channel_reduction_ratio)
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.BN7(self.conv7(out0))
        y = torch.cat([self.relu(y3), self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = (self.channel_attention(y)+1) * y
        out2 = self.BN8(self.feature_compress(out2))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,192)), l2_normalize(out3.view(-1,192))


# feature fusion v4
class L2FusionNetv4256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio=8):
        super(L2FusionNetv4256, self).__init__()
        self.block1_1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, 5, padding=2),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 5, padding=2, stride=2),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.block1_2 = nn.Sequential(nn.Conv2d(1, 32, 9, padding=4),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, 9, padding=4),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 9, padding=4, stride=2),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.block1_3 = nn.Sequential(nn.Conv2d(1, 32, 7, padding=3),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, 7, padding=3),
                                      nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 7, padding=3, stride=2),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.compress1 = nn.Sequential(nn.Conv2d(64*3, 64, 1),
                                       nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track))
        self.BN1 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.block2_1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 3, padding=1, stride=2),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.block2_2 = nn.Sequential(nn.Conv2d(64, 64, 5, padding=2),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 5, padding=2, stride=2),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.block2_3 = nn.Sequential(nn.Conv2d(64, 64, 7, padding=3),
                                      nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 7, padding=3, stride=2),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.compress2 = nn.Sequential(nn.Conv2d(128*3, 128, 1),
                                       nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track))
        self.BN2 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.block3_1 = nn.Sequential(nn.Conv2d(128, 128, 1),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, 8),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.block3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 128, 8),
                                      nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                      nn.ReLU())
        self.compress3 = nn.Sequential(nn.Conv2d(128*2, 128, 1),
                                       nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track))
        self.BN3 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        y1_1 = self.block1_1(x)
        y1_2 = self.block1_2(x)
        y1_3 = self.block1_3(x)
        out1 = self.compress1(torch.cat([y1_1, y1_2, y1_3], dim=1))
        x = self.relu(out1)
        y2_1 = self.block2_1(x)
        y2_2 = self.block2_2(x)
        y2_3 = self.block2_3(x)
        out0 = self.relu(self.compress2(torch.cat([y2_1, y2_2, y2_3], dim=1)))
        y3_1 = self.block3_1(out0)
        y3_2 = self.block3_2(out0)
        out2 = self.compress3(torch.cat([y3_1, y3_2], dim=1))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,16*16)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

# 128维带CBAM
class L2AttentionNetv1128(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio=8):
        super(L2AttentionNetv1128, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(32, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(64, 3, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(128, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(128, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(128, 128, 1), 
                                         nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.BN7 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.relu(self.BN7(self.conv7(out0)))
        out2 = self.transformer((self.channel_attention7(out2) + 1) * out2)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

# 滤波器个数变化256维
class L2AttentionNetv1256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio=8):
        super(L2AttentionNetv1256, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(64, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(128, 3, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(256, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(256, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(256, 256, 1), 
                                         nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(256, 256, 8)
        self.BN7 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.relu(self.BN7(self.conv7(out0)))
        out2 = self.transformer((self.channel_attention7(out2) + 1) * out2)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# 滤波器个数变化192维
class L2AttentionNetv1192(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True, channel_reduction_ratio=8):
        super(L2AttentionNetv1192, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(48, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(96, 3, self.channel_reduction_ratio)
        #self.CBAM3 = CBAM(512, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(192, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(192, 192, 1), 
                                         nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 48, 7, padding=3)
        self.BN1 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(48, 48, 7, padding=3)
        self.BN2 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(48, 96, 7, padding=3, stride=2)
        self.BN3 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(96, 192, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(192, 192, 8)
        self.BN7 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.relu(self.BN7(self.conv7(out0)))
        out2 = self.transformer((self.channel_attention7(out2) + 1) * out2)
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1,192))
        #return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,192)), l2_normalize(out3.view(-1,192))

# 滤波器个数变化320维
class L2AttentionNetv1320(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio=8):
        super(L2AttentionNetv1320, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(80, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(160, 3, self.channel_reduction_ratio)
        #self.CBAM3 = CBAM(512, 1, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(320, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(320, 320, 1), 
                                         nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 80, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(80, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(80, 80, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(80, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(80, 160, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(160, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(160, 160, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(160, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(160, 320, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(320, 320, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(320, 320, 8)
        self.BN7 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.relu(self.BN7(self.conv7(out0)))
        out2 = self.transformer((self.channel_attention7(out2) + 1) * out2)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,320)), l2_normalize(out3.view(-1,320))


# 256维，通过加深实现
class L2Netv1256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False):
        super(L2Netv1256, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN7 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv8 = nn.Conv2d(256, 256, 4)
        self.BN8 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(5)
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        x = self.relu(self.BN7(self.conv7(out0)))
        out2 = self.BN8(self.conv8(x))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# 256维，通过修改滤波器个数实现
class L2Netv2256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False):
        super(L2Netv2256, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(256, 256, 8)
        self.BN7 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.BN7(self.conv7(out0))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# 512维，通过加深实现
class L2Netv1512(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(L2Netv1512, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN7 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.BN8 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.BN9 = nn.BatchNorm2d(512, affine=BN_affine, track_running_stats=BN_track)
        self.conv10 = nn.Conv2d(512, 512, 2)
        self.BN10 = nn.BatchNorm2d(512, affine=BN_affine, track_running_stats=BN_track)
        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(5)
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        x = self.relu(self.BN7(self.conv7(out0)))
        x = self.relu(self.BN8(self.conv8(x)))
        x = self.relu(self.BN9(self.conv9(x)))
        out2 = self.BN10(self.conv10(x))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,512)), l2_normalize(out3.view(-1,512))

# 512维，通过改变滤波器个数实现
class L2Netv2512(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False):
        super(L2Netv2512, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(512, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(512, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(512, 512, 8)
        self.BN7 = nn.BatchNorm2d(512, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.BN7(self.conv7(out0))
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,512)), l2_normalize(out3.view(-1,512))

# 192维，通过改变滤波器个数实现
class L2Net192v3(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False):
        super(L2Net192v3, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, 5, padding=2)
        self.BN1 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(48, 48, 5, padding=2)
        self.BN2 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(48, 96, 5, padding=2, stride=2)
        self.BN3 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(96, 192, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(192, 192, 8)
        self.BN7 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        x = self.relu(self.BN2(self.conv2(x)))
        x = self.relu(self.BN3(self.conv3(x)))
        x = self.relu(self.BN4(self.conv4(x)))
        x = self.relu(self.BN5(self.conv5(x)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        out2 = self.BN7(self.conv7(out0))
        out3 = self.lrn(self.relu(out2))
        return l2_normalize(out3.view(-1,192))
        #return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,192)), l2_normalize(out3.view(-1,192))

class DenseAttentionNetv2256(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio = 8):
        super(DenseAttentionNetv2256, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(32, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(64, 3, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(128, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(128, 128, 1),
                                         nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.BN7 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.downsampling1 = nn.Conv2d(32, 32, 32)
        self.downsampling2 = nn.Conv2d(64, 64, 16)
        self.feature_compress = nn.Sequential(nn.Conv2d(32+64+128, 256, 1),
                                         nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track))
        self.channel_attention = ChannelAttentionModule(32+64+128, channel_reduction_ratio)
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.relu(self.BN7(self.conv7(out0)))
        y3 = self.transformer((self.channel_attention7(y3) + 1) * y3)
        y = torch.cat([y3, self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y) + 1) * y)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

class DenseAttentionNetv2384(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio = 8):
        super(DenseAttentionNetv2384, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(48, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(96, 3, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(192, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(192, 192, 1),
                                         nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 48, 5, padding=2)
        self.BN1 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(48, 48, 5, padding=2)
        self.BN2 = nn.BatchNorm2d(48, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(48, 96, 5, padding=2, stride=2)
        self.BN3 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(96, 192, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(192, 192, 8)
        self.BN7 = nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.downsampling1 = nn.Conv2d(48, 48, 32)
        self.downsampling2 = nn.Conv2d(96, 96, 16)
        self.feature_compress = nn.Sequential(nn.Conv2d(48+96+192, 192, 1),
                                         nn.BatchNorm2d(192, affine=BN_affine, track_running_stats=BN_track))
        self.channel_attention = ChannelAttentionModule(48+96+192, channel_reduction_ratio)
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.relu(self.BN7(self.conv7(out0)))
        y3 = self.transformer((self.channel_attention7(y3) + 1) * y3)
        y = torch.cat([y3, self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y) + 1) * y)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,192)), l2_normalize(out3.view(-1,192))


class DenseAttentionNetv2512(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio = 8):
        super(DenseAttentionNetv2512, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(64, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(128, 3, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(256, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(256, 256, 1),
                                         nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(256, 256, 8)
        self.BN7 = nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.downsampling1 = nn.Conv2d(64, 64, 32)
        self.downsampling2 = nn.Conv2d(128, 128, 16)
        self.feature_compress = nn.Sequential(nn.Conv2d(64+128+256, 256, 1),
                                         nn.BatchNorm2d(256, affine=BN_affine, track_running_stats=BN_track))
        self.channel_attention = ChannelAttentionModule(64+128+256, channel_reduction_ratio)
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.relu(self.BN7(self.conv7(out0)))
        y3 = self.transformer((self.channel_attention7(y3) + 1) * y3)
        y = torch.cat([y3, self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y) + 1) * y)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))


class DenseAttentionNetv2640(nn.Module):
    def __init__(self, BN_affine=False, BN_track=False, channel_reduction_ratio = 8):
        super(DenseAttentionNetv2640, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.CBAM1 = CBAM(80, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(160, 3, self.channel_reduction_ratio)
        self.channel_attention7 = ChannelAttentionModule(320, self.channel_reduction_ratio)
        self.transformer = nn.Sequential(nn.Conv2d(320, 320, 1), 
                                         nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track))
        self.conv1 = nn.Conv2d(1, 80, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(80, affine=BN_affine, track_running_stats=BN_track)
        self.conv2 = nn.Conv2d(80, 80, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(80, affine=BN_affine, track_running_stats=BN_track)
        self.conv3 = nn.Conv2d(80, 160, 3, padding=1, stride=2)
        self.BN3 = nn.BatchNorm2d(160, affine=BN_affine, track_running_stats=BN_track)
        self.conv4 = nn.Conv2d(160, 160, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(160, affine=BN_affine, track_running_stats=BN_track)
        self.conv5 = nn.Conv2d(160, 320, 3, padding=1, stride=2)
        self.BN5 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.conv6 = nn.Conv2d(320, 320, 3, padding=1)
        self.BN6 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.conv7 = nn.Conv2d(320, 320, 8)
        self.BN7 = nn.BatchNorm2d(320, affine=BN_affine, track_running_stats=BN_track)
        self.lrn = nn.LocalResponseNorm(5)
        self.relu = nn.ReLU()
        self.input_size = 32
        self.downsampling1 = nn.Conv2d(80, 80, 32)
        self.downsampling2 = nn.Conv2d(160, 160, 16)
        self.feature_compress = nn.Sequential(nn.Conv2d(80+160+320, 640, 1),
                                         nn.BatchNorm2d(640, affine=BN_affine, track_running_stats=BN_track))
        self.channel_attention = ChannelAttentionModule(80+160+320, channel_reduction_ratio)
    
    def forward(self, x):
        out1 = self.BN1(self.conv1(x))
        x = self.relu(out1)
        y1 = self.CBAM1(self.relu(self.BN2(self.conv2(x))))
        x = self.relu(self.BN3(self.conv3(y1)))
        y2 = self.CBAM2(self.relu(self.BN4(self.conv4(x))))
        x = self.relu(self.BN5(self.conv5(y2)))
        out0 = self.relu(self.BN6(self.conv6(x)))
        y3 = self.relu(self.BN7(self.conv7(out0)))
        y3 = self.transformer((self.channel_attention7(y3) + 1) * y3)
        y = torch.cat([y3, self.downsampling2(y2), self.downsampling1(y1)], dim=1)
        out2 = self.feature_compress((self.channel_attention(y) + 1) * y)
        out3 = self.lrn(self.relu(out2))
        return out0, l2_normalize(out1.sum(dim=1).view(-1,32*32)), l2_normalize(out2.view(-1,640)), l2_normalize(out3.view(-1,640))
        
# v1是结合了fusion的v1和CBAM
class DenseAttentionNetv1128(nn.Module):
    def __init__(self, channel_reduction_ratio = 8):
        super(DenseAttentionNetv1128, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.input_size = 64
        self.relu = nn.ReLU()
        self.CBAM1 = CBAM(32, 7, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(64, 3, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(128, 1, self.channel_reduction_ratio)
        self.block1_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True))
        self.block1_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 8),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True))
        self.lrn = nn.LocalResponseNorm(5)
    
    # CBAM和downsampling哪个在前哪个在后仍需商榷
    def forward(self, x):
        out1 = self.block1_1(x)
        x1_1 = self.relu(out1)
        x1_2 = self.block1_2(x1_1)
        y1 = torch.cat([x1_2, x1_1], dim=1)
        x2_1 = self.CBAM1(self.downsampling1(y1))
        x2_2 = self.block2(x2_1)
        y2 = torch.cat([x2_2, x2_1], dim=1)
        x3_1 = self.CBAM2(self.downsampling2(y2))
        x3_2 = self.block3(x3_1)
        y3 = torch.cat([x3_2, x3_1], dim=1)
        x4_1 = self.CBAM3(self.downsampling3(y3)) #需要输出给stage2作为feature map
        out2 = self.block4(x4_1)
        out3 = self.lrn(self.relu(out2))
        return x4_1, l2_normalize(out1.sum(dim=1).view(-1,64*64)), l2_normalize(out2.view(-1,128)), l2_normalize(out3.view(-1,128))

# 256维
class DenseAttentionNetv1256(nn.Module):
    def __init__(self, channel_reduction_ratio = 8):
        super(DenseAttentionNetv1256, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.input_size = 64
        self.relu = nn.ReLU()
        self.CBAM1 = CBAM(32, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(64, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(128, self.channel_reduction_ratio)
        self.CBAM4 = CBAM(256, self.channel_reduction_ratio)
        self.block1_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True))
        self.block1_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True))
        self.downsampling4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, 4),
            nn.BatchNorm2d(256, affine=True, track_running_stats=True))
        self.lrn = nn.LocalResponseNorm(5)
    
    # CBAM和downsampling哪个在前哪个在后仍需商榷
    def forward(self, x):
        out1 = self.block1_1(x)
        x1_1 = self.relu(out1)
        x1_2 = self.block1_2(x1_1)
        y1 = torch.cat([x1_2, x1_1], dim=1)
        x2_1 = self.CBAM1(self.downsampling1(y1))
        x2_2 = self.block2(x2_1)
        y2 = torch.cat([x2_2, x2_1], dim=1)
        x3_1 = self.CBAM2(self.downsampling2(y2))
        x3_2 = self.block3(x3_1)
        y3 = torch.cat([x3_2, x3_1], dim=1)
        x4_1 = self.CBAM3(self.downsampling3(y3)) #需要输出给stage2作为feature map
        x4_2 = self.block4(x4_1)
        y4 = torch.cat([x4_2, x4_1], dim=1)
        x5_1 = self.CBAM4(self.downsampling4(y4))
        out2 = self.block5(x5_1)
        out3 = self.lrn(self.relu(out2))
        return x4_1, l2_normalize(out1.sum(dim=1).view(-1,64*64)), l2_normalize(out2.view(-1,256)), l2_normalize(out3.view(-1,256))

# 512维
class DenseAttentionNetv1512(nn.Module):
    def __init__(self, channel_reduction_ratio = 8):
        super(DenseAttentionNetv1512, self).__init__()
        self.channel_reduction_ratio = channel_reduction_ratio
        self.input_size = 64
        self.relu = nn.ReLU()
        self.CBAM1 = CBAM(32, self.channel_reduction_ratio)
        self.CBAM2 = CBAM(64, self.channel_reduction_ratio)
        self.CBAM3 = CBAM(128, self.channel_reduction_ratio)
        self.CBAM4 = CBAM(256, self.channel_reduction_ratio)
        self.CBAM5 = CBAM(512, self.channel_reduction_ratio)
        self.block1_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True))
        self.block1_2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU())
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True))
        self.downsampling4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, affine=True, track_running_stats=True))
        self.downsampling5 = nn.Sequential(
            nn.Conv2d(512, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.ReLU())
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512, affine=True, track_running_stats=True))
        self.lrn = nn.LocalResponseNorm(5)
    
    # CBAM和downsampling哪个在前哪个在后仍需商榷
    def forward(self, x):
        out1 = self.block1_1(x)
        x1_1 = self.relu(out1)
        x1_2 = self.block1_2(x1_1)
        y1 = torch.cat([x1_2, x1_1], dim=1)
        x2_1 = self.CBAM1(self.downsampling1(y1))
        x2_2 = self.block2(x2_1)
        y2 = torch.cat([x2_2, x2_1], dim=1)
        x3_1 = self.CBAM2(self.downsampling2(y2))
        x3_2 = self.block3(x3_1)
        y3 = torch.cat([x3_2, x3_1], dim=1)
        x4_1 = self.CBAM3(self.downsampling3(y3)) #需要输出给stage2作为feature map
        x4_2 = self.block4(x4_1)
        y4 = torch.cat([x4_2, x4_1], dim=1)
        x5_1 = self.CBAM4(self.downsampling4(y4))
        x5_2 = self.block5(x5_1)
        y5 = torch.cat([x5_2, x5_1], dim=1)
        x6_1 = self.CBAM5(self.downsampling5(y5))
        out2 = self.block6(x6_1)
        out3 = self.lrn(self.relu(out2))
        return x4_1, l2_normalize(out1.sum(dim=1).view(-1,64*64)), l2_normalize(out2.view(-1,512)), l2_normalize(out3.view(-1,512))


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel, kernel_size=7, channel_reduction_ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel, channel_reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        # 这是原版
        out = (self.channel_attention(x)) * x
        # x = out
        out = (self.spatial_attention(out)) * out
        
        #out = (self.spatial_attention(x)) * x
        #out = (self.channel_attention(out)) * out
        # 这是并联
        #out = (self.channel_attention(x)) * x
        #out = (self.spatial_attention(x)) * out
        return out

# 专门用于展开后的向量的注意力模块，和CBAM一样直接输出处理后而不是掩膜
class FCAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=32):
        super(FCAttentionModule, self).__init__()
        self.ratio = ratio
        self.FC1 = nn.Linear(in_channels, in_channels // ratio)
        self.FC2 = nn.Linear(in_channels // ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.FC2(self.relu(self.FC1(x)))) * x

#v1回归的是H矩阵，不太容易训练，不太好用
class HNetv1(nn.Module):
    def __init__(self, input_size, in_channels, stride=1, BN_track=True, BN_affine=False):
        super(HNetv1, self).__init__()
        self.input_size = input_size
        self.compact_in = int(math.pow(input_size, 2)) #做相关且要conv时
        self.compact_in = in_channels #不做相关时
        self.regressor_in = input_size//stride//stride*input_size//stride//stride*self.compact_in//4
        self.regressor_in = int(math.pow(input_size, 4)) #做相关不要conv时
        
        self.compact_feature = nn.Sequential(
            nn.Conv2d(self.compact_in, self.compact_in//2, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in//2, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU(),
            nn.Conv2d(self.compact_in//2, self.compact_in//4, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in//4, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.regressor = nn.Sequential(nn.Linear(self.regressor_in, self.regressor_in // 4),
                                       nn.ReLU(),
                                       nn.Linear(self.regressor_in // 4, self.regressor_in // 16),
                                       nn.ReLU(),
                                       nn.Linear(self.regressor_in // 16, self.regressor_in // 64),
                                       nn.Tanh(),
                                       nn.Linear(self.regressor_in // 64, 8))
    
    def forward(self, x1, x2):
        x = self.correlation_process(x1, x2)
        x = x.view(-1, int(math.pow(self.input_size, 4)))
        H = self.regressor(x)
        #return (torch.transpose(torch.transpose(H, 0, 1) / H[:,8], 0, 1)).view(-1, 3, 3)
        return torch.cat([H, torch.ones(len(H), 1).to("cuda")], dim=1).view(-1, 3, 3)
    
    # 要保证x1和x2的特征向量是归一化过的，需要在输出feature map的时候进行处理
    def correlation_process(self, x1, x2):
        N, C, H, W = x1.shape
        out = torch.Tensor(N, H*W, H, W)
        if self.cuda:
            out = out.to("cuda")
        x1 = l2_normalize(x1)
        x2 = l2_normalize(x2)
        for i in range(H):
            for j in range(W):
                out[:,:,i,j] = torch.matmul(torch.squeeze(x1[:,:,i,j]).reshape((N,1,C)),
                                            x2.reshape((N,C,H*W))).transpose(2,1).reshape((N,H*W))
        return out        

#v2回归的是patch中心点的位置偏差比率
#这里采用的是两个feature map特征拼接的形式
class HNetv2(nn.Module):
    def __init__(self, input_size, in_channels, stride=1, BN_track=True, BN_affine=False):
        super(HNetv2, self).__init__()
        self.input_size = input_size
        #self.compact_in = int(math.pow(input_size, 2)) #做相关且要conv时
        #self.regressor_in = int(math.pow(input_size, 4)) #做相关不要conv时
        self.compact_in1 = in_channels #不做相关时
        #self.compact_in2 = in_channels // 2
        self.regressor_in1 = input_size//stride//stride*input_size//stride//stride*self.compact_in1//16
        #self.regressor_in2 = input_size//stride//stride*input_size//stride//stride*self.compact_in2//16
        self.regressor_in = self.regressor_in1
        
        self.attention1_1 = SpatialAttentionModule(5)
        self.attention1_2 = SpatialAttentionModule(3)
        #self.attention2_1 = SpatialAttentionModule(5)
        #self.attention2_2 = SpatialAttentionModule(3)
        #self.attention3_1 = SpatialAttentionModule(5)
        #self.attention3_2 = SpatialAttentionModule(3)
        self.attention1 = FCAttentionModule(self.regressor_in1, 32)
        #self.attention2 = FCAttentionModule(self.regressor_in2, 32)
        #self.attention3 = FCAttentionModule(self.regressor_in3, 32)
        
        self.compact_feature1_1 = nn.Sequential(
            nn.Conv2d(self.compact_in1, self.compact_in1//4, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in1//4, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.compact_feature1_2 = nn.Sequential(
            nn.Conv2d(self.compact_in1//4, self.compact_in1//16, input_size//stride//4 - 1, padding=(input_size//stride//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in1//16, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        """self.compact_feature2_1 = nn.Sequential(
            nn.Conv2d(self.compact_in2, self.compact_in2//4, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in2//4, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.compact_feature2_2 = nn.Sequential(
            nn.Conv2d(self.compact_in2//4, self.compact_in2//16, input_size//stride//4 - 1, padding=(input_size//stride//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in2//16, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())"""
        """self.compact_feature3_1 = nn.Sequential(
            nn.Conv2d(self.compact_in3, self.compact_in3//4, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in3//4, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())
        self.compact_feature3_2 = nn.Sequential(
            nn.Conv2d(self.compact_in3//4, self.compact_in3//16, input_size//stride//4 - 1, padding=(input_size//stride//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in3//16, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())"""
        
        
        """self.compact_feature = nn.Sequential(
            nn.Conv2d(self.compact_in1, self.compact_in1//4, input_size//4 - 1, padding=(input_size//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in1//4, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU(),
            nn.Conv2d(self.compact_in1//4, self.compact_in1//16, input_size//stride//4 - 1, padding=(input_size//stride//4 - 1 -1)//2, stride=stride),
            nn.BatchNorm2d(self.compact_in1//16, affine=BN_affine, track_running_stats=BN_track),
            nn.ReLU())"""
        # 目前的这个是修改后的回归器
        self.regressor = nn.Sequential(nn.Linear(self.regressor_in, self.regressor_in // 4),
                                       nn.ReLU(),
                                       nn.Linear(self.regressor_in // 4, self.regressor_in // 16),
                                       nn.ReLU(),
                                       nn.Linear(self.regressor_in // 16, self.regressor_in // 64),
                                       nn.ReLU(),
                                       nn.Linear(self.regressor_in // 64, 2),
                                       nn.Tanh())
        
    def forward(self, x1, x2):
        y1 = torch.cat([x1, x2], dim=1)
        #y2 = x1 - x2
        #y3 = x1 * x2
        #x = self.correlation_process(x1, x2)
        y1 = self.compact_feature1_1(y1)
        y1 = self.attention1_1(y1) * y1
        y1 = self.compact_feature1_2(y1)
        y1 = self.attention1_2(y1) * y1
        
        """y2 = self.compact_feature2_1(y2)
        y2 = self.attention2_1(y2) * y2
        y2 = self.compact_feature2_2(y2)
        y2 = self.attention2_2(y2) * y2"""
        
        y = self.regressor(self.attention1(y1.view(-1, self.regressor_in1)))
        return y
    
    def correlation_process(self, x1, x2):
        N, C, H, W = x1.shape
        out = torch.Tensor(N, H*W, H, W)
        if self.cuda:
            out = out.to("cuda")
        x1 = l2_normalize(x1)
        x2 = l2_normalize(x2)
        for i in range(H):
            for j in range(W):
                out[:,:,i,j] = torch.matmul(torch.squeeze(x1[:,:,i,j]).reshape((N,1,C)),
                                            x2.reshape((N,C,H*W))).transpose(2,1).reshape((N,H*W))
        return out

# v1是使用自己的DenseAttentionNet 
class stage1_modelv1(nn.Module):
    def __init__(self, model_path, output_type=True):
        super(stage1_modelv1, self).__init__()
        self.basic_model = DenseAttentionNetv2256()
        self.basic_model.load_state_dict(torch.load(model_path))
        self.output_type = output_type #True代表输入压缩特征图
        self.relu = nn.ReLU()
        self.input_size = self.basic_model.input_size
        
    def forward(self, x):
        y1 = self.relu(self.basic_model.BN2(self.basic_model.conv2(self.relu(self.basic_model.BN1(self.basic_model.conv1(x))))))
        y2 = self.relu(self.basic_model.BN4(self.basic_model.conv4(self.relu(self.basic_model.BN3(self.basic_model.conv3(y1))))))
        y3 = self.relu(self.basic_model.BN6(self.basic_model.conv6(self.relu(self.basic_model.BN5(self.basic_model.conv5(y2))))))
        if self.output_type:
            return y3
        return torch.cat([y1, nn.functional.interpolate(y2, scale_factor=2, mode='nearest'), nn.functional.interpolate(y3, scale_factor=4, mode='nearest')], dim=1)

#这个是用resnet18作为提取特征的模型
class stage1_modelv2(nn.Module):
    def __init__(self, model_path, output_type=True):
        super(stage1_modelv2, self).__init__()
        self.basic_model = torchvision.models.resnet18(pretrained=False)
        self.basic_model.load_state_dict(torch.load(model_path))
        self.output_type = output_type
        self.input_size = 64
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y0 = self.relu(self.basic_model.bn1(self.basic_model.conv1(x))) #64维
        y1 = self.basic_model.layer1(y0) #64维
        y2 = self.basic_model.layer2(y1) #128维
        y3 = self.basic_model.layer3(y2) #256维
        #y4 = self.basic_model.layer4(y3) #512维
        if self.output_type:
            return y3
        return torch.cat([y1, nn.functional.interpolate(y2, scale_factor=2, mode='nearest'), nn.functional.interpolate(y3, scale_factor=4, mode='nearest')], dim=1)
        

if __name__ == "__main__":
    print("This is a model util!")
    
    
    

