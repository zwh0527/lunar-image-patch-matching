# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:44:52 2022

@author: knight
"""

# 专门用于存放需要另外编写的other_model的网络结构与相关模块的类

import os
import torch
from torch import nn
import torchvision
import cv2
import numpy as np
import math
from model_util import l2_normalize

class MatchNet(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(MatchNet, self).__init__()
        self.input_size = 64
        self.feature_network = nn.Sequential(nn.Conv2d(1, 24, 7, padding=3),
                                             nn.BatchNorm2d(24, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(24, 64, 5, padding=2),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(64, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 64, 3, padding=1),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1))
        self.metric_network = nn.Sequential(nn.Linear(4096*2, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 2),
                                            nn.LogSoftmax(dim=1))
        
    def forward(self, x1, x2):
        y1 = self.feature_network(x1).view(-1, 4096)
        y2 = self.feature_network(x2).view(-1, 4096)
        y = torch.cat([y1, y2], dim=1)
        return self.metric_network(y)
    
    def get_feature(self, x):
        return self.feature_network(x).view(-1, 4096)
    
    def get_metric(self, x1, x2):
        return self.metric_network(torch.cat([x1, x2], dim=1))
        
class MatchNet_2ch(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(MatchNet_2ch, self).__init__()
        self.input_size = 64
        self.feature_network = nn.Sequential(nn.Conv2d(2, 24, 7, padding=3),
                                             nn.BatchNorm2d(24, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(24, 64, 5, padding=2),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(64, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 64, 3, padding=1),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1))
        self.metric_network = nn.Sequential(nn.Linear(4096, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 2),
                                            nn.LogSoftmax(dim=1))
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        y = self.feature_network(x).view(-1, 4096)
        return self.metric_network(y)
    
class MatchNet_2ch2stream(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(MatchNet_2ch2stream, self).__init__()
        self.input_size = 64
        self.feature_network1 = nn.Sequential(nn.Conv2d(1, 24, 7, padding=3),
                                             nn.BatchNorm2d(24, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(24, 64, 5, padding=2),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(64, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 64, 3, padding=1),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1))
        self.feature_network2 = nn.Sequential(nn.Conv2d(1, 24, 7, padding=3),
                                             nn.BatchNorm2d(24, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(24, 64, 5, padding=2),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1),
                                             nn.Conv2d(64, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 96, 3, padding=1),
                                             nn.BatchNorm2d(96, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.Conv2d(96, 64, 3, padding=1),
                                             nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                             nn.ReLU(),
                                             nn.MaxPool2d(3, stride=2, padding=1))
        self.metric_network = nn.Sequential(nn.Linear(4096*4, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 2),
                                            nn.LogSoftmax(dim=1))
        
    def forward(self, x1, x2):
        y1_1 = self.feature_network1(x1).view(-1, 4096)
        y1_2 = self.feature_network1(x2).view(-1, 4096)
        y2_1 = self.feature_network2(nn.functional.interpolate(x1[:,:,self.input_size//4:self.input_size//4*3,self.input_size//4:self.input_size//4*3], scale_factor=2, mode='nearest')).view(-1, 4096)
        y2_2 = self.feature_network2(nn.functional.interpolate(x2[:,:,self.input_size//4:self.input_size//4*3,self.input_size//4:self.input_size//4*3], scale_factor=2, mode='nearest')).view(-1, 4096)
        y = torch.cat([y1_1, y2_1, y1_2, y2_2], dim=1)
        return self.metric_network(y)
    
    def get_feature(self, x):
        y1 = self.feature_network1(x).view(-1, 4096)
        y2 = self.feature_network2(nn.functional.interpolate(x[:,:,self.input_size//4:self.input_size//4*3,self.input_size//4:self.input_size//4*3], scale_factor=2, mode='nearest')).view(-1, 4096)
        return torch.cat([y1, y2], dim=1)
    
    def get_metric(self, x1, x2):
        return self.metric_network(torch.cat([x1, x2], dim=1))

class L2Net_2ch2stream(nn.Module):
    def __init__(self, BN_affine=False, BN_track=True):
        super(L2Net_2ch2stream, self).__init__()
        self.input_size = 32
        self.feature_network1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                              nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(32, 32, 3, padding=1),
                                              nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(32, 64, 3, padding=1, stride=2),
                                              nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(64, 64, 3, padding=1),
                                              nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(64, 128, 3, padding=1, stride=2),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(128, 128, 3, padding=1, stride=1),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(128, 128, 8),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.LocalResponseNorm(5))
        self.feature_network2 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                                              nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(32, 32, 3, padding=1),
                                              nn.BatchNorm2d(32, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(32, 64, 3, padding=1, stride=2),
                                              nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(64, 64, 3, padding=1),
                                              nn.BatchNorm2d(64, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(64, 128, 3, padding=1, stride=2),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(128, 128, 3, padding=1, stride=1),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.Conv2d(128, 128, 8),
                                              nn.BatchNorm2d(128, affine=BN_affine, track_running_stats=BN_track),
                                              nn.ReLU(),
                                              nn.LocalResponseNorm(5))
        
    def forward(self, x):
        out1 = self.feature_network1(x).view(-1, 128)
        out2 = self.feature_network2(nn.functional.interpolate(x[:,:,self.input_size//4:self.input_size//4*3,self.input_size//4:self.input_size//4*3], scale_factor=2, mode='nearest')).view(-1, 128)
        return l2_normalize(torch.cat([out1, out2], dim=1))
        
# 这个是基于度量的描述子学习
class Feature_Metric(nn.Module):
    def __init__(self, feature_dims, BN_affine=False, BN_track=True):
        super(Feature_Metric, self).__init__()
        self.feature_dims = feature_dims
        self.metric_network = nn.Sequential(#nn.BatchNorm1d(self.feature_dims*2, affine=BN_affine, track_running_stats=BN_track),
                                    nn.Linear(self.feature_dims*2, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2),
                                    nn.LogSoftmax(dim=1))
        
    def forward(self, x1, x2):
        # 此处x1与x2应该是descriptor组成的batch_size
        x = torch.cat([x1, x2], dim=1)
        return self.metric_network(x)
    
# 这个是基于描述符的描述子学习
class Feature_Descriptor(nn.Module):
    def __init__(self, feature_dims, BN_affine=False, BN_track=True):
        super(Feature_Descriptor, self).__init__()
        self.feature_dims = feature_dims
        self.descriptor_network = nn.Sequential(#nn.BatchNorm1d(self.feature_dims, affine=BN_affine, track_running_stats=BN_track),
                                                nn.Linear(self.feature_dims, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 128),
                                                nn.ReLU())
        
    def forward(self, x):
        # 此处的x应该是descriptor
        return l2_normalize(self.descriptor_network(x))
        
# 这个是基于度量的描述子学习，和Feature_Metric相同只不过参数不同因为只对块取一个描述符
class Feature_SVM(nn.Module):
    def __init__(self, feature_dims, BN_affine=False, BN_track=True):
        super(Feature_SVM, self).__init__()
        self.feature_dims = feature_dims
        self.metric_network = nn.Sequential(nn.BatchNorm1d(self.feature_dims*2, affine=BN_affine, track_running_stats=BN_track),
                                    nn.Linear(self.feature_dims*2, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    nn.LogSoftmax(dim=1))
        
    def forward(self, x1, x2):
        # 此处x1与x2应该是descriptor组成的batch_size
        x = torch.cat([x1, x2], dim=1)
        return self.metric_network(x)







