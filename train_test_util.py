# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:38:46 2022

@author: knight
"""

# 专门存放训练和测试过程中关于loss和acc等相关函数

import os
import torch
from torch import nn
import torchvision
import cv2
import numpy as np

# 比较相对距离的损失函数
def get_loss_relative(img_map, img_lcam, net, feature_dims):
    _, BN1_map, BN7_map, des_map = net(img_map)
    _, BN1_lcam, BN7_lcam, des_lcam = net(img_lcam)
    D = torch.sqrt(2 * (1 - torch.matmul(des_map, torch.transpose(des_lcam, 0, 1))))
    D_exp = torch.exp(2 - D)
    Sc = D_exp / D_exp.sum(dim=0)
    Sr = torch.transpose(torch.transpose(D_exp,0,1) / D_exp.sum(dim=1),0,1)
    E1 = -0.5 * (torch.trace(torch.log(Sc)) + torch.trace(torch.log(Sr)))
    
    G_BN1 = torch.matmul(BN1_map, torch.transpose(BN1_lcam, 0, 1))
    G_BN7 = torch.matmul(BN7_map, torch.transpose(BN7_lcam, 0, 1))
    G_BN1_exp = torch.exp(G_BN1)
    G_BN7_exp = torch.exp(G_BN7)
    Vc_BN1 = G_BN1_exp / G_BN1_exp.sum(dim=0)
    Vr_BN1 = torch.transpose(torch.transpose(G_BN1_exp,0,1) / G_BN1_exp.sum(dim=1),0,1)
    Vc_BN7 = G_BN7_exp / G_BN7_exp.sum(dim=0)
    Vr_BN7 = torch.transpose(torch.transpose(G_BN7_exp,0,1) / G_BN7_exp.sum(dim=1),0,1)
    E3 = -0.5 * (torch.trace(torch.log(Vc_BN1)) + torch.trace(torch.log(Vr_BN1)) + 
                 torch.trace(torch.log(Vc_BN7)) + torch.trace(torch.log(Vr_BN7)))
    
    R_map = torch.matmul(torch.transpose(BN7_map, 0, 1), BN7_map) / feature_dims
    R_lcam = torch.matmul(torch.transpose(BN7_lcam, 0, 1), BN7_lcam) / feature_dims
    E2 = 0.5 * (torch.pow(R_map, 2).sum() - torch.trace(torch.pow(R_map, 2)) +
                torch.pow(R_lcam, 2).sum() - torch.trace(torch.pow(R_lcam, 2)))
    
    return D, E1, E2, E3

# 用于L2Net_2ch_2stream比较相对距离且不考虑中间层的监督项的损失函数
def get_loss_relative_2ch2stream(img_map, img_lcam, net, feature_dims):
    des_map = net(img_map)
    des_lcam = net(img_lcam)
    D = torch.sqrt(2 * (1 - torch.matmul(des_map, torch.transpose(des_lcam, 0, 1))))
    D_exp = torch.exp(2 - D)
    Sc = D_exp / D_exp.sum(dim=0)
    Sr = torch.transpose(torch.transpose(D_exp,0,1) / D_exp.sum(dim=1),0,1)
    E1 = -0.5 * (torch.trace(torch.log(Sc)) + torch.trace(torch.log(Sr)))
    
    R_map = torch.matmul(torch.transpose(des_map, 0, 1), des_map) / feature_dims
    R_lcam = torch.matmul(torch.transpose(des_lcam, 0, 1), des_lcam) / feature_dims
    E2 = 0.5 * (torch.pow(R_map, 2).sum() - torch.trace(torch.pow(R_map, 2)) +
                torch.pow(R_lcam, 2).sum() - torch.trace(torch.pow(R_lcam, 2)))
    
    return D, E1, E2

# 结合了hard mining的思想，自己改进的、摈除了其他层监督项的损失函数
def get_loss_relative_triplet(des_map, des_lcam, feature_dims, batch_size):
    D = 2 * (1 - torch.matmul(des_map, torch.transpose(des_lcam, 0, 1)))
    Sc = D / D.sum(dim=0)
    for i in range(len(Sc)):      
        if i != 0:
            tmpval1, tmpindex1 = torch.min(Sc[0:i, i], dim=0)
            if i+1 < len(D):
                tmpval2, tmpindex2 = torch.min(Sc[i+1:, i], dim=0)
                if tmpval1 < tmpval2:
                    Sc[i, i] = Sc[i, i] - tmpval1
                else:
                    Sc[i, i] = Sc[i, i] - tmpval2
            else:
                Sc[i, i] = Sc[i, i] - tmpval1
        else:
            tmpval2, tmpindex2 = torch.min(Sc[i+1:, i], dim=0)
            Sc[i, i] = Sc[i, i] - tmpval2
    
    E1 = torch.trace(2 + Sc) / batch_size
    
    R_map = torch.matmul(torch.transpose(des_map, 0, 1), des_map) / feature_dims
    R_lcam = torch.matmul(torch.transpose(des_lcam, 0, 1), des_lcam) / feature_dims
    E2 = 0.5 * (torch.pow(R_map, 2).sum() - torch.trace(torch.pow(R_map, 2)) +
                torch.pow(R_lcam, 2).sum() - torch.trace(torch.pow(R_lcam, 2)))
    
    return D, E1, E2

# 比较双塔特征向量的绝对距离，这个函数返回了若干个相关的可能用到的损失函数
def get_loss_siam(des_map, des_lcam, labels, margin, loss_type):
    D = 2 * (1 - torch.matmul(des_map, torch.transpose(des_lcam, 0, 1)))
    d = torch.diagonal(D)
    if loss_type == 1:
        loss_fn = nn.HingeEmbeddingLoss(margin)
        loss = loss_fn(d, labels)
    elif loss_type == 2:
        loss_fn = nn.SoftMarginLoss()
        loss = loss_fn(2 - d, labels)
    elif loss_type == 3:
        loss_fn = nn.CosineEmbeddingLoss(margin)
        loss = loss_fn(des_map, des_lcam, labels)
    else:
        raise Exception("请指定损失函数类型！")
    return D, loss

# 比较triplet的hinge loss
def get_loss_triplet(des_map, des_lcam_pos, des_lcam_neg, margin=1):
    loss_fn = nn.TripletMarginLoss(margin)
    return loss_fn(des_map, des_lcam_pos, des_lcam_neg)

# 该函数可用于L2net的精度获取和所有特征向量距离比较测试集的精度获取
def get_acc(D, cuda=True):
    batch_size = len(D)
    _, min_c = torch.min(D, axis=0)
    #_, min_r = torch.min(D, axis=1)
    mask = torch.linspace(0, batch_size-1, batch_size)
    if cuda:
        mask = mask.to("cuda")
    acc_c = torch.eq(min_c, mask).type(torch.float32)
    #acc_r = torch.eq(min_r, mask).type(torch.float32)
    #acc = torch.eq(acc_c, acc_r).type(torch.float32) * acc_c
    return acc_c.mean(), acc_c

# 该函数用于双塔模型训练时的精度获取
def get_acc_siam(D, labels, acc_margin=1, cuda=True):
    d = torch.diagonal(D)
    acc = torch.Tensor([0])
    if cuda:
        acc = acc.to("cuda")
    for i in range(len(labels)):
        if((labels[i] == 1 and d[i] < acc_margin) or (labels[i] == -1 and d[i] >= acc_margin)):
            acc += 1
    acc = torch.true_divide(acc, len(labels))
    return acc

# 该函数用于stage2中获取了H矩阵后loss和精度的同时获取
# 这个函数的矩阵写法可能有一定的问题，后续可以观察        
def get_loss_and_acc_stage2v1(H, H_gt, img_size, margin=0.1, cuda=True):
    tmp = torch.linspace(0, img_size-1, img_size)
    
    if H.shape != H_gt.shape:
        H_gt = H_gt.expand(H.shape[0],3,3)
    
    H = H.unsqueeze(dim=1)
    H_gt = H_gt.unsqueeze(dim=1)
    img = torch.cat(
        [tmp.reshape((1,img_size)).expand(img_size, img_size).reshape((img_size, img_size, 1)),
         tmp.reshape((img_size,1)).expand(img_size, img_size).reshape((img_size, img_size, 1)),
         torch.ones(img_size, img_size, 1) * img_size / 128], dim=2).reshape(img_size * img_size, 3, 1).expand(len(H), img_size * img_size, 3, 1)
    if cuda:
        img = img.to("cuda")
        
    H_img = torch.squeeze(torch.matmul(H, img), dim=3)
    H_img = (img_size / 128 * H_img / H_img[:,:,2].expand(3, len(H), img_size * img_size).transpose(0,1).transpose(1,2))[:,:,:2]
    H_gt_img = torch.squeeze(torch.matmul(H_gt, img), dim=3)
    H_gt_img = (img_size / 128 * H_gt_img / H_gt_img[:,:,2].expand(3, len(H_gt), img_size * img_size).transpose(0,1).transpose(1,2))[:,:,:2]
    mask = torch.abs(H_img - H_gt_img).sum(dim=2) < (margin * img_size)
    #loss_fn = nn.MSELoss(reduction="none")
    #loss = loss_fn(H_img, H_gt_img)
    #loss[mask] = 0 #这种写法是正确的！
    loss = ((H - H_gt).abs() + 1).log()
    return loss.mean(), mask.type(torch.float32).mean()

def get_loss_and_acc_stage2v2(pred, gt, margin=0.02):
    mask = (pred - gt).abs().sum(dim=1) < margin
    loss = (pred - gt).abs()
    #loss_fn = nn.MSELoss()
    #loss = loss_fn(pred, gt)
    #loss[mask] = 0
    return loss.mean() * 100, mask.type(torch.float32).mean()

# 该函数用于基于度量模型的loss获取
def get_loss_matchnet(metric, labels):
    batch_size = len(metric)
    label1 = torch.zeros(batch_size, dtype=torch.int64).to("cuda")
    label2 = torch.ones(batch_size, dtype=torch.int64).to("cuda")
    label = torch.where(labels>0, label1, label2)
    loss_fn = nn.NLLLoss()
    return loss_fn(metric, label)

# 该函数用于基于度量双塔模型的精度获取
def get_acc_matchnet(img_map, img_lcam, net):
    des_map = net.get_feature(img_map)
    des_lcam = net.get_feature(img_lcam)
    batch_size, feature_dim = des_map.size()
    des_map = des_map.repeat(1, batch_size).view(-1, feature_dim).to("cuda")
    des_lcam = des_lcam.repeat(batch_size, 1).to("cuda")
    metric = net.get_metric(des_map, des_lcam)[:, 0].view(-1, batch_size)
    mask = torch.linspace(0, batch_size-1, batch_size).to("cuda")
    _, max_r = torch.max(metric, axis=1)
    acc_r = torch.eq(max_r, mask).type(torch.float32)
    return acc_r.mean()
  
# 该函数用于基于度量的2ch模型的精度获取
def get_acc_matchnet_2ch(img_map, img_lcam, net):
    batch_size, _, h ,w = img_map.size()
    img_map = img_map.repeat(1, 1, batch_size, 1).view(batch_size**2, 1, h, w).to("cuda")
    img_lcam = img_lcam.repeat(batch_size, 1, 1, 1).to("cuda")
    metric = net(img_map, img_lcam)[:, 0].view(-1, batch_size)
    mask = torch.linspace(0, batch_size-1, batch_size).to("cuda")
    _, max_r = torch.max(metric, axis=1)
    acc_r = torch.eq(max_r, mask).type(torch.float32)
    return acc_r.mean()
    
# 该函数用于基于度量的特征学习模型的精度获取——与get_acc_matchnet基本类似
def get_acc_feature_metric(des_map, des_lcam, net):
    batch_size, feature_dim = des_map.size()
    des_map = des_map.repeat(1, batch_size).view(-1, feature_dim).to("cuda")
    des_lcam = des_lcam.repeat(batch_size, 1).to("cuda")
    metric = net(des_map, des_lcam)[:, 0].view(-1, batch_size)
    mask = torch.linspace(0, batch_size-1, batch_size).to("cuda")
    _, max_r = torch.max(metric, axis=1)
    acc_r = torch.eq(max_r, mask).type(torch.float32)
    return acc_r.mean()


if __name__ == '__main__':
    print("This is a util when training and testing")
