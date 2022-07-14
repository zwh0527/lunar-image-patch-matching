# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:38:26 2022

@author: knight
"""

# 专门存放数据集和数据集处理相关的函数和类

import os
import torch
from torch import nn
import torchvision
import cv2
import numpy as np

# stage1用到的数据集
class ImagePatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, num_of_each_map=1, input_size=128):
        self.root = root
        self.transforms = transforms
        self.num_of_each_map = num_of_each_map
        self.maps = list(sorted(os.listdir(os.path.join(root, "map"))))
        self.lcams = list(sorted(os.listdir(os.path.join(root[0:30], "lcam"))))
        self.input_size = input_size

    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        if idx % 2 ==0 :
            idx = int(self.__len__() / 2 + idx // 2)
        else:
            idx = int(self.__len__() / 2 - idx // 2)
        
        map_idx = idx % len(self.maps)
        lcam_idx = map_idx * self.num_of_each_map + idx // len(self.maps) 
        map_path = os.path.join(self.root, "map", self.maps[map_idx])
        lcam_path = os.path.join(self.root[0:30], "lcam", self.lcams[lcam_idx])
        img_map = cv2.resize(cv2.imread(map_path, 0), (self.input_size,self.input_size))
        img_lcam = cv2.resize(cv2.imread(lcam_path, 0), (self.input_size,self.input_size))
        if self.transforms is not None:
            img_map = self.transforms(img_map)
            img_lcam = self.transforms(img_lcam)
        
        return img_map, img_lcam

# stage2用到的数据集
# num_of_each_map：每一个map有的lcam数量
# num_of_each_H：每一个视角有多少张图
# H_of_each_map：每一个map的lcam有多少个视角变换
class PatchWithHomographDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train, gray=0, num_of_each_map=1, num_of_each_H=1, H_of_each_map=1, input_size=32):
        self.root = root
        self.transforms = transforms
        self.num_of_each_map = num_of_each_map
        self.maps = list(sorted(os.listdir(os.path.join(root, "map"))))
        self.lcams = list(sorted(os.listdir(os.path.join(root, "lcam"))))
        f = open(os.path.join(root, "H_info.txt"), 'r', encoding="utf-8")
        self.H_info = f.readlines()
        f.close()
        self.input_size = input_size
        self.num_of_each_H = num_of_each_H
        self.H_of_each_map = H_of_each_map
        self.gray=gray
        self.train = train

    def __len__(self):
        return len(self.lcams)
    
    def __getitem__(self, idx):
        map_idx = idx % len(self.maps)
        lcam_idx = map_idx * self.num_of_each_map + idx // len(self.maps) 
        H_info_idx = map_idx * self.H_of_each_map + (idx // len(self.maps)) // self.num_of_each_H
        map_path = os.path.join(self.root, "map", self.maps[map_idx])
        lcam_path = os.path.join(self.root, "lcam", self.lcams[lcam_idx])
        img_map = cv2.resize(cv2.imread(map_path, self.gray), (self.input_size,self.input_size))
        img_lcam = cv2.resize(cv2.imread(lcam_path, self.gray), (self.input_size,self.input_size))
        """
        dx = np.random.randint(0, 13)
        dy = np.random.randint(0, 13)
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])
        img_lcam = cv2.resize(cv2.warpAffine(cv2.imread(lcam_path, self.gray), MAT, (128,128)), (self.input_size,self.input_size))
        """
        if self.train:
            H_info = np.array(self.H_info[H_info_idx].strip().split(" ")[2:], 
                              dtype=np.float32).reshape((3,3))
        else:
            H_info = np.array(self.H_info[H_info_idx].strip().split(" ")[1:], 
                              dtype=np.float32).reshape((3,3))
            
        center_point = np.matmul(H_info, np.array([64, 64, 1]).reshape(3,1))
        center_point_ratio = torch.Tensor(((center_point / center_point[2,0])[:2,0] - 64) / 128)
        """
        center_point_ratio[0] = center_point_ratio[0] + dx / 128
        center_point_ratio[1] = center_point_ratio[1] + dy / 128
        """
        H = torch.Tensor(H_info)
        if self.transforms is not None:
            img_map = self.transforms(img_map)
            img_lcam = self.transforms(img_lcam)
        
        return img_map, img_lcam, H, center_point_ratio

# 最终inference time用的测试数据集
class InferenceTimeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, patch_size=128):
        self.root = root
        self.transforms = transforms
        self.patch_size = patch_size
        self.maps = list(sorted(os.listdir(os.path.join(root, "map"))))
        self.lcams = list(sorted(os.listdir(os.path.join(root, "lcam"))))
        f = open(os.path.join(root, "H_info.txt"), 'r', encoding="utf-8")
        self.H_info = f.readlines()
        f.close()

    def __len__(self):
        return len(self.lcams)
    
    def __getitem__(self, idx):
        map_path = os.path.join(self.root, "map", self.maps[idx])
        lcam_path = os.path.join(self.root, "lcam", self.lcams[idx])
        img_map = cv2.imread(map_path, 0)
        img_lcam = cv2.imread(lcam_path, 0)
        H_info = np.array(self.H_info[idx].strip().split(" ")[3:], 
                          dtype=np.float32).reshape((3,3))
        H = torch.Tensor(H_info)
        corner_point = np.array(self.H_info[idx].strip().split(" ")[1:3], dtype=np.int32)
        r, c = (corner_point + 1) // self.patch_size
        match_labels = torch.Tensor(img_lcam.shape[0] // self.patch_size, img_lcam.shape[1] // self.patch_size)
        for i in range(match_labels.shape[0]):
            for j in range(match_labels.shape[1]):
                match_labels[i, j] = (r + i) * (img_map.shape[1] // self.patch_size) + c + j
        
        if self.transforms is not None:
            img_map = self.transforms(img_map)
            img_lcam = self.transforms(img_lcam)
        
        return img_map, img_lcam, H, match_labels.flatten().squeeze()

# 用于双塔模型训练中获取正负样本均等的batch
def get_samples_siam(batch_map, batch_lcam):
    size = len(batch_map)
    labels_former = torch.ones(size, dtype = torch.float32)
    idx = np.random.randint(0, size, size=size)
    batch_lcam_latter = batch_lcam[list(idx)]
    labels_latter = torch.eq(torch.Tensor(idx), torch.linspace(0, size-1, size)).type(torch.float32)
    labels_latter = torch.where(labels_latter>0, labels_latter, -torch.ones(size, dtype = torch.float32))
    batch_map_new = torch.cat((batch_map, batch_map), 0)
    batch_lcam_new = torch.cat((batch_lcam, batch_lcam_latter), 0)
    labels = torch.cat((labels_former, labels_latter), 0)
    return batch_map_new, batch_lcam_new, labels

def get_samples_triplet(des_map, des_lcam):
    D = 2 * (1 - torch.matmul(des_map, torch.transpose(des_lcam, 0, 1)))
    _, min_c = torch.min(D, dim=1)
    for i in range(len(D)):
        if min_c[i] == i:
            if i != 0:
                tmpval1, tmpindex1 = torch.min(D[i][0:i], dim=0)
                if i+1 < len(D):
                    tmpval2, tmpindex2 = torch.min(D[i][i+1:], dim=0)
                    if tmpval1 < tmpval2:
                        min_c[i] = tmpindex1
                    else:
                        min_c[i] = tmpindex2+i+1
                else:
                    min_c[i] = tmpindex1
            else:
                tmpval2, tmpindex2 = torch.min(D[i][i+1:], dim=0)
                min_c[i] = tmpindex2+i+1
    return D, des_lcam[min_c.tolist()]

def input_normalize(tensor):
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
    #return (tensor - tensor.mean()) / tensor.std()
        
def get_transforms(train=False):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    #transforms.append(torchvision.transforms.Lambda(lambda tensor:input_normalize(tensor)))
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
        #transforms.append(torchvision.transforms.RandomVerticalFlip(0.5))
        #transforms.append(torchvision.transforms.RandomRotation(180))
    return torchvision.transforms.Compose(transforms)

# 只适用于inference time时候
def split_patch(img_tensor, patch_size):
    N, C, H, W = img_tensor.shape
    img_tensor_new = torch.Tensor((H // patch_size) * (W // patch_size), C, patch_size, patch_size)
    for idx in range(img_tensor_new.shape[0]):
        i = idx // (W // patch_size)
        j = idx % (W // patch_size)
        img_tensor_new[idx] = img_tensor[0, :, i*patch_size : (i+1)*patch_size, j*patch_size: (j+1)*patch_size]
    
    return img_tensor_new    

if __name__ == "__main__":
    print("This is a data util!")











