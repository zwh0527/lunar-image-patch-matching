# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 20:20:31 2022

@author: knight
"""

# 用于评测基于特征的匹配方法，主要是SIFT等特征描述子

import os
import torch
import argparse
import cv2
import numpy as np
import time
from data_util import ImagePatchDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for area_based method')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-image_size", type=int, default=16)
    parser.add_argument("-shuffle", type=bool, default=False)
    parser.add_argument("-feature_type", type=str, default="orb")
    parser.add_argument("-times", type=int, default=1)
    args = parser.parse_args()
    
    root_test = os.path.join(args.root, "test5")
    test_data = ImagePatchDataset(root_test, None, 1, args.image_size)
    batch_size = args.batch_size
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size)
    size = len(test_data)
    kp = cv2.KeyPoint(x=(args.image_size-1)/2, y=(args.image_size-1)/2, _size=args.image_size)
    if args.feature_type == "sift":
        feature = cv2.xfeatures2d.SIFT_create()
    elif args.feature_type == "surf":
        feature = cv2.xfeatures2d.SURF_create()
    elif args.feature_type == "orb":
        feature = cv2.ORB_create()
    else:
        raise Exception("请指定特征类型！")
    
    for i in range(args.times):
        acc_test = []
        tmp_acc = 0
        t1 = time.time()
        print("start test!")
        for j, (img_map_batch, img_lcam_batch) in enumerate(test_dataloader):
            batch_size = len(img_map_batch)
            if args.feature_type == "sift":
                des_map = np.zeros((batch_size, 128))
                des_lcam = np.zeros((batch_size, 128))
            elif args.feature_type == "surf":
                des_map = np.zeros((batch_size, 64))
                des_lcam = np.zeros((batch_size, 64))
            else:
                des_map = np.zeros((batch_size, 32))
                des_lcam = np.zeros((batch_size, 32))
            
            for k in range(len(img_map_batch)):
                _, des_map[k] = feature.compute(img_map_batch[k].numpy(), [kp])
                des_map[k] = np.true_divide(des_map[k], np.linalg.norm(des_map[k]))
                _, des_lcam[k] = feature.compute(img_lcam_batch[k].numpy(), [kp])
                des_lcam[k] = np.true_divide(des_lcam[k], np.linalg.norm(des_lcam[k]))
                
            D = torch.Tensor(np.matmul(des_map, des_lcam.transpose()))
            _, max_c = torch.max(D, axis=0)
            mask = torch.linspace(0, batch_size-1, batch_size)
            acc_c = torch.eq(max_c, mask).type(torch.float32)
            acc_test.append(acc_c.mean())
            
            if j % 10 == 9:
                current = (j+1) * args.batch_size
                acc = sum(acc_test) / len(acc_test)
                print(f"acc: {acc :>5f} [{current :>5d}/{size :>5d}]")
            
        avg_acc_test = sum(acc_test)/len(acc_test)
        tmp_acc += avg_acc_test / args.times
        print(f"test acc:{avg_acc_test :>5f}")
        
    t2 = time.time()
    print(f"time: {t2-t1 :>6f}")
    print(f"avg test acc:{tmp_acc :>5f}")
    
    
    
    
    
    
    
    