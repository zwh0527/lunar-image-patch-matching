# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:33:51 2022

@author: knight
"""

# 用于评测基于区域的匹配方法，主要是模板匹配

import os
import torch
import argparse
import cv2
import time
from data_util import ImagePatchDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for area_based method')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-image_size", type=int, default=32)
    parser.add_argument("-shuffle", type=bool, default=False)
    parser.add_argument("-times", type=int, default=1)
    args = parser.parse_args()
    
    root_test = os.path.join(args.root, "test5")
    test_data = ImagePatchDataset(root_test, None, 1, args.image_size)
    batch_size = args.batch_size
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=args.shuffle)
    size = len(test_data)
    
    for i in range(args.times):
        acc_test = []
        tmp_acc = 0
        t1 = time.time()
        print("start test!")
        for j, (img_map_batch, img_lcam_batch) in enumerate(test_dataloader):
            batch_size = len(img_map_batch)
            res = torch.zeros(batch_size, batch_size)
            for k in range(len(img_map_batch)):
                for m in range(len(img_lcam_batch)):
                    res[k, m] = torch.Tensor([cv2.matchTemplate(img_map_batch[k].numpy(), img_lcam_batch[m].numpy(), cv2.TM_SQDIFF_NORMED).min()])
            
            _, max_c = torch.max(res, axis=0)
            mask = torch.linspace(0, batch_size-1, batch_size)
            acc_c = torch.eq(max_c, mask).type(torch.float32)
            acc_test.append(acc_c.mean().item())
            
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
    
    
            
            
    
    
    
    
    
    
    