# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:32:56 2022

@author: knight
"""

# 此文件专门用于对已经正常处理过的测试集数据集test5的LCAM再次读取后进行
# 强弱光照变换
# 再一次的视角变换
# 再一次的分辨率降低

import simu_data_generator_0
import cv2
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simu_test_data_generator')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-type", type=str, default="texture")
    args = parser.parse_args()
    
    root_read = os.path.join(args.root, "test5/origin")
    imglist = list(sorted(os.listdir(root_read)))
    
    for time in range(1,4):
        root_write = os.path.join(args.root, "test5_"+args.type+"_"+str(time) +"/map")
        
        for i in range(len(imglist)//10):
            img_path = os.path.join(root_read, imglist[i])
            img_origin = cv2.imread(img_path, 0)
            h, w = img_origin.shape
            # 对前半数据集施加固定变亮、后半数据集施加固定变暗
            if args.type == "bright":
                ratio = 0.85-0.15*time if i < len(imglist)//10//2 else 0.9+0.5*time
                img_new = np.uint8(np.power(np.float32(img_origin)/255.0, ratio)*255.0)
            
            # 通过一个参数1.5，模板大小9*9的高斯滤波降低LCAM的清晰度
            elif args.type == "texture":
                #sigma = 1 + 0.5*time
                #img_new = cv2.GaussianBlur(img_origin, (9,9), sigma)
                ratio = 1/(time+8)
                img_new = simu_data_generator_0.scale_transform(img_origin, ratio)
                
            # 在LCAM原视角变换的基础上，再次施加一次视角变换
            elif args.type == "view":
                H_transform_ratio = 0.7-0.1*time
                img_new, _ = simu_data_generator_0.H_transformv2(img_origin, H_transform_ratio)
                
            elif args.type == "rotate":
                angle = np.random.randint(-4, 5) * 45
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img_new = cv2.warpAffine(img_origin, M, (w, h))
                
            else:
                raise Exception("指定鲁棒性验证类型！")
            
            cv2.imwrite(os.path.join(root_write, imglist[i]), img_new)
            
        
    
    
    










