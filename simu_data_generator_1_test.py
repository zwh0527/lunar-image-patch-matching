# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:31:37 2022

@author: knight
"""

# 此文件专门用于生成验证集

import simu_data_generator_0
import cv2
import numpy as np
import argparse
import os

num_of_H_each_map = 1
num_of_scale_enhancement = 1
num_of_bright_enhancement = 1
H_transform_ratio = 0.6
lcam_scale_ratio = 0.3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simu_test_data_generator')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    args = parser.parse_args()
    
    global_map_path = os.path.join(args.root, "CE2_GRAS_DOM_07m_C103_66N142W_A.tif")
    global_map = cv2.imread(global_map_path, flags=0)
    
    map_h, map_w = global_map.shape
    img_h, img_w = 128, 128
    row_num = map_h // img_h
    col_num = map_w // img_w
    mapdir = os.path.join(args.root, "test5/map/")
    lcamdir = os.path.join(args.root, "test5/lcam/")
    origindir = os.path.join(args.root, "test5/origin/")
    H_info_path = os.path.join(args.root, "test5/")
    file = open(H_info_path + "H_info.txt", "w")
    
    count = 0
    for i in range(0, 100):
        for j in range(col_num):
            img_origin = cv2.equalizeHist(global_map[i*img_h : (i+1)*img_h, j*img_h : (j+1)*img_h])
            #cv2.imwrite(origindir + "origin_" + "{:0>7d}".format(count) + ".tif", img_origin)
            map_scale_ratio = 1 / np.random.randint(4,9)
            img_map = simu_data_generator_0.scale_transform(img_origin, map_scale_ratio)
            cv2.imwrite(mapdir + "map_" + "{:0>7d}".format(count) + ".tif", img_map)
            img_lcam, H = simu_data_generator_0.H_transform(img_origin, H_transform_ratio)
            img_lcam = np.squeeze(simu_data_generator_0.scale_enhancement(img_lcam, num_of_scale_enhancement, lcam_scale_ratio))
            img_lcam = np.squeeze(simu_data_generator_0.bright_enhancement(img_lcam, num_of_bright_enhancement))
            cv2.imwrite(lcamdir + "lcam_" + "{:0>7d}".format(count) + ".tif", img_lcam)
            H_info = " ".join(str(round(h,5)) for h in H.flatten())
            file.write(str(count) + " " + H_info + "\n")
            count += 1
    
    file.close()
