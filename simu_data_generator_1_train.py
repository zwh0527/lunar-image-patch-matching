# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:22:08 2022

@author: knight
"""

# 此文件专门用于生成训练集

import simu_data_generator_0
import numpy as np
import cv2
import argparse
import os

num_of_H_each_map = 1
num_of_scale_enhancement = 1
num_of_bright_enhancement = 1
H_transform_ratio = 0.6
lcam_scale_ratio = 0.3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simu_train_data_generator')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    args = parser.parse_args()
    
    global_map_path = os.path.join(args.root, "CE2_GRAS_DOM_07m_C102_66N157W_A.tif")
    global_map = cv2.imread(global_map_path, flags=0)
    
    map_h, map_w = global_map.shape
    img_h, img_w = 128, 128
    row_num = map_h // img_h
    col_num = map_w // img_w 
    mapdir = os.path.join(args.root, "train3/map/")
    lcamdir = os.path.join(args.root, "train3/lcam/")
    H_info_path = os.path.join(args.root, "train3/")
    file = open(H_info_path + "H_info.txt", "a")
    
    count = 62832
    for i in range(row_num):
        for j in range(col_num):
            img_origin = cv2.equalizeHist(global_map[i*img_h : (i+1)*img_h, j*img_h : (j+1)*img_h])
            map_scale_ratio =  1 / np.random.randint(4,9)
            img_map = simu_data_generator_0.scale_transform(img_origin, map_scale_ratio)
            cv2.imwrite(mapdir + "map_" + "{:0>7d}".format(count) + ".tif", img_map)
            for k in range(num_of_H_each_map):
                img, H = simu_data_generator_0.H_transform(img_origin, H_transform_ratio)
                #cv2.imwrite(lcamdir + "lcam_" + "{:0>6d}".format(count) + "_view_" + str(k) + "_origin" + ".tif", img)
                H_info = " ".join(str(round(h,5)) for h in H.flatten())
                file.write(str(count) + " " + str(k) + " " + H_info + "\n")
                img = np.squeeze(simu_data_generator_0.scale_enhancement(img, num_of_scale_enhancement, lcam_scale_ratio))
                img = np.squeeze(simu_data_generator_0.bright_enhancement(img, num_of_bright_enhancement))
                cv2.imwrite(lcamdir + "lcam_" + "{:0>7d}".format(count) + ".tif", img)
                #for m in range(num_of_scale_enhancement):
                #    cv2.imwrite(lcamdir + "lcam_" + "{:0>6d}".format(count) + "_view_" + str(k) + "_scale_" + str(m) + ".tif", 
                #                np.squeeze(img_scale[m]))
                #img_bright = simu_data_generator_0.bright_enhancement(
                #    img, num_of_bright_enhancement)
                #for m in range(num_of_bright_enhancement):
                #    cv2.imwrite(lcamdir + "lcam_" + "{:0>7d}".format(count) + "_view_" + str(k) + "_bright_" + str(m) + ".tif", 
                #                np.squeeze(img_bright[m]))
            count += 1
            if count % 1000 == 999:
                print("*", end="")
    
    file.close()











