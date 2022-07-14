# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:57:19 2022

@author: knight
"""

# 此文件是关于生成数据集的一些util function的实现

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


# 尺度变换函数
def scale_transform(img, ratio):
    img_small = cv2.resize(img, (0,0), fx=ratio, fy=ratio)
    img_small = cv2.resize(img_small, (img.shape[1], img.shape[0]))
    return img_small

# 函数内实现透视变换，视角变换增强
def H_transform(img, ratio):
    h, w = img.shape
    block_h = h * (1 - ratio) / 2
    block_w = w * (1 - ratio) / 2
    rand_x = np.random.randint(1, block_w, 4)
    rand_y = np.random.randint(1, block_h, 4)
    rand_zero = np.random.randint(0, 4)
    rand_x[rand_zero] = 0
    rand_y[rand_zero] = 0
    rand_x[(rand_zero + 2) % 4] = block_w
    rand_y[(rand_zero + 2) % 4] = block_h
    point1 = [rand_x[0], rand_y[0]]
    point2 = [w - rand_x[1], rand_y[1]]
    point3 = [w - rand_x[2], h - rand_y[2]]
    point4 = [rand_x[3], h - rand_y[3]]
    src_points = np.float32([point1, point2, point3, point4])
    dst_points = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H, _ = cv2.findHomography(src_points, dst_points)
    #center_origin = np.array([[(w-1)/2], [(h-1)/2], [1]])
    #center_target = np.dot(H, center_origin)
    #center_target /= center_target[2,0]
    img_H = cv2.warpPerspective(img, H, (w, h))
    return img_H, H

# 函数内实现透视变换，视角变换增强，为了验证鲁棒性而提出的一个版本
def H_transformv2(img, ratio):
    h, w = img.shape
    block_h = h * (1 - ratio) / 2
    block_w = w * (1 - ratio) / 2
    rand_zero = np.random.randint(0, 2, 4)
    point1 = [0 + block_w * rand_zero[0], 0 + block_h * rand_zero[0]]
    point2 = [w - block_w * rand_zero[1], 0 + block_h * rand_zero[1]]
    point3 = [w - block_w * rand_zero[2], h - block_h * rand_zero[2]]
    point4 = [0 + block_w * rand_zero[3], h - block_h * rand_zero[3]]
    src_points = np.float32([point1, point2, point3, point4])
    dst_points = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    H, _ = cv2.findHomography(src_points, dst_points)
    img_H = cv2.warpPerspective(img, H, (w, h))
    return img_H, H


# 对图像进行不同尺度的变换增强
def scale_enhancement(img, num, ratio):
    h, w = img.shape
    img_scale = np.zeros((num, h, w), np.uint8)
    ratios = np.random.rand(num) * ratio + 1 - ratio
    for i in range(num):
        img_scale[i] = scale_transform(img, ratios[i])
    return img_scale
   
# 对图像进行不同对比度的变换增强
def bright_enhancement(img, num):
    ratios = np.random.rand(num) * 0.7 + 0.7
    h, w = img.shape
    img_bright = np.zeros((num, h, w), np.float32)
    img = np.float32(img) / 255.0
    for i in range(num):
        img_bright[i] = np.power(img, ratios[i])
    return np.uint8(img_bright * 255.0)


if __name__ == "__main__":
    img = cv2.imread("C:\\Users\\knight\\Desktop\\white.jpg", flags = 0)
    plt.figure()
    plt.imshow(img, cmap = "gray")
    
    # 考虑了两次内参变换的H矩阵
    # 目前的结果是，无论K内参是多少，可以进行尺度变换+旋转变换的成像，但是难以处理图像块
    # 的选取
    cos_val = math.cos(math.pi * 1.0 / 180.0)
    sin_val = math.sin(math.pi * 1.0 / 180.0)
    Rx = np.array([[1, 0, 0], [0, cos_val, -sin_val], [0, sin_val, cos_val]])
    Ry = np.array([[cos_val, 0, sin_val], [0, 1, 0], [-sin_val, 0, cos_val]])
    Rz = np.array([[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]])
    R = np.dot(np.dot(Rx, Ry), Rz)
    t = np.array([[0], [0], [0]])
    K = np.array([[1000, 0, 299.5],[0, 1000, 399.5],[0, 0, 1]]) # 相机内参，f/DX随便设置，假设光点中心为图像中心
    K_inv = np.linalg.inv(K)
    H = np.dot(np.dot(K, R - np.dot(t, np.array([[0, 0, 1]]))), K_inv)
    img_H = cv2.warpPerspective(img, H, (600, 800))
    plt.figure()
    plt.imshow(img_H, cmap = "gray")
    
    
    # 考虑直接透视变换的H矩阵
    img_small = scale_transform(img, 0.2)
    plt.figure()
    plt.imshow(img_small, cmap = "gray")
    img_H, H = H_transform(img, 3/4)
    plt.figure()
    plt.imshow(img_H, cmap = "gray")
    img_scale = scale_enhancement(img_H, 2, 0.5)
    plt.figure()
    plt.imshow(np.squeeze(img_scale[0]), cmap = "gray")
    plt.figure()
    plt.imshow(np.squeeze(img_scale[1]), cmap = "gray")
    img_bright = bright_enhancement(img_H, 2)
    plt.figure()
    plt.imshow(np.squeeze(img_bright[0]), cmap = "gray")
    plt.figure()
    plt.imshow(np.squeeze(img_bright[1]), cmap = "gray")













