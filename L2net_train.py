# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:08:41 2022

@author: knight
"""

# 用于L2net的训练与测试

import os
import sys
sys.path.append(r"C:\Users\knight\Desktop\bishe\code_and_data\change2")
sys.path.append(r"C:\Users\knight\Desktop\bishe\code_and_data")
import simu_data_generator_1_train as simu_data_generator_1
import torch
import copy
import argparse
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from model_util import L2Net, L2AttentionNetv1, L2FusionNetv3, DenseAttentionNetv2, L2Net128, L2Netv1256, L2Netv2256, L2Netv1512, L2Netv2512, L2Net192v3, L2FusionNetv1128, L2FusionNetv2256, L2FusionNetv3256, L2FusionNetv3192, L2FusionNetv4256, L2AttentionNetv1128, L2AttentionNetv1256, L2AttentionNetv1192, L2AttentionNetv1320, DenseAttentionNetv2256, DenseAttentionNetv2512, DenseAttentionNetv2384, DenseAttentionNetv2640
from other_model_util import L2Net_2ch2stream
from data_util import ImagePatchDataset, get_transforms
from train_test_util import get_loss_relative, get_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for L2net')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=24)
    parser.add_argument("-feature_dims", type=int, default=128)
    args = parser.parse_args()
    
    start_epoch = 0
    root_train = os.path.join(args.root, "train3")
    root_test = os.path.join(args.root, "test5")
    root_model = os.path.join(args.root, "model")
    root_log = os.path.join(args.root, "tensorboard_log_dir")
    net = L2Net128().to("cuda")
    #net.load_state_dict(torch.load(os.path.join(root_model, "消融实验/L2FusionNet(direct_train)(前五轮不带res后面不小心带了res)/L2FusionNetv3_epoch_12.pth")))

    """net_dict1 = torch.load(os.path.join(root_model, "消融实验/L2net/L2net_epoch_25.pth"))
    net_dict2 = net.state_dict()
    for key in net_dict1:
        if key in net_dict2:
            net_dict2[key] = net_dict1[key]
            
    net.load_state_dict(net_dict2)
    
    for k, v in net.named_parameters():
        if k in net_dict1:
            v.requires_grad = False"""
    
    optimizer = torch.optim.SGD(net.parameters(), lr=4e-4*math.pow(0.95, start_epoch//2), momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)
    num_of_each_map = simu_data_generator_1.num_of_H_each_map * (simu_data_generator_1.num_of_scale_enhancement + simu_data_generator_1.num_of_bright_enhancement)
    train_data = ImagePatchDataset(root_train, get_transforms(True), 1, net.input_size)
    test_data = ImagePatchDataset(root_test, get_transforms(False), 1, net.input_size)
    batch_size = args.batch_size # 要保证这个量在train and test中不会大于map数量！！
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size * 2, shuffle=False)
    #writer = SummaryWriter(os.path.join(root_log, "L2net"))
    
    epochs = args.epochs
    size = len(train_data)
    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())
    print("This is a model training for L2net!")
    pred_avg_acc_train = 0
    stop_flag = 0
    for epoch in range(start_epoch,epochs):
        print("Epoch %d" % (epoch + 1))
        net.train()
        loss_train = []
        acc_train = []
        loss = 0
        acc = 0
        count = 0
        optimizer.zero_grad()
        for i, (img_map, img_lcam) in enumerate(train_dataloader):
            img_map = img_map.to("cuda")
            img_lcam = img_lcam.to("cuda")
            
            D, E1, E2, E3 = get_loss_relative(img_map, img_lcam, net, args.feature_dims)
            E = (E1 + 50 * E2 * len(img_map) + 0.5 * E3) / len(img_map)
            
            if np.isnan(E.item()):
                print("There is a non-sensing batch!")
                continue
            
            count += 1
            loss += E.item()
            acc += get_acc(D)[0].item()
            
            E.backward()
            
            if i % 10 == 9:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 50 == 49:
                current = (i+1) * batch_size
                loss_train.append(loss / count)
                acc_train.append(acc / count)
                #writer.add_scalar("loss/train", loss / count, 
                #                  epoch * len(train_dataloader) + i)
                #writer.add_scalar("accuracy/train", acc / count, 
                #                  epoch * len(train_dataloader) + i)
                print(f"loss: {loss/count :>7f} acc:{acc/count :>5f} [{current :>5d}/{size :>5d}]")
                loss = 0
                acc = 0
                count = 0
                
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        avg_loss_train = sum(loss_train)/len(loss_train)
        avg_acc_train = sum(acc_train)/len(acc_train)
        print(f"train loss: {avg_loss_train :>7f} train acc:{avg_acc_train :>5f}")
        
        net.eval()
        with torch.no_grad():
            loss_test = []
            acc_test = []
            print("one epoch training done! start test!")
            for j, (img_map, img_lcam) in enumerate(test_dataloader):
                img_map = img_map.to("cuda")
                img_lcam = img_lcam.to("cuda")
                
                D, E1, E2, E3 = get_loss_relative(img_map, img_lcam, net, args.feature_dims)
                E = (E1 + 50 * E2 * len(img_map) + 0.5 * E3) / len(img_map)
                
                if np.isnan(E.item()):
                    print("There is a non-sensing batch")
                    continue
                
                loss_test.append(E.item())
                acc_test.append(get_acc(D)[0].item())
            
            if len(loss_test) * len(acc_test) != 0:
                avg_loss_test = sum(loss_test)/len(loss_test)
                avg_acc_test = sum(acc_test)/len(acc_test)
                #writer.add_scalar("loss/test", avg_loss_test, epoch)
                #writer.add_scalar("accuracy/test", avg_acc_test, epoch)
                print(f"test loss: {avg_loss_test :>7f} test acc:{avg_acc_test :>5f}")
            
            if epoch % 2 == 1 and epoch >= 3:
                torch.save(net.state_dict(), os.path.join(root_model, "L2FusionNetv3_epoch_" + str(epoch + 1) + ".pth"))
            
            if avg_acc_train >= 0.98:
                torch.save(net.state_dict(), os.path.join(root_model, "L2FusionNetv3_epoch_" + str(epoch + 1) + ".pth"))
                print("acc in train has reached 98%")
                break
            
            if avg_acc_test > best_acc:
                best_acc = avg_acc_test
                #writer.add_scalar("best accuracy", best_acc, epoch)
                best_model = copy.deepcopy(net.state_dict())
                torch.save(best_model, os.path.join(root_model, "L2FusionNetv3_best_model.pth"))
                
            if abs(avg_acc_train - pred_avg_acc_train) < 0.0005:
                stop_flag += 1
                if stop_flag >= 2:
                    break
            else:
                stop_flag = 0
            pred_avg_acc_train = avg_acc_train
            
            
        
