# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:17:33 2022

@author: knight
"""

# 用于SIFT等加入特征学习的训练模板，分为基于度量和基于描述符两种

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
import cv2
from torch.utils.tensorboard import SummaryWriter
from model_util import l2_normalize
from other_model_util import Feature_Metric, Feature_Descriptor
from data_util import ImagePatchDataset, get_transforms, get_samples_siam
from train_test_util import get_loss_matchnet, get_acc_feature_metric, get_loss_siam, get_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train for features')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-data_root", type=str, default="/disk527/Datadisk/b527_zwh/bishe")
    parser.add_argument("-model_root", type=str, default="/disk527/Commondisk/b527_zwh/bishe/model/other_model")
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-input_size", type=int, default=32)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-pretrain", type=bool, default=False)
    parser.add_argument("-pretrain_epoch", type=int, default=0)
    parser.add_argument("-gpu", type=str, default="0")
    parser.add_argument("-nfeatures", type=int, default=10)
    parser.add_argument("-feature_type", type=str, default="sift")
    parser.add_argument("-model_type", type=str, default="descriptor")
    parser.add_argument("-margin", type=float, default=1.5)
    parser.add_argument("-loss_type", type=int, default=1)
    args = parser.parse_args()
    
    if args.gpu == "None":
        ex = Exception("指定GPU！")
        raise ex
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    start_epoch = args.pretrain_epoch
    root_train = os.path.join(args.root, "train3")
    root_test = os.path.join(args.root, "test5")
    root_model = args.root
    #root_log = os.path.join(args.model_root, "tensorboard_log_dir")
    
    if args.feature_type == "sift":
        feature = cv2.xfeatures2d.SIFT_create(nfeatures=args.nfeatures)
        feature_dims = args.nfeatures*128
    elif args.feature_type == "surf":
        feature = cv2.xfeatures2d.SURF_create(nfeatures=args.nfeatures)
        feature_dims = args.nfeatures*64
    elif args.feature_type == "orb":
        feature = cv2.ORB_create(nfeatures=args.nfeatures)
        feature_dims = args.nfeatures*32
    else:
        raise Exception("请指定特征类型！")
    
    if args.model_type == "metric":
        net = Feature_Metric(feature_dims).to("cuda")
    elif args.model_type == "descriptor":
        net = Feature_Descriptor(feature_dims).to("cuda")
    else:
        raise Exception("指定模型类型")
    
    if args.pretrain:
        net.load_state_dict(torch.load(os.path.join(root_model, "FeatureNet_epoch_" + str(args.pretrain_epoch) + ".pth")))
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr*math.pow(0.95, start_epoch//2), momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)
    num_of_each_map = simu_data_generator_1.num_of_H_each_map * (simu_data_generator_1.num_of_scale_enhancement + simu_data_generator_1.num_of_bright_enhancement)
    train_data = ImagePatchDataset(root_train, get_transforms(True), 1, args.input_size)
    test_data = ImagePatchDataset(root_test, get_transforms(False), 1, args.input_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, args.batch_size*2, shuffle=False)
    #writer = SummaryWriter(os.path.join(root_log, "Siamnet"))
    
    epochs = args.epochs
    size = len(train_data)
    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())
    print("This is a model training for FeatureNet!")
    f = open(os.path.join(root_model, "train.txt"), "a+")
    for epoch in range(start_epoch,epochs):
        print("Epoch %d" % (epoch + 1))
        f.write("Epoch %d\n" % (epoch + 1))
        net.train()
        loss_train = []
        acc_train = []
        loss = 0
        acc = 0
        count = 0
        optimizer.zero_grad()
        for i, (img_map, img_lcam) in enumerate(train_dataloader):
            img_map, img_lcam, labels = get_samples_siam(img_map, img_lcam)
            labels = labels.to("cuda")
            
            batch_size = len(img_map)
            des_map = l2_normalize(torch.ones(batch_size, feature_dims))
            des_lcam = l2_normalize(torch.ones(batch_size, feature_dims))
            for j in range(batch_size):
                _, tmp_des_map = feature.detectAndCompute((img_map[j,0]*255).numpy().astype(np.uint8), None)
                _, tmp_des_lcam = feature.detectAndCompute((img_lcam[j,0]*255).numpy().astype(np.uint8), None)
                if tmp_des_map is not None and len(tmp_des_map) == args.nfeatures:
                    des_map[j] = l2_normalize(torch.Tensor(tmp_des_map).view(1, feature_dims))
                if tmp_des_lcam is not None and len(tmp_des_lcam) == args.nfeatures:
                    des_lcam[j] = l2_normalize(torch.Tensor(tmp_des_lcam).view(1, feature_dims))
            
            des_map = des_map.to("cuda")
            des_lcam = des_lcam.to("cuda")
            
            if args.model_type == "metric":
                metric = net(des_map, des_lcam)
                E = get_loss_matchnet(metric, labels)
            else:
                des_map = net(des_map)
                des_lcam = net(des_lcam)
                D, E = get_loss_siam(des_map, des_lcam, labels, margin=args.margin, loss_type=args.loss_type)
                
            if np.isnan(E.item()):
                print("There is a non-sensing batch!")
                continue
            
            count += 1
            loss += E.item()
            if args.model_type == "metric":
                acc += get_acc_feature_metric(des_map[0:len(des_map)//2], des_lcam[0:len(des_lcam)//2], net).item()
            else:
                acc += get_acc(D[0:len(D)//2, 0:len(D)//2])[0].item()
            
            E.backward()
            
            if i % 10 == 9:
                optimizer.step()
                optimizer.zero_grad()
            
            if i % 50 == 49:
                current = (i+1) * args.batch_size
                loss_train.append(loss / count)
                acc_train.append(acc / count)
                #writer.add_scalar("loss/train", loss / count, 
                #                  epoch * len(train_dataloader) + i)
                #writer.add_scalar("accuracy/train", acc / count, 
                #                  epoch * len(train_dataloader) + i)
                print(f"loss: {loss/count :>7f} acc:{acc/count :>5f} [{current :>5d}/{size :>5d}]")
                f.write(f"loss: {loss/count :>7f} acc:{acc/count :>5f} [{current :>5d}/{size :>5d}]\n")
                loss = 0
                acc = 0
                count = 0
        
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        avg_loss_train = sum(loss_train)/len(loss_train)
        avg_acc_train = sum(acc_train)/len(acc_train)
        print(f"train loss: {avg_loss_train :>7f} train acc:{avg_acc_train :>5f}")
        f.write(f"train loss: {avg_loss_train :>7f} train acc:{avg_acc_train :>5f}\n")
        
        net.eval()
        with torch.no_grad():
            loss_test = []
            acc_test = []
            print("one epoch training done! start test!")
            for j, (img_map, img_lcam) in enumerate(test_dataloader):
                img_map, img_lcam, labels = get_samples_siam(img_map, img_lcam)
                labels = labels.to("cuda")
                
                batch_size = len(img_map)
                des_map = l2_normalize(torch.ones(batch_size, feature_dims))
                des_lcam = l2_normalize(torch.ones(batch_size, feature_dims))
                for j in range(batch_size):
                    _, tmp_des_map = feature.detectAndCompute((img_map[j,0]*255).numpy().astype(np.uint8), None)
                    _, tmp_des_lcam = feature.detectAndCompute((img_lcam[j,0]*255).numpy().astype(np.uint8), None)
                    if tmp_des_map is not None and len(tmp_des_map) == args.nfeatures:
                        des_map[j] = l2_normalize(torch.Tensor(tmp_des_map).view(1, feature_dims))
                    if tmp_des_lcam is not None and len(tmp_des_lcam) == args.nfeatures:
                        des_lcam[j] = l2_normalize(torch.Tensor(tmp_des_lcam).view(1, feature_dims))
                
                des_map = des_map.to("cuda")
                des_lcam = des_lcam.to("cuda")
                
                if args.model_type == "metric":
                    metric = net(des_map, des_lcam)
                    E = get_loss_matchnet(metric, labels)
                else:
                    des_map = net(des_map)
                    des_lcam = net(des_lcam)
                    D, E = get_loss_siam(des_map, des_lcam, labels, margin=args.margin, loss_type=args.loss_type)
                    
                if np.isnan(E.item()):
                    print("There is a non-sensing batch!")
                    continue
                
                loss_test.append(E.item())
                if args.model_type == "metric":
                    acc_test.append(get_acc_feature_metric(des_map[0:len(des_map)//2], des_lcam[0:len(des_lcam)//2], net).item())
                else:
                    acc_test.append(get_acc(D[0:len(D)//2, 0:len(D)//2])[0].item())
            
            if len(loss_test) * len(acc_test) != 0:
                avg_loss_test = sum(loss_test)/len(loss_test)
                avg_acc_test = sum(acc_test)/len(acc_test)
                #writer.add_scalar("loss/test", avg_loss_test, epoch)
                #writer.add_scalar("accuracy/test", avg_acc_test, epoch)
                print(f"test loss: {avg_loss_test :>7f} test acc:{avg_acc_test :>5f}")
                f.write(f"test loss: {avg_loss_test :>7f} test acc:{avg_acc_test :>5f}\n")
            
            if epoch % 2 == 1 and epoch >= 3:
                torch.save(net.state_dict(), os.path.join(root_model, "FeatureNet_epoch_" + str(epoch + 1) + ".pth"))
            
            if avg_acc_train >= 0.99:
                torch.save(net.state_dict(), os.path.join(root_model, "FeatureNet_epoch_" + str(epoch + 1) + ".pth"))
                print("acc in train has reached 99%")
                break
            
            if avg_acc_test > best_acc:
                best_acc = avg_acc_test
                #writer.add_scalar("best accuracy", best_acc, epoch)
                best_model = copy.deepcopy(net.state_dict())
                torch.save(best_model, os.path.join(root_model, "FeatureNet_best_model.pth"))
    f.close()



















