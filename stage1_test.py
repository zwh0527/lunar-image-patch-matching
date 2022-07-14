# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:12:56 2022

@author: knight
"""

# 专门用于stage1模型的精度评测

import os
import torch
import argparse
import numpy as np
from model_util import L2Net, L2AttentionNetv1, L2FusionNetv3, DenseAttentionNetv2, L2Net128
from other_model_util import MatchNet, MatchNet_2ch, MatchNet_2ch2stream, L2Net_2ch2stream
from data_util import ImagePatchDataset, get_transforms, get_samples_siam, get_samples_triplet
from train_test_util import get_loss_relative, get_loss_relative_2ch2stream, get_acc, get_loss_siam, get_acc_siam, get_loss_triplet, get_acc_matchnet, get_loss_matchnet, get_acc_matchnet_2ch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for stage1_model')
    parser.add_argument("-root", type=str, default="E:/data_of_bishe/change2")
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-feature_dims", type=int, default=128)
    parser.add_argument("-margin", type=float, default=1.0)
    parser.add_argument("-loss_type", type=int, default=1)
    parser.add_argument("-BN_track", type=bool, default=False)
    parser.add_argument("-shuffle", type=bool, default=True)
    parser.add_argument("-times", type=int, default=3)
    parser.add_argument("-kernel_channels", type=int, default=40)
    parser.add_argument("-kernel_size1", type=int, default=5)
    parser.add_argument("-kernel_size2", type=int, default=5)
    parser.add_argument("-kernel_size3", type=int, default=3)
    parser.add_argument("-out_channels", type=int, default=320) 
    args = parser.parse_args()
    
    test_name = ["test5", "test5_bright_1", "test5_bright_2", "test5_bright_3",
                 "test5_view_1", "test5_view_2", "test5_view_3",
                 "test5_texture_1", "test5_texture_2", "test5_texture_3"]
    
    for name in test_name[0:1]:
        root_test = os.path.join(args.root, name)
        root_model = os.path.join(args.root, "model")
        
        #net = MatchNet(BN_track=args.BN_track).to("cuda")
        #net = MatchNet_2ch(BN_track=args.BN_track).to("cuda")
        #net = MatchNet_2ch2stream(BN_track=args.BN_track).to("cuda")
        #net = L2Net_2ch2stream(BN_track=args.BN_track).to("cuda")
        #net = L2Net128(BN_track=args.BN_track).to("cuda")
        #net = L2FusionNetv3(kernel_channels=args.kernel_channels, kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2, kernel_size3=args.kernel_size3, out_channels=args.out_channels, BN_track=args.BN_track).to("cuda")
        #net = L2Net(kernel_channels=args.kernel_channels, kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2, kernel_size3=args.kernel_size3, BN_track=args.BN_track).to("cuda")
        #net = L2AttentionNetv1(kernel_channels=args.kernel_channels, kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2, kernel_size3=args.kernel_size3, BN_track=args.BN_track).to("cuda")
        net = DenseAttentionNetv2(kernel_channels=args.kernel_channels, kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2, kernel_size3=args.kernel_size3, out_channels=args.out_channels, BN_track=args.BN_track).to("cuda")
        test_data = ImagePatchDataset(root_test, get_transforms(False), 1, net.input_size)
        batch_size = args.batch_size
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=args.shuffle)
        net_dict = torch.load(os.path.join(root_model, "DenseAttentionNet/triplet/输出维度对比/320/DenseAttentionNetv2_epoch_20.pth"))
        
        if args.BN_track is True:
            """
            BN_key = ["feature_network.1.running_mean", "feature_network.1.running_var", 
                      "feature_network.1.num_batches_tracked", 
                      "feature_network.5.running_mean", 
                      "feature_network.5.running_var", 
                      "feature_network.5.num_batches_tracked", 
                      "feature_network.9.running_mean", 
                      "feature_network.9.running_var", 
                      "feature_network.9.num_batches_tracked", 
                      "feature_network.12.running_mean", 
                      "feature_network.12.running_var", 
                      "feature_network.12.num_batches_tracked", 
                      "feature_network.15.running_mean", 
                      "feature_network.15.running_var", 
                      "feature_network.15.num_batches_tracked"]"""
            
            """
            BN_key = ["feature_network1.1.running_mean", 
                      "feature_network1.1.running_var", 
                      "feature_network1.1.num_batches_tracked", 
                      "feature_network1.5.running_mean", 
                      "feature_network1.5.running_var", 
                      "feature_network1.5.num_batches_tracked", 
                      "feature_network1.9.running_mean", 
                      "feature_network1.9.running_var", 
                      "feature_network1.9.num_batches_tracked", 
                      "feature_network1.12.running_mean", 
                      "feature_network1.12.running_var", 
                      "feature_network1.12.num_batches_tracked", 
                      "feature_network1.15.running_mean", 
                      "feature_network1.15.running_var", 
                      "feature_network1.15.num_batches_tracked", 
                      "feature_network2.1.running_mean", 
                      "feature_network2.1.running_var", 
                      "feature_network2.1.num_batches_tracked", 
                      "feature_network2.5.running_mean", 
                      "feature_network2.5.running_var", 
                      "feature_network2.5.num_batches_tracked", 
                      "feature_network2.9.running_mean", 
                      "feature_network2.9.running_var", 
                      "feature_network2.9.num_batches_tracked", 
                      "feature_network2.12.running_mean", 
                      "feature_network2.12.running_var", 
                      "feature_network2.12.num_batches_tracked", 
                      "feature_network2.15.running_mean", 
                      "feature_network2.15.running_var", 
                      "feature_network2.15.num_batches_tracked"]"""
            
            """
            BN_key = ["feature_network1.1.running_mean", 
                      "feature_network1.1.running_var", 
                      "feature_network1.1.num_batches_tracked", 
                      "feature_network1.4.running_mean", 
                      "feature_network1.4.running_var", 
                      "feature_network1.4.num_batches_tracked", 
                      "feature_network1.7.running_mean", 
                      "feature_network1.7.running_var", 
                      "feature_network1.7.num_batches_tracked", 
                      "feature_network1.10.running_mean", 
                      "feature_network1.10.running_var", 
                      "feature_network1.10.num_batches_tracked", 
                      "feature_network1.13.running_mean", 
                      "feature_network1.13.running_var", 
                      "feature_network1.13.num_batches_tracked", 
                      "feature_network1.16.running_mean", 
                      "feature_network1.16.running_var", 
                      "feature_network1.16.num_batches_tracked", 
                      "feature_network1.19.running_mean", 
                      "feature_network1.19.running_var", 
                      "feature_network1.19.num_batches_tracked", 
                      "feature_network2.1.running_mean", 
                      "feature_network2.1.running_var", 
                      "feature_network2.1.num_batches_tracked", 
                      "feature_network2.4.running_mean", 
                      "feature_network2.4.running_var", 
                      "feature_network2.4.num_batches_tracked", 
                      "feature_network2.7.running_mean", 
                      "feature_network2.7.running_var", 
                      "feature_network2.7.num_batches_tracked", 
                      "feature_network2.10.running_mean", 
                      "feature_network2.10.running_var", 
                      "feature_network2.10.num_batches_tracked", 
                      "feature_network2.13.running_mean", 
                      "feature_network2.13.running_var", 
                      "feature_network2.13.num_batches_tracked", 
                      "feature_network2.16.running_mean", 
                      "feature_network2.16.running_var", 
                      "feature_network2.16.num_batches_tracked", 
                      "feature_network2.19.running_mean", 
                      "feature_network2.19.running_var", 
                      "feature_network2.19.num_batches_tracked"]"""
            
            """
            BN_key = ["BN1.running_mean", "BN1.running_var", 
                      "BN1.num_batches_tracked", "BN2.running_mean", 
                      "BN2.running_var", "BN2.num_batches_tracked", 
                      "BN3.running_mean", "BN3.running_var", 
                      "BN3.num_batches_tracked", "BN4.running_mean", 
                      "BN4.running_var", "BN4.num_batches_tracked", 
                      "BN5.running_mean", "BN5.running_var", 
                      "BN5.num_batches_tracked", "BN6.running_mean", 
                      "BN6.running_var", "BN6.num_batches_tracked", 
                      "BN7.running_mean", "BN7.running_var", 
                      "BN7.num_batches_tracked"]"""
            
            """
            BN_key = ["BN1.running_mean", "BN1.running_var", 
                      "BN1.num_batches_tracked", "BN2.running_mean", 
                      "BN2.running_var", "BN2.num_batches_tracked", 
                      "BN3.running_mean", "BN3.running_var", 
                      "BN3.num_batches_tracked", "BN4.running_mean", 
                      "BN4.running_var", "BN4.num_batches_tracked", 
                      "BN5.running_mean", "BN5.running_var", 
                      "BN5.num_batches_tracked", "BN6.running_mean", 
                      "BN6.running_var", "BN6.num_batches_tracked", 
                      "BN7.running_mean", "BN7.running_var", 
                      "BN7.num_batches_tracked", "transformer.1.running_mean", 
                      "transformer.1.running_var", 
                      "transformer.1.num_batches_tracked"]"""
            
            """
            BN_key = ["BN1.running_mean", "BN1.running_var", 
                      "BN1.num_batches_tracked", "BN2.running_mean", 
                      "BN2.running_var", "BN2.num_batches_tracked", 
                      "BN3.running_mean", "BN3.running_var", 
                      "BN3.num_batches_tracked", "BN4.running_mean", 
                      "BN4.running_var", "BN4.num_batches_tracked", 
                      "BN5.running_mean", "BN5.running_var", 
                      "BN5.num_batches_tracked", "BN6.running_mean", 
                      "BN6.running_var", "BN6.num_batches_tracked", 
                      "BN7.running_mean", "BN7.running_var", 
                      "BN7.num_batches_tracked", 
                      "feature_compress.1.running_mean", 
                      "feature_compress.1.running_var", 
                      "feature_compress.1.num_batches_tracked"]"""
            
            
            BN_key = ["BN1.running_mean", "BN1.running_var", 
                      "BN1.num_batches_tracked", "BN2.running_mean", 
                      "BN2.running_var", "BN2.num_batches_tracked", 
                      "BN3.running_mean", "BN3.running_var", 
                      "BN3.num_batches_tracked", "BN4.running_mean", 
                      "BN4.running_var", "BN4.num_batches_tracked", 
                      "BN5.running_mean", "BN5.running_var", 
                      "BN5.num_batches_tracked", "BN6.running_mean", 
                      "BN6.running_var", "BN6.num_batches_tracked", 
                      "BN7.running_mean", "BN7.running_var", 
                      "BN7.num_batches_tracked", "transformer.1.running_mean", 
                      "transformer.1.running_var", 
                      "transformer.1.num_batches_tracked", 
                      "feature_compress.1.running_mean", 
                      "feature_compress.1.running_var", 
                      "feature_compress.1.num_batches_tracked"]
            
            """
            BN_key = ["block1_1.1.running_mean", "block1_1.1.running_var", 
                      "block1_1.1.num_batches_tracked", 
                      "block1_2.1.running_mean", "block1_2.1.running_var", 
                      "block1_2.1.num_batches_tracked", 
                      "downsampling1.1.running_mean", 
                      "downsampling1.1.running_var", 
                      "downsampling1.1.num_batches_tracked", 
                      "block2.1.running_mean", "block2.1.running_var", 
                      "block2.1.num_batches_tracked", 
                      "downsampling2.1.running_mean", 
                      "downsampling2.1.running_var", 
                      "downsampling2.1.num_batches_tracked", 
                      "block3.1.running_mean", "block3.1.running_var", 
                      "block3.1.num_batches_tracked"]"""
            
            
            for i in range(len(BN_key)):
                pop_val = net_dict.pop(BN_key[i])
          
        
        net.load_state_dict(net_dict)
        
        net.eval()
        with torch.no_grad():
            tmp_loss = 0
            tmp_acc = 0
            #tmp_acc_siam = 0
            for i in range(args.times):
                loss_test = []
                acc_test = []
                #acc_test_siam = []
                print("start test!")
                for j, (img_map, img_lcam) in enumerate(test_dataloader):
                    #img_map, img_lcam, labels = get_samples_siam(img_map, img_lcam)
                    img_map = img_map.to("cuda")
                    img_lcam = img_lcam.to("cuda")
                    #labels = labels.to("cuda")
                    
                    # 这是相对距离训练模型
                    """D, E1, E2, E3 = get_loss_relative(img_map, img_lcam, net, args.feature_dims)
                    E = (E1 + 50 * E2 * len(img_map) + 0.5 * E3) / len(img_map)
                    acc = get_acc(D)[0]"""
                    
                    # 这个是双塔描述符距离
                    """des_map = net(img_map)
                    des_lcam = net(img_lcam)
                    D, E = get_loss_siam(des_map, des_lcam, labels, margin=args.margin, loss_type=args.loss_type)
                    acc = get_acc(D[0:len(D)//2, 0:len(D)//2])[0]
                    acc_siam = get_acc_siam(D, labels, acc_margin=1)"""
                    
                    # 这个是三元组描述符
                    des_map = net(img_map)
                    des_lcam_pos = net(img_lcam)
                    D, des_lcam_neg = get_samples_triplet(des_map, des_lcam_pos)
                    E = get_loss_triplet(des_map, des_lcam_pos, des_lcam_neg, margin=args.margin)
                    acc = get_acc(D)[0]
                    
                    # 这个是基于度量的
                    """metric = net(img_map, img_lcam)
                    E = get_loss_matchnet(metric, labels)
                    acc = get_acc_matchnet(img_map[0:len(img_map)//2], img_lcam[0:len(img_lcam)//2], net)"""
                    
                    if np.isnan(E.item()):
                        print("There is a non-sensing batch")
                        continue
                    
                    loss_test.append(E.item())
                    acc_test.append(acc.item())
                    #acc_test_siam.append(acc_siam.item())
                    
                if len(loss_test) * len(acc_test) != 0:
                    avg_loss_test = sum(loss_test)/len(loss_test)
                    avg_acc_test = sum(acc_test)/len(acc_test)
                    #avg_acc_test_siam = sum(acc_test_siam)/len(acc_test_siam)
                    tmp_loss += avg_loss_test / args.times
                    tmp_acc += avg_acc_test / args.times
                    #tmp_acc_siam += avg_acc_test_siam / args.times
                    #print(f"siam test acc:{avg_acc_test_siam :>5f}")
                    print(f"test loss: {avg_loss_test :>7f} test acc:{avg_acc_test :>5f}")
        
        print(f"avg test loss: {tmp_loss :>7f} avg test acc: {tmp_acc :>7f}")
    
    
    
    
    

    

