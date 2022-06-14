#!/bin/bash
save_dir="/media/hdd1/etri/save_model/train_adap_sw"
dataset="mskda_bdd"
net="vgg16"
pretrained_path="/media/hdd1/paper/CR-DA-DET/SW_Faster_ICR_CCR/pre_trained_model/vgg16_caffe.pth"
max_epoch=20
burn_in=10

CUDA_VISIBLE_DEVICES=2 python train_msda.py --cuda --dataset ${dataset} \
--net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --max_epoch ${max_epoch} --burn_in ${burn_in}\
 #>result_msda.txt 2>&1

#13849