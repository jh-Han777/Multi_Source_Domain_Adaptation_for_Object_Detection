# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from msda.vgg16 import vgg16
from msda.utils import WeightEMA

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (
    EFocalLoss,
    FocalLoss,
    adjust_learning_rate,
    clip_gradient,
    load_net,
    save_checkpoint,
    save_net,
    weights_normal_init,
)
from roi_da_data_layer.roibatchLoader import roibatchLoader
#from lib.last.roibatchloader import roibatchLoader

from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from PIL import Image

print(sys.path)
torch.autograd.set_detect_anomaly(True)

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="cityscape",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )
    parser.add_argument(
        "--pretrained_path",
        dest="pretrained_path",
        help="vgg16, res101",
        default="",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to save checkpoint",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default=" ",
        type=str,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of worker to load data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether perform class_agnostic bbox regression",
        action="store_true",
    )

    # config optimization
    parser.add_argument(
        "--max_iter",
        dest="max_iter",
        help="max iteration for train",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="sgd", type=str
    )
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is iter",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--lamda", dest="lamda", help="DA loss param", default=0.1, type=float
    )

    # set training session
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )

    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--resume_name",
        dest="resume_name",
        help="resume checkpoint path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="resume from which model",
        default="",
        type=str,
    )

    # setting display config
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--ef",
        dest="ef",
        help="whether use exponential focal loss",
        action="store_true",
    )
    parser.add_argument(
        "--gamma", dest="gamma", help="value of gamma", default=1, type=float
    )
    parser.add_argument(
        "--max_epochs",
        dest="max_epochs",
        help="max epoch for train",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )

    parser.add_argument(
        "--eta",
        dest="eta",
        help="trade-off parameter between detection loss and domain-alignment loss."
        " Used for Car datasets",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--instance_da_eta",
        dest="instance_da_eta",
        help="instance_da_eta",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--burn_in",
        dest = "burn_in",
        default=10,
        type=int
    )

    parser.add_argument(
        "--weight",
        dest="weight",
        default=1,
        type=float
    )

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if args.dataset == "pascal_voc":
        print("loading our dataset...........")
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4,8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "cityscape":
        print("loading our dataset...........")
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscape_2007_test_t"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "rpc":
        print("loading our dataset...........")
        args.s_imdb_name = "rpc_fake_train"
        args.t_imdb_name = "rpc_val"
        # args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "rpc_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif args.dataset == "clipart":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_trainval"
        args.t_imdbtest_name = "clipart_trainval"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "water":
        print("loading our dataset...........")
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "water_train"
        args.t_imdbtest_name = "water_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    elif args.dataset == "sim10k":
        print("loading our dataset...........")
        args.s_imdb_name = "sim10k_2019_train"
        args.t_imdb_name = "cityscapes_car_2019_train"
        args.s_imdbtest_name = "sim10k_2019_val"
        args.t_imdbtest_name = "cityscapes_car_2019_val"

    elif args.dataset == "mskda_bdd":
        print("loading our dataset...........")
        args.s1_imdb_name = "bdd100k_daytime_train"
        args.s2_imdb_name = "bdd100k_night_train"
        args.s1_imdbtest_name = "bdd100k_daytime_val"
        args.s2_imdbtest_name = "bdd100k_night_val"
        args.t_imdb_name = "bdd100k_dd_train"
        args.t_imdbtest_name = "bdd100k_dd_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    s1_imdb, s1_roidb, s1_ratio_list, s1_ratio_index = combined_roidb(args.s1_imdb_name)
    s1_train_size = len(s1_roidb)  # add flipped         image_index*2

    s2_imdb, s2_roidb, s2_ratio_list, s2_ratio_index = combined_roidb(args.s2_imdb_name)
    s2_train_size = len(s2_roidb)  # add flipped         image_index*2

    t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
    t_train_size = len(t_roidb)  # add flipped         image_index*2

    print("source1 {:d} source2 {:d} target {:d} roidb entries".format(len(s1_roidb), len(s2_roidb), len(t_roidb)))

    # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    s1_sampler_batch = sampler(s1_train_size, args.batch_size)
    s2_sampler_batch = sampler(s2_train_size, args.batch_size)
    t_sampler_batch = sampler(t_train_size, args.batch_size)

    dataset_s1 = roibatchLoader(
        s1_roidb,
        s1_ratio_list,
        s1_ratio_index,
        args.batch_size,
        s1_imdb.num_classes,
        training=True,
    )

    dataloader_s1 = torch.utils.data.DataLoader(
        dataset_s1,
        batch_size=args.batch_size,
        sampler=s1_sampler_batch,
        num_workers=args.num_workers,
    )

    dataset_s2 = roibatchLoader(
        s2_roidb,
        s2_ratio_list,
        s2_ratio_index,
        args.batch_size,
        s2_imdb.num_classes,
        training=True,
    )

    dataloader_s2 = torch.utils.data.DataLoader(
        dataset_s2,
        batch_size=args.batch_size,
        sampler=s2_sampler_batch,
        num_workers=args.num_workers,
    )

    dataset_t = roibatchLoader(
        t_roidb,
        t_ratio_list,
        t_ratio_index,
        args.batch_size,
        t_imdb.num_classes,
        training=True,
    )

    dataloader_t = torch.utils.data.DataLoader(
        dataset_t,
        batch_size=args.batch_size,
        sampler=t_sampler_batch,
        num_workers=args.num_workers,
    )

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_cls_lb = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_cls_lb = Variable(im_cls_lb)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    if args.net == "vgg16":
        fasterRCNN = vgg16(
            t_imdb.classes,
            pretrained_path=args.pretrained_path,
            pretrained=True,
            class_agnostic=args.class_agnostic,
        )

    else:
        print("network is not defined")
        pdb.set_trace()

    # create model
    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        print(args.resume_name)
        load_name = os.path.join(output_dir, args.resume_name)
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        fasterRCNN.load_state_dict(checkpoint["model_faster"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    lmb1 = []
    lmb2 = []
    count_iter = 0

    scheduler = CosineAnnealingLR(optimizer, T_max=4, eta_min=0.0001)

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()

        loss_temp = 0
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s1 = iter(dataloader_s1)
        data_iter_s2 = iter(dataloader_s2)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s1 = next(data_iter_s1)
                data_s2 = next(data_iter_s2)
            except:
                data_iter_s1 = iter(dataloader_s1)
                data_iter_s2 = iter(dataloader_s2)

                data_s1 = next(data_iter_s1)
                data_s2 = next(data_iter_s2)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            # eta = 1.0
            count_iter += 1
            # put source data into variable

            with torch.no_grad():
                im_data.resize_(data_s1[0].size()).copy_(data_s1[0])
                im_info.resize_(data_s1[1].size()).copy_(data_s1[1])
                im_cls_lb.resize_(data_s1[2].size()).copy_(data_s1[2])
                gt_boxes.resize_(data_s1[3].size()).copy_(data_s1[3])
                num_boxes.resize_(data_s1[4].size()).copy_(data_s1[4])

            fasterRCNN.zero_grad()

            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls_s1,
                rpn_loss_bbox_s1,
                RCNN_loss_cls_s1,
                RCNN_loss_bbox_s1,
                rois_label,
                out_d_pixel_s1,
                out_d_s1,
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                subnet = "subnet1"
            )

            loss = (
                 rpn_loss_cls_s1.mean()
                + rpn_loss_bbox_s1.mean()
                + RCNN_loss_cls_s1.mean()
                + RCNN_loss_bbox_s1.mean()
            )
            loss_temp = loss.item()

            # domain label
            domain_s1 = Variable(torch.zeros(out_d_s1.size(0)).long().cuda())
            # global alignment loss
            dloss_s1 = 0.5 * FL(out_d_s1, domain_s1)
            # local alignment loss
            dloss_s_p1 = 0.5 * torch.mean(out_d_pixel_s1 ** 2)

            # Source2
            with torch.no_grad():
                im_data.resize_(data_s2[0].size()).copy_(data_s2[0])
                im_info.resize_(data_s2[1].size()).copy_(data_s2[1])
                im_cls_lb.resize_(data_s2[2].size()).copy_(data_s2[2])
                gt_boxes.resize_(data_s2[3].size()).copy_(data_s2[3])
                num_boxes.resize_(data_s2[4].size()).copy_(data_s2[4])

            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls_s2,
                rpn_loss_bbox_s2,
                RCNN_loss_cls_s2,
                RCNN_loss_bbox_s2,
                rois_label,
                out_d_pixel_s2,
                out_d_s2,
            ) = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                subnet="subnet2"
            )

            loss += (
                    rpn_loss_cls_s2.mean()
                    + rpn_loss_bbox_s2.mean()
                    + RCNN_loss_cls_s2.mean()
                    + RCNN_loss_bbox_s2.mean()
            )

            loss_temp += loss.item()

            # domain label
            domain_s2 = Variable(torch.zeros(out_d_s2.size(0)).long().cuda())
            # global alignment loss
            dloss_s2 = 0.5 * FL(out_d_s2, domain_s2)
            # local alignment loss
            dloss_s_p2 = 0.5 * torch.mean(out_d_pixel_s2 ** 2)



            with torch.no_grad():
                im_data.resize_(data_t[0].size()).copy_(data_t[0])
                im_info.resize_(data_t[1].size()).copy_(data_t[1])
                # gt is empty
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()

            out_d_pixel_t1, out_d_t1, = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                target=True,
                subnet="subnet1"
            )

            out_d_pixel_t2, out_d_t2, = fasterRCNN(
                im_data,
                im_info,
                gt_boxes,
                num_boxes,
                target=True,
                subnet="subnet2"
            )

            # domain label
            domain_t1 = Variable(torch.ones(out_d_t1.size(0)).long().cuda())
            dloss_t1 = 0.5 * FL(out_d_t1, domain_t1)
            # local alignment loss
            dloss_t_p1 = 0.5 * torch.mean((1 - out_d_pixel_t1) ** 2)

            # domain label
            domain_t2 = Variable(torch.ones(out_d_t2.size(0)).long().cuda())
            dloss_t2 = 0.5 * FL(out_d_t2, domain_t2)
            # local alignment loss
            dloss_t_p2 = 0.5 * torch.mean((1 - out_d_pixel_t2) ** 2)

            loss_low_align = dloss_s_p1 + dloss_s_p2 + dloss_t_p1 + dloss_t_p2
            loss_high_align1 = dloss_s1 + dloss_t1
            loss_high_align2 = dloss_s2 + dloss_t2

            lmb1.append(loss_high_align1)
            lmb2.append(loss_high_align2)
            loss_align = 0.5 * loss_low_align + 0.5 * (loss_high_align1 + loss_high_align2)

            if epoch > args.burn_in:
                lmb1_mean = round((sum(lmb1) / len(lmb1)),2)
                lmb2_mean = round((sum(lmb2) / len(lmb2)), 2)

                if epoch == args.burn_in +1 and step == 0:
                    fasterRCNN.step(lmb1_mean,lmb2_mean)

                align_cls, align_bbox = fasterRCNN(
                    im_data,
                    im_info,
                    gt_boxes,
                    num_boxes,
                    subnet="ema",
                )
                loss_rpn_align = align_cls + align_bbox
                fasterRCNN.step(lmb1_mean,lmb2_mean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval + 1

                #Megvii loss
                # source_ins_da_loss = source_ins_da.item() * args.instance_da_eta
                # target_ins_da_loss = target_ins_da.item() * args.instance_da_eta

                # loss_category_cls = category_loss_cls.item() #Megvii
                loss_rpn_cls = rpn_loss_cls_s1.item() + rpn_loss_cls_s2.item()
                loss_rpn_box = rpn_loss_bbox_s1.item() + rpn_loss_bbox_s2.item()
                loss_rcnn_cls = RCNN_loss_cls_s1.item() + RCNN_loss_cls_s2.item()
                loss_rcnn_box = RCNN_loss_bbox_s1.item() + RCNN_loss_bbox_s2.item()
                if epoch > args.burn_in:
                    loss_measure = loss_align.item()
                else:
                    loss_measure = 0
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel()

                print(
                    "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                    % (args.session, epoch, step, iters_per_epoch, loss_temp, lr)
                )
                print(
                    "\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start)
                )

                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f loss_distance: %.4f eta: %.4f"
                    % (
                        #loss_category_cls, #Megvii
                        loss_rpn_cls,
                        loss_rpn_box,
                        loss_rcnn_cls,
                        loss_rcnn_box,
                        loss_measure,
                        args.eta,
                    ),flush=True
                )

                loss_temp = 0
                start = time.time()
        if epoch % args.checkpoint_interval == 0 or epoch == args.max_epochs:
            save_name = os.path.join(
                output_dir, "{}.pth".format(args.dataset + "_" + str(epoch)),
            )
            save_checkpoint(
                {
                    "session": args.session,
                    "epoch": epoch + 1,
                    "model_faster": fasterRCNN.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "pooling_mode": cfg.POOLING_MODE,
                    "class_agnostic": args.class_agnostic,
                },
                save_name,
            )
            print("save model: {}".format(save_name))

        scheduler.step()
