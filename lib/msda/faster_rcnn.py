import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
#from model.rpn.rpn import _RPN
from msda.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (
    _affine_grid_gen,
    _affine_theta,
    _crop_pool_layer,
    _smooth_l1_loss,
    grad_reverse,
)
from torch.autograd import Variable
from msda.utils import get_max_iou, consist_loss

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, lc, gc):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc

        # define rpn
        self.RCNN_rpn1 = _RPN(self.dout_base_model)
        self.RCNN_rpn2 = _RPN(self.dout_base_model)
        self.RCNN_rpn_ema = _RPN(self.dout_base_model)

        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool1 = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_pool2 = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_pool_ema = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)

        self.RCNN_roi_align1 = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )
        self.RCNN_roi_align2 = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )
        self.RCNN_roi_align_ema = ROIAlign(
            (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0
        )

        self.grid_size = (
            cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        )

        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(
            self, im_data, im_info,  gt_boxes, num_boxes, target=False,subnet=False, eta=1.0
    ):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))
            # print(d_pixel)
            if not target:
                _, feat_pixel = self.netD_pixel(base_feat1.detach())
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))

        # base_feat = self.RCNN_base2(base_feat1)
        if subnet == "subnet1":
            base_feat = self.RCNN_base_sub1(base_feat1)

            if self.gc:
                domain_p, _ = self.netD1(grad_reverse(base_feat, lambd=eta))
                if target:
                    return d_pixel, domain_p  # , diff
                _, feat = self.netD1(base_feat.detach())
            else:
                domain_p = self.netD1(grad_reverse(base_feat, lambd=eta))
                if target:
                    return d_pixel, domain_p  # ,diff
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn1(
                base_feat, im_info, gt_boxes, num_boxes
            )

        elif subnet == "subnet2":
            base_feat = self.RCNN_base_sub2(base_feat1)

            if self.gc:
                domain_p, _ = self.netD2(grad_reverse(base_feat, lambd=eta))
                if target:
                    return d_pixel, domain_p  # , diff
                _, feat = self.netD2(base_feat.detach())
            else:
                domain_p = self.netD2(grad_reverse(base_feat, lambd=eta))
                if target:
                    return d_pixel, domain_p  # ,diff
            # feed base feature map tp RPN to obtain rois
            rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn2(
                base_feat, im_info, gt_boxes, num_boxes
            )

        elif subnet == "ema":
            base_feat = self.RCNN_base_sub_ema(base_feat1)

            if self.training:
                base_feat_1 = self.RCNN_base_sub1(base_feat1)
                base_feat_2 = self.RCNN_base_sub1(base_feat1)

                rpn_bbox_pred1 = self.RCNN_rpn1(
                    base_feat_1, im_info, gt_boxes, num_boxes
                )
                rpn_bbox_pred2 = self.RCNN_rpn2(
                    base_feat_2, im_info, gt_boxes, num_boxes
                )
                rpn_bbox_pred_ema = self.RCNN_rpn_ema(
                    base_feat, im_info, gt_boxes, num_boxes
                )

                consist_loss1 = consist_loss(rpn_bbox_pred1,rpn_bbox_pred_ema)
                consist_loss2 = consist_loss(rpn_bbox_pred2,rpn_bbox_pred_ema)

                loss = consist_loss1 + consist_loss2
                return loss

            else:
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn_ema(
                    base_feat, im_info, gt_boxes, num_boxes
                )

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2))
            )
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        if subnet == "subnet1":
            if cfg.POOLING_MODE == "align":
                pooled_feat = self.RCNN_roi_align1(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == "pool":
                pooled_feat = self.RCNN_roi_pool1(base_feat, rois.view(-1, 5))

            pooled_feat = self._head_to_tail1(pooled_feat)

        elif subnet == "subnet2":
            if cfg.POOLING_MODE == "align":
                pooled_feat = self.RCNN_roi_align2(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == "pool":
                pooled_feat = self.RCNN_roi_pool2(base_feat, rois.view(-1, 5))

            pooled_feat = self._head_to_tail2(pooled_feat)

        elif subnet == "ema":
            if cfg.POOLING_MODE == "align":
                pooled_feat = self.RCNN_roi_align_ema(base_feat, rois.view(-1, 5))
            elif cfg.POOLING_MODE == "pool":
                pooled_feat = self.RCNN_roi_pool_ema(base_feat, rois.view(-1, 5))

            pooled_feat = self._head_to_tail_ema(pooled_feat)

        else:
            raise Exception("Not Defined Pooling of Subnet")

        if self.lc:
            feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat_pixel, pooled_feat), 1)
        if self.gc:
            feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)
            # compute bbox offset

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )
            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(
                bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
            )

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return (
            rois,
            cls_prob,
            bbox_pred,
            # category_loss_cls, #Megvii
            rpn_loss_cls,
            rpn_loss_bbox,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
            d_pixel,
            domain_p,
        )  # ,diff

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean
                )  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn1.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn2.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_ema.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn1.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn2.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_ema.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn1.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn2.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_ema.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    @torch.no_grad()
    def step(self,lmb1,lmb2):
        self.RCNN_rpn_ema.RPN_Conv.weight = lmb1 * self.RCNN_rpn1.RPN_Conv.weight + lmb2 * self.RCNN_rpn2.RPN_Conv.weight
        self.RCNN_rpn_ema.RPN_cls_score.weight = lmb1 * self.RCNN_rpn1.RPN_cls_score.weight + lmb2 * self.RCNN_rpn2.RPN_cls_score.weight
        self.RCNN_rpn_ema.RPN_bbox_pred.weight = lmb1 * self.RCNN_rpn1.RPN_bbox_pred.weight + lmb2 * self.RCNN_rpn2.RPN_bbox_pred.weight

        self.RCNN_rpn_ema.RPN_Conv.weight.required_grad = False
        self.RCNN_rpn_ema.RPN_cls_score.weight.required_grad = False
        self.RCNN_rpn_ema.RPN_bbox_pred.weight.required_grad = False
        # for param in self.RCNN_rpn_ema.RPN_cls_score.parameters():
        #     param.required_grad = False
        # for param in self.RCNN_rpn_ema.RPN_Conv.parameters():
        #     param.required_grad = False
        # for param in self.RCNN_rpn_ema.RPN_bbox_pred.parameters():
        #     param.required_grad = False
