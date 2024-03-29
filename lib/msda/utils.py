import torch

def get_max_iou(pred_bboxes, gt_bbox):
    '''
    :param pred_bboxs: [[x1, y1, x2, y2] [x1, y1, x2, y2],,,]
    :param gt_bbox: [x1, y1, x2, y2]
    :return:
    '''
    ixmin = torch.max(pred_bboxes[:, 1], gt_bbox[1])
    iymin = torch.max(pred_bboxes[:, 2], gt_bbox[2])
    ixmax = torch.min(pred_bboxes[:, 3], gt_bbox[3])
    iymax = torch.min(pred_bboxes[:, 4], gt_bbox[4])

    iws = torch.max(ixmax - ixmin + 1.0, torch.tensor(0.).cuda())
    ihs = torch.max(iymax - iymin + 1.0, torch.tensor(0.).cuda())

    inters = iws * ihs

    unis = (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.0) * (pred_bboxes[:, 4] - pred_bboxes[:, 2] + 1.0) + (
            gt_bbox[3] - gt_bbox[1] + 1.0) * (gt_bbox[4] - gt_bbox[2] + 1.0) - inters

    ious = inters / unis
    max_iou = torch.max(ious)
    max_index = torch.argmax(ious)

    return ious, max_iou, max_index

def consist_loss(pred, pred_ema):
    loss = 0
    pred = pred[0, :256, :]
    pred_ema = pred_ema[0, :256, :]
    
    for idx, gt in enumerate(pred_ema):     # pop best?
        ious, max_iou, max_index = get_max_iou(pred,gt)
        loss += abs(max_index - idx) * max_iou
    loss /= len(pred_ema)
    return loss



@torch.no_grad()
class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
