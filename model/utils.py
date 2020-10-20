import itertools
import torch
import torch.nn as nn
import numpy as np

from torchvision.ops.boxes import batched_nms, nms

class BBoxTransform(nn.Module):
    def __init__(self):
        super(BBoxTransform, self).__init__()
        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2])

    def forward(self, anchors, regression):

        wa = anchors[..., 2] - anchors[..., 0]
        ha = anchors[..., 3] - anchors[..., 1]
        x_centers_a = anchors[..., 0] + 0.5 * wa
        y_centers_a = anchors[..., 1] + 0.5 * ha

        dx = regression[:, :, 0] * self.std[0]
        dy = regression[:, :, 1] * self.std[1]
        dw = regression[:, :, 2] * self.std[2]
        dh = regression[:, :, 3] * self.std[3]

        pre_ctr_x = x_centers_a + dx * wa
        pre_ctr_y = y_centers_a + dy * ha
        pred_w = torch.exp(dw) * wa
        pred_h = torch.exp(dh) * ha

        xmin = pre_ctr_x - 0.5 * pred_w
        ymin = pre_ctr_y - 0.5 * pred_h
        xmax = pre_ctr_x + 0.5 * pred_w
        ymax = pre_ctr_y + 0.5 * pred_h

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)  # 修正超出边界的anchor [x1, y1, x2, y2]
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


def nms_with_class(boxes, scores, idxs, iou_threshold):
    return batched_nms(boxes, scores, idxs, iou_threshold)


def nms_without_class(boxes, scores, iou_threshold):
    return nms(boxes, scores, iou_threshold)


def soft_nms(dets, box_scores, sigma=0.5, thresh=0.1, mode="linear"):
    """A simple implement for soft-nms
    """
    assert mode in ["linear", 'gaussian']

    N = dets.shape[0]
    if torch.cuda.is_available():
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
        dets = dets.cuda()
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    if torch.cuda.is_available():
        scores = box_scores.cuda()
    else:
        scores = box_scores.cuda()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1
        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = torch.max(dets[i, 0], dets[pos:, 0])
        yy1 = torch.max(dets[i, 1], dets[pos:, 1])
        xx2 = torch.min(dets[i, 2], dets[pos:, 2])
        yy2 = torch.min(dets[i, 3], dets[pos:, 3])
        
        w = torch.max(torch.zeros_like(xx1), xx2 - xx1 + torch.ones_like(xx1))
        h = torch.max(torch.zeros_like(yy1), yy2 - yy1 + torch.ones_like(yy1))
        inter = w * h
        over = torch.div(inter, (areas[i] + areas[pos:] - inter))

        if mode == "linear":
            weight = 1 - over
            if torch.cuda.is_available():
                weight = weight.to(over)
        else:
            weight = torch.exp(-(over * over) / sigma)
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()
    return keep, scores[scores > thresh]
