import torch 
import torch.nn as nn
import numpy as np

def iou(a, g):
    """
    a: Anchors: [num_anchors, (x1, y1, x2, y2)]
    g: Gt: [num_gts, (x1, y1, x2, y2)]
    """
    dtype = a.dtype
    g = g.to(dtype)

    gt_area = (g[:, 2]  - g[:, 0]) * (g[:, 3] - g[:, 1])
    intersection_w = torch.min(torch.unsqueeze(a[:, 2], dim=1), g[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), g[:, 0])
    intersection_h = torch.min(torch.unsqueeze(a[:, 3], dim=1), g[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), g[:, 1])
    intersection_w = torch.clamp(intersection_w, min=0)
    intersection_h = torch.clamp(intersection_h, min=0)

    intersection_area = intersection_w * intersection_h

    union_area = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + gt_area - intersection_area
    union_area = torch.clamp(union_area, min=1e-6)

    IoU = intersection_area / union_area

    return IoU


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, regressions, anchors, annots):
        batch_size = classifications.shape[0]
        classification_loss = []
        regression_loss = []

        # Anchor -> [x1, y1, x2, y2]
        anchor = anchors[0, :, :]
        a_w = anchor[:, 2] - anchor[:, 0]
        a_h = anchor[:, 3] - anchor[:, 1]
        a_ctr_x = anchor[:, 0] + 0.5 * a_w
        a_ctr_y = anchor[:, 1] + 0.5 * a_h

        for i in range(batch_size):
            classification = classifications[i, :, :]
            regression = regressions[i, :, :]

            bbox_annot = annots[i]
            bbox_annot = bbox_annot[bbox_annot[:, 4] != -1]  # remove padding annots

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)  # clamp classification scores

            if bbox_annot.shape[0] == 0:  # no objects contained, only classification loss caled.
                alpha_factor = torch.ones_like(classification) * self.alpha

                if torch.cuda.is_available:
                    alpha_factor.cuda()

                alpha_factor = 1.0 - alpha_factor

                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                ce_loss = -(torch.log(1.0 - classification))
                cls_loss = focal_weight * ce_loss
                
                if torch.cuda.is_available(): 
                    regression_loss.append(torch.tensor(0.0).cuda())
                else:
                    regression_loss.append(torch.tensor(0.0))

                classification_loss.append(cls_loss.sum())
                continue

            IoU = iou(anchor[:, :], bbox_annot[:, :4])  # [anchor_nums, annot_nums]
            # get gt corresponding to anchors
            # IoU_max -> [anchor_nums]: match max iou with assigned gt
            # IoU_argmax -> [anchor_nums]: assigne gt to anchors
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) 

            # classification loss
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
            
            targets[torch.lt(IoU_max, 0.4), :] = 0  # anchors with iou < 0.4 as negetive samples
            positive_idx = torch.ge(IoU_max, 0.5)  # get positive sampels index
            positive_anchor_nums = positive_idx.sum() 
            assigned_annotations = bbox_annot[IoU_argmax, :]  # assign gt for every anchor

            targets[positive_idx, :] = 0  # reset
            targets[positive_idx, assigned_annotations[positive_idx, 4].long()] = 1  # the index for true id assign as 1

            alpha_factor = torch.ones_like(targets) * self.alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()
            
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)  # positive samples: alpha=0.25, negative samples: alpha=0.75
            focal_weight = torch.where(torch.eq(targets, 1.), 1.0 - classification, classification)  # for a positive sample, down-weight score as (1 - score)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)  # head factor for ce loss

            ce_loss = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * ce_loss

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)  # ignore anchors with iou in [0.4, 0.5] 
            classification_loss.append(cls_loss.sum() / torch.clamp(positive_anchor_nums, min=1.0))

            # bbox loss
            if positive_idx.sum().cpu().numpy() > 0:
                assigned_annotations = assigned_annotations[positive_idx, :]

                positive_a_w = a_w[positive_idx]
                positive_a_h = a_h[positive_idx]
                positive_a_ctr_x = a_ctr_x[positive_idx]
                positive_a_ctr_y = a_ctr_y[positive_idx]

                gt_w = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_h = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_w
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_h

                gt_w = torch.clamp(gt_w, min=1)
                gt_h = torch.clamp(gt_h, min=1)

                # encode
                targets_dx = (gt_ctr_x - positive_a_ctr_x) / positive_a_w
                targets_dy = (gt_ctr_y - positive_a_ctr_y) / positive_a_h
                targets_dw = torch.log(gt_w / positive_a_w)
                targets_dh = torch.log(gt_h / positive_a_h)
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t() 

                # retinanet style
                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_idx, :])
                # smooth-l1 loss
                reg_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0),
                                                0.5 * 9.0 * torch.pow(regression_diff, 2),
                                                regression_diff - 0.5 / 9.0)
                
                regression_loss.append(reg_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_loss.append(torch.tensor(0.0).cuda())
                else:
                    regression_loss.append(torch.tensor(0.0))
        
        return torch.stack(classification_loss).mean(dim=0, keepdim=True), \
               torch.stack(regression_loss).mean(dim=0, keepdim=True)
            





