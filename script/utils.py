import os
import cv2
import random
import numpy as np

import torch
from model.utils import BBoxTransform, ClipBoxes, nms_with_class, nms_without_class, soft_nms

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

def predict_single_imgs(model, img_path, cfg):
    img = cv2.imread(img_path)[..., ::-1]
    normalized_imgs = (img / 255 - cfg.mean) / cfg.std 

    ori_h, ori_w = normalized_imgs.shape[:2]
    scale = cfg.inference_resize_wh[1] / ori_h
    if scale * ori_w > cfg.inference_resize_wh[0]:
        scale = cfg.inference_resize_wh[0] / ori_w
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)
    img_resized = cv2.resize(normalized_imgs, (new_w, new_h))

    img_paded = np.zeros([cfg.inference_resize_wh[1], cfg.inference_resize_wh[0], 3])
    img_paded[:new_h, :new_w, :] = img_resized

    img_input = torch.from_numpy(img_paded).permute(2, 0, 1).unsqueeze(dim=0).float()
    
    if torch.cuda.is_available():
        img_input = img_input.cuda()
    _, regression, classification, anchors = model(img_input)

    trans_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(trans_anchors, img_input)

    scores, classes = torch.max(classification, dim=2)  # 计算anchor的最高得分和对应的类别
    over_score_thresh_mask = (scores > cfg.score_thresh)

    if over_score_thresh_mask.sum() < 1:
        print('No object fund!')
        return [], [], []
    
    scores_over_thresh = scores[over_score_thresh_mask]
    classes_over_thresh = classes[over_score_thresh_mask]
    anchors_over_thresh = transformed_anchors[over_score_thresh_mask, ...]
    
    
    if cfg.use_soft_nms:
        anchors_nms_idx, weighted_scores = soft_nms(anchors_over_thresh, scores_over_thresh.clone(), thresh=0.3, mode='linear')
    else:
        # anchors_nms_idx = nms_with_class(anchors_over_thresh, scores_over_thresh,classes_over_thresh, iou_threshold=0.2)
        anchors_nms_idx = nms_without_class(anchors_over_thresh, scores_over_thresh, iou_threshold=0.3)
    
    after_nms_anchors = anchors_over_thresh[anchors_nms_idx]
    after_nms_classes = classes_over_thresh[anchors_nms_idx]
    
    if cfg.use_soft_nms:
        after_nms_scores = weighted_scores
    else:
        after_nms_scores = scores_over_thresh[anchors_nms_idx]
    
    bbox_results = (after_nms_anchors / scale)
    bbox_results[:, 0] = torch.clamp(bbox_results[:, 0], min=0)
    bbox_results[:, 1] = torch.clamp(bbox_results[:, 1], min=0)
    bbox_results[:, 2] = torch.clamp(bbox_results[:, 2], max=ori_w-1)
    bbox_results[:, 3] = torch.clamp(bbox_results[:, 3], max=ori_h-1)
    
    bbox_results = bbox_results.cpu().detach().numpy()
    classes_results = after_nms_classes.cpu().detach().numpy()
    scores_results = after_nms_scores.cpu().detach().numpy()

    return bbox_results, classes_results, scores_results

