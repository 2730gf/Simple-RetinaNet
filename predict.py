import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from model.retinanet import get_model
from model.utils import BBoxTransform, ClipBoxes, nms_with_class, nms_without_class

model_path = '/home/workspace/chencheng/Learning/ObjectDetection/retinanet.pytorch/checkpoints/mchar/epoch_11_loss_0.57797.pth'
retinanet = get_model(num_classes=10, pretrained=False)
retinanet.load_state_dict(torch.load(model_path))
retinanet.eval()

classes = {0: '1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9', 9:'0'}


def process_imgs(model, img_path, classes, score_thresh=0.3, \
                iou_thresh=0.3, nms_with_cls=True, \
                resize_wh=(768, 640), save_path='results/' ,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    img = cv2.imread(img_path)[..., ::-1]
    normalized_img = (img / 255 - mean) / std
    ori_h, ori_w = normalized_img.shape[:2]

    scale = resize_wh[1] / ori_h
    if scale * resize_wh[0] > ori_w:
        scale = resize_wh[0] / ori_w
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)
    img_resized = cv2.resize(normalized_img, (new_w, new_h))

    img_paded = np.zeros([resize_wh[1], resize_wh[0], 3])
    img_paded[:new_h, :new_w, :] = img_resized

    img_input = torch.from_numpy(img_paded).permute(2, 0, 1).unsqueeze(dim=0).float()
    features, regression, classification, anchors = model(img_input)

    # add predict offset
    trans_anchors = regressBoxes(anchors, regression)
    trans_anchors = clipBoxes(trans_anchors, img_input)

    scores = torch.max(classification, dim=2, keepdim=True)[0]  # 计算anchor的最高得分
    scores_over_thresh = (scores > score_thresh)[:, :, 0]

    top_cls = classification[0, scores_over_thresh[0, :], ...].permute(1, 0)  # get classes over thresh
    top_bbox = trans_anchors[0, scores_over_thresh[0, :], ...]  # get bbox over thresh
    top_scores = scores[0, scores_over_thresh[0, :], ...]  # get scores
    scores_, classes_ = top_cls.max(dim=0)  # score and label_id for every bbox

    if nms_with_cls:
        anchors_nms_idx = nms_with_class(top_bbox, top_scores[:, 0], classes_, iou_threshold=iou_thresh)
    else:
        anchors_nms_idx = nms_without_class(top_bbox, top_scores[:, 0], iou_threshold=iou_thresh)
    
    final_bbox = top_bbox[anchors_nms_idx]
    final_cls = classes_[anchors_nms_idx]

    # reduct to ori img and detach
    if torch.cuda.is_available():
        map2ori_bbox = (final_bbox / scale).cpu().detach().numpy()
        final_cls = final_cls.cpu().detach().numpy()
    else:
        map2ori_bbox = (final_bbox / scale).detach().numpy()
        final_cls = final_cls.detach().numpy()
    
    # imshow
    plt.imshow(img)
    for idx, rect in enumerate(map2ori_bbox):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        w = x2 - x1
        h = y2 - y1
        plt.gca().add_patch(patches.Rectangle((x1, y1), w, h, color=(1, 0, 0), fill=False))
        plt.text(x1, y1, classes[final_cls[idx]], color=(0, 1, 0))
    plt.savefig(save_path + 'demo.jpg', dpi=120)


if __name__ == '__main__':
    img_path = '/home/workspace/chencheng/libc++/efficientdet/datasets/char_recognize/mchar_val/000026.png'
    process_imgs(retinanet, img_path, classes)

