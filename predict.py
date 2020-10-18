import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from model.retinanet import get_model
from script.utils import predict_single_imgs
from config import get_args


def process_imgs(model, img_path, cfg):
    img = cv2.imread(img_path)
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(cfg.num_classes)]
    bbox_results, classes_results, scores_results = predict_single_imgs(model, img_path, cfg)
    # imshow
    plt.imshow(img[..., ::-1])
    for idx, rect in enumerate(bbox_results):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        w = x2 - x1
        h = y2 - y1
        cls_id = classes_results[idx]
        plt.gca().add_patch(patches.Rectangle((x1, y1), w, h, color=colors[cls_id], fill=False))
        plt.text(x1, y1, cfg.classes[cls_id], bbox=dict(boxstyle='square,pad=0.2',fc=colors[cls_id], lw=0, alpha=0.5))
    # plt.savefig(save_path + 'demo.jpg', dpi=120)
    plt.show()


if __name__ == '__main__':
    cfg = get_args()
    retinanet = get_model(num_classes=cfg.num_classes, pretrained=False)
    state_dict = torch.load(cfg.load_from_path)
    
    if cfg.GPU_nums > 1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]   # remove `module.`
            new_state_dict[name] = v
        retinanet.load_state_dict(new_state_dict)
    else:
        retinanet.load_state_dict(state_dict)

    retinanet = retinanet.cuda()
    retinanet.eval()

    img_path = '/home/workspace/chencheng/Learning/ObjectDetection/Datasets/CoCodataset/train2017/000000580908.jpg'
    process_imgs(retinanet, img_path, cfg)

