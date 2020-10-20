import cv2
import numpy as np

import torch
import torch.nn as nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from model.retinanet import get_model
from script.utils import predict_single_imgs, load_model
from config import get_args

def process_imgs(model, img_path, cfg):
    img = cv2.imread(img_path)
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
        plt.gca().add_patch(patches.Rectangle((x1, y1), w, h, color=cfg.color_map[cls_id], fill=False))
        plt.text(x1, y1, cfg.classes[cls_id], bbox=dict(boxstyle='square,pad=0.2',fc=cfg.color_map[cls_id], lw=0, alpha=0.5))
    plt.savefig(cfg.results_save_path + img_path.split('/')[-1], dpi=120)
    plt.show()


if __name__ == '__main__':
    cfg = get_args()
    retinanet = get_model(num_classes=cfg.num_classes, pretrained=False)
    retinanet = load_model(retinanet, cfg)
    retinanet = retinanet.cuda()
    retinanet.eval()
    img_path = 'results/test_img/000000429109.jpg'
    process_imgs(retinanet, img_path, cfg)

