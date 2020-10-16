import cv2
import time
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from model.retinanet import get_model
from model.utils import BBoxTransform, ClipBoxes, nms_with_class, nms_without_class

from config import get_args
cfg = get_args()

retinanet = get_model(num_classes=cfg.num_classes, pretrained=False)
state_dict = torch.load(cfg.load_from_path)

# cause we trained model on multi GPU, remove 'module'
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v

retinanet.load_state_dict(new_state_dict)
retinanet = retinanet.cuda()
retinanet.eval()

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# visulize config
color_map = [(int(np.random.rand()*255), int(np.random.rand()*255), int(np.random.rand()*255)) for _ in range(cfg.num_classes)] 

def web_camera_inference(model, camera_ip, cfg):
    """
    web camera inference
    """
    pre_time = time.time()
    cap = cv2.VideoCapture(camera_ip)
    while(cap.isOpened()):

        time_elapse = time.time() - pre_time
        pre_time = time.time()
        fps = round(1. / time_elapse)

        ret, img = cap.read()
        img = img[..., ::-1]

        normalized_img = (img / 255 - cfg.mean) / cfg.std
        ori_h, ori_w = normalized_img.shape[:2]

        scale = cfg.inference_resize_wh[1] / ori_h
        if scale * ori_w > cfg.inference_resize_wh[0]:
            scale = cfg.inference_resize_wh[0] / ori_w
        new_h, new_w = int(scale * ori_h), int(scale * ori_w)
        img_resized = cv2.resize(normalized_img, (new_w, new_h))

        img_paded = np.zeros([cfg.inference_resize_wh[1], cfg.inference_resize_wh[0], 3])
        img_paded[:new_h, :new_w, :] = img_resized

        img_input = torch.from_numpy(img_paded).permute(2, 0, 1).unsqueeze(dim=0).float()

        if torch.cuda.is_available():
            img_input = img_input.cuda()

        _, regression, classification, anchors = model(img_input)

        # add predict offset
        trans_anchors = regressBoxes(anchors, regression)
        trans_anchors = clipBoxes(trans_anchors, img_input)

        scores = torch.max(classification, dim=2, keepdim=True)[0]  # 计算anchor的最高得分
        scores_over_thresh = (scores > cfg.score_thresh)[:, :, 0]
        
        if scores_over_thresh.sum() < 1:
            print("No roi find!")
            continue

        top_cls = classification[0, scores_over_thresh[0, :], ...].permute(1, 0)  # get classes over thresh
        top_bbox = trans_anchors[0, scores_over_thresh[0, :], ...]  # get bbox over thresh
        top_scores = scores[0, scores_over_thresh[0, :], ...]  # get scores
        scores_, classes_ = top_cls.max(dim=0)  # score and label_id for every bbox

        if cfg.nms_with_cls:
            anchors_nms_idx = nms_with_class(top_bbox, top_scores[:, 0], classes_, iou_threshold=cfg.nms_thresh)
        else:
            anchors_nms_idx = nms_without_class(top_bbox, top_scores[:, 0], iou_threshold=cfg.nms_thresh)
        
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
        
        for idx, rect in enumerate(map2ori_bbox):
            x1 = rect[0]
            y1 = rect[1]
            x2 = rect[2]
            y2 = rect[3]
            color = color_map[final_cls[idx]]
            
            cv2.rectangle(img[..., ::-1], (x1, y1), (x2, y2), color, 1)
            cv2.putText(img[..., ::-1], cfg.classes[final_cls[idx]], \
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, \
                        0.75, color, 2)
            cv2.putText(img[..., ::-1], "fps: " + str(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        
            cv2.imshow("camera_detection", img[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    carame_ip = 'http://admin:admin@10.28.242.238:8081/'
    # carame_ip = 0
    web_camera_inference(retinanet, carame_ip, cfg)

