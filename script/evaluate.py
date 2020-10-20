import os
import sys
sys.path.append('../')

import datetime
import json
import torch
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from model.retinanet import get_model
from config import get_args
from utils import predict_single_imgs, load_model
from dataset.CocoDataset import CocoDataset


def eval_coco(model, cfg):
    image_ids = []
    json_results = []

    val_coco = COCO(os.path.join(cfg.data_path, 'annotations', 'instances_val2017.json'))
    coco_ids = val_coco.getCatIds()
    coco_id_map = {}
    for i in range(cfg.num_classes):
        coco_id_map[i] = coco_ids[i]

    json_results = []
    for img_info in tqdm(list(val_coco.imgs.values())):
        img_name = img_info['file_name']
        img_id = img_info['id']
        img_path = os.path.join(cfg.data_path, 'val2017', img_name)
        rois, classes, scores = predict_single_imgs(model, img_path, cfg)

        for index, rect in enumerate(rois):
            x1 = rect[0]
            y1 = rect[1]
            x2 = rect[2]
            y2 = rect[3]
            width = x2 - x1
            height = y2 - y1
            class_id = coco_id_map[classes[index]]
            score = scores[index]
            image_result = {
                'image_id': img_id,
                'category_id': float(class_id),
                'score': float(score),
                'bbox': [float(x1), float(y1), float(width), float(height)]
            }
            json_results.append(image_result)
        image_ids.append(img_id)
    
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    save_path = os.path.join(cfg.eval_json_save_path, current_time + '-val2017.json')
    json.dump(json_results, open(save_path, 'w'), indent=4)

    val_set = CocoDataset(root_dir=cfg.data_path, set='val2017')
    coco_true = val_set.coco
    coco_pred = coco_true.loadRes(save_path)

    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    cfg = get_args()
    retinanet = get_model(num_classes=cfg.num_classes, pretrained=False)
    retinanet = load_model(retinanet, cfg)
    retinanet = retinanet.cuda()
    retinanet.eval()
    eval_coco(retinanet, cfg)


