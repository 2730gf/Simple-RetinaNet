import os
import torch

import cv2
import numpy as np
from numpy import random
from torch import scalar_tensor
from torch.utils import data
from pycocotools.coco import COCO
from torchvision import transforms


##################
#   Define Coco
##################
class CocoDataset(data.Dataset):

    def __init__(self, root_dir, set='train2017', transform=None):
        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_'+self.set_name+'.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.ori_label_2_sort_label = {}
        for i in range(len(categories)):
            self.ori_label_2_sort_label[categories[i]['id']] = i + 1  # cause background id is 0

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)
        
        self.labels ={}
        for key, value in self.classes.items():
            self.labels[value] = key
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img = self.load_img(idx)
        anno = self.load_annos(idx)
        sample = {'img': img, 'anno': anno}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_img(self, idx):
        img_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path = os.path.join(self.root_dir, self.set_name, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.
    
    def load_annos(self, idx):
        anno_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx], iscrowd=False)
        annos = np.zeros((0, 5))

        if len(anno_ids) == 0: 
            return annos
        
        coco_annos = self.coco.loadAnns(anno_ids)
        for anno in coco_annos:
            if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                continue
            curr_anno = np.zeros((1, 5))
            curr_anno[0, :4] = anno['bbox']
            curr_anno[0, 4] = self.ori_label_2_sort_label[anno['category_id']] - 1
            annos = np.append(annos, curr_anno, axis=0)
        
        # from [x, y, w, h] to [x1, y1, x2, y2]
        annos[:, 2] = annos[:, 0] + annos[:, 2]
        annos[:, 3] = annos[:, 1] + annos[:, 3] 

        return annos


##################
#  for dataloader
##################
def collater(data):
    imgs = [s['img'] for s in data]
    annos = [s['anno'] for s in data]
    # scales = [s['scale'] for s in data]

    # imgs = torch.from_numpy(np.stack(imgs, axis=0))
    imgs = torch.stack(imgs, axis=0).type(torch.FloatTensor)
    max_num_annos = max(anno.shape[0] for anno in annos)

    if max_num_annos > 0:
        annoted_padded = torch.ones((len(annos), max_num_annos, 5)) * -1
        for idx, anno in enumerate(annos):
            annoted_padded[idx, :anno.shape[0], :] = anno
    else:
        annoted_padded = torch.ones((len(annos), 1, 5)) * -1

    return {'img': imgs, 'annot': annoted_padded}


##########################
#  utils for random crop
##########################
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


##############
## Transform
##############

# 1. 按长边 Resize
class Resize(object):

    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image, anno = sample['img'], sample['anno']
        height, width, _ = image.shape
        if height > width:
            scale = self.size / height
            resized_height = self.size
            resized_width = int(scale * width)
        else:
            scale = self.size / width
            resized_height = int(scale * height)
            resized_width = self.size
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_img = np.zeros((self.size, self.size, 3))
        new_img[0:resized_height, 0:resized_width, :] = image
        anno[:, :4] *= scale
        # sample = {'img': torch.from_numpy(new_img).to(torch.float32), 'anno': torch.from_numpy(anno), 'scale': scale}
        sample = {'img': new_img, 'anno': anno}
        return sample

class Resize_wh(object):

    def __init__(self, size=(1024, 800)):
        self.size = size
        self.w = self.size[0]
        self.h = self.size[1]
    
    def __call__(self, sample):
        image, anno = sample['img'], sample['anno']
        height, width, _ = image.shape
        # keep aspect
        scale = self.h / height
        if (scale * width > self.w):
            scale = self.w / width
        new_h, new_w = int(scale * height), int(scale * width)
        img_resized = cv2.resize(image, (new_w, new_h))
        img_paded = np.zeros(shape=[self.h, self.w, 3])
        img_paded[:new_h, :new_w, :] = img_resized

        if len(anno) == 0:
            sample = {'img': img_paded, 'anno': anno}
        else:
            anno[:, :4] *= scale
            sample = {'img': img_paded, 'anno': anno}

        return sample




# 2. 随机翻转
class RandomFlip(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            img, anno = sample['img'], sample['anno']
            img = img[:, ::-1, :]
            _, cols, _ = img.shape
            x1 = anno[:, 0].copy()
            x2 = anno[:, 2].copy()
            x_tmp = x1.copy()
            anno[:, 0] = cols - x2
            anno[:, 2] = cols - x_tmp
            sample = {'img': img, 'anno': anno}
        return sample

# 3. 随机光学变换
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, sample):
        if random.randint(2):
            img, anno = sample['img'], sample['anno']
            delta = random.uniform(-self.delta, self.delta)
            img += delta
            sample = {'img': img, 'anno': anno}
        return sample

# 4. 归一化
class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['anno']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'anno': annots}


# 5. 转换成Tensor
class ToTensor(object):
    def __call__(self, sample):
        img = sample['img']
        anno = sample['anno']
        tensor_img = torch.from_numpy(img).permute(2, 0, 1)
        tensor_anno = torch.from_numpy(anno)
        return {'img': tensor_img, 'anno': tensor_anno}

# 6. 随机裁剪
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, sample):
        image = sample['img']
        annos = sample['anno']
        boxes = annos[:, :4].copy()
        labels = annos[:, 4].copy()

        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return sample

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            # max trails (50)
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)
                # is min and max overlap constraint satisfied? if not try again
                if len(overlap) == 0 or overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()
                # take only matching gt labels
                current_labels = labels[mask]
                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                new_anno = np.zeros((current_boxes.shape[0], 5))
                new_anno[:, :4] = current_boxes
                new_anno[:, 4] = current_labels
                sample = {'img': current_image, 'anno': new_anno}
                return sample

if __name__ == '__main__':
    char_data_root = '/home/workspace/chencheng/libc++/lib-mchar/datasets/char_recognize/'
    coco_data_root = '/home/workspace/chencheng/Learning/ObjectDetection/Datasets/CoCodataset/'
    training_set = CocoDataset(root_dir=coco_data_root,set='val2017',
                            transform=transforms.Compose([Resize_wh()]))
