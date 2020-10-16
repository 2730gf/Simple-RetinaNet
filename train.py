import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import backends
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict

from model.retinanet import get_model
from model.loss import FocalLoss
from dataset.CocoDataset import CocoDataset, Resize_wh, RandomFlip, Normalizer, ToTensor, collater

from config import get_args

import datetime
import logging

backends.cudnn.fastest = True
backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

def train(cfg):
    
    current_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    log_save_path = os.path.join(cfg.log_path, cfg.project_name)

    # create new folder or file
    os.makedirs(os.path.join(cfg.save_path, cfg.project_name), exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    f = open(log_save_path+'/{}-train.log'.format(current_time), mode='w', encoding='utf-8')
    f.close()

    # logging config
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO,
                        filename=log_save_path+'/{}-train.log'.format(current_time))

    # training datasets
    training_set = CocoDataset(root_dir=cfg.data_path, set=cfg.train_set,
                               transform=transforms.Compose([Normalizer(),
                                                            RandomFlip(),
                                                            Resize_wh(size=cfg.resize_wh),
                                                            ToTensor()]))
    training_generator = DataLoader(training_set, 
                                    batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers,
                                    collate_fn=collater)
    
    val_set = CocoDataset(root_dir=cfg.data_path, set=cfg.val_set,
                          transform=transforms.Compose([Normalizer(),
                                                        Resize_wh(size=cfg.resize_wh),
                                                        ToTensor()]))
    val_generator = DataLoader(val_set, 
                               batch_size=cfg.batch_size,
                               num_workers=cfg.num_workers,
                               collate_fn=collater)

    # get model and assign to devices
    model = get_model(num_classes=cfg.num_classes, backbone='resnet50')

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if cfg.GPU_nums > 1 and torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device)
    else:
        if cfg.GPU_nums == 1 and torch.cuda.is_available():
            model = model.cuda()


    # load model dict
    if cfg.load_from:
        state_dict = torch.load(cfg.load_from_path)
        model.load_state_dict(state_dict)

    
    # optimizer config
    if cfg.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    
    # loss function config
    criterion = FocalLoss()
    
    # lr config
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.lr_patience, verbose=True)

    model.train()
    iters_all = len(training_generator)
    for epoch in range(cfg.start_epoch, cfg.epoches+1):
        epoch_loss = []
        for iter, data in enumerate(training_generator):
            imgs = data['img']
            annot = data['annot']
            # print(annot)
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                annot = annot.cuda()
            optimizer.zero_grad()

            _, regressions, classifications, anchors = model(imgs)

            cls_loss, reg_loss = criterion(classifications, regressions, anchors, annot)

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()

            loss = cls_loss + reg_loss
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            message = 'epoch: {}/{}, iters: {}/{}, Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}.'\
                    .format(epoch, cfg.epoches, iter+1, iters_all, cls_loss.item(), reg_loss.item(), loss.item())

            logging.info(message)
            print(message)
        
        scheduler.step(np.array(epoch_loss).mean())
        # model saving
        model_saving_path = os.path.join(cfg.save_path, cfg.project_name, 'epoch_{}_loss_{:.5f}.pth'.format(epoch, np.array(epoch_loss).mean()))
        torch.save(model.state_dict(), model_saving_path)


if __name__ == '__main__':
    cfg = get_args()
    train(cfg)







