import os
import sys
sys.path.append('/home/workspace/chencheng/Learning/ObjectDetection/retinanet.pytorch/model')

import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from backbone.resnet_cbam import resnet50_cbam, resnet101_cbam
from backbone.resnet import resnet50, resnet101
from fpn import FPN
from anchors import Anchor

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Classifier(nn.Module):
    """
    classification subnet
    share conv between Muti FPN level
    """
    def __init__(self, in_channels, num_anchors, num_classes, feature_size=256):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.head = nn.Conv2d(feature_size, num_classes * num_anchors, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        features = []
        for x in inputs:
            out = self.act1(self.conv1(x))
            out = self.act2(self.conv2(out))
            out = self.act3(self.conv3(out))
            out = self.act4(self.conv4(out))

            out = self.sigmoid(self.head(out))

            out = out.permute(0, 2, 3, 1)
            B, W, H, C = out.shape
            out = out.view(B, W, H, self.num_anchors, self.num_classes)
            out = out.contiguous().view(B, -1, self.num_classes)
            features.append(out)
        
        res = torch.cat(features, dim=1)
        return res


class Regressor(nn.Module):
    """
    regression subnet
    share conv between Muti FPN level
    """
    def __init__(self, in_channels, num_anchors, feature_size=256):
        super(Regressor, self).__init__()
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 =nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 =nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 =nn.ReLU()

        self.head = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
    
    def forward(self, inputs):
        features = []
        for x in inputs:
            out = self.act1(self.conv1(x))
            out = self.act2(self.conv2(out))
            out = self.act3(self.conv3(out))
            out = self.act4(self.conv4(out))
            out = self.head(out)

            out = out.permute(0, 2, 3, 1)
            res = out.contiguous().view(out.shape[0], -1, 4)

            features.append(res)

        res = torch.cat(features, dim=1)
        return res


class RetinaNet(nn.Module):
    def __init__(self, num_classes, fpn_features=256, ratios= None, scales=None, backbone='resnet50'):
        super(RetinaNet, self).__init__()

        self.anchor_ratios = [0.5, 1, 2] if ratios == None else ratios
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)] if scales == None else scales
        num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        self.num_classes = num_classes

        if backbone == 'resnet50':
            self.backbone = resnet50()
        if backbone == 'resnet101':
            self.backbone = resnet101()
        if backbone == 'resnet50_cbam':
            self.backbone = resnet50_cbam()
        if backbone == 'resnet101_cbam':
            self.backbone = resnet101_cbam()

        self.fpn = FPN(features=fpn_features)
        self.classifier = Classifier(in_channels=fpn_features,
                                     num_anchors=num_anchors, 
                                     num_classes=num_classes)
        self.regressor = Regressor(in_channels=fpn_features,
                                   num_anchors=num_anchors)
        self.anchors = Anchor()


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
    def forward(self, batch_img):
        C3, C4, C5 = self.backbone(batch_img)

        fpn_features = self.fpn([C3, C4, C5])

        classifications = self.classifier(fpn_features)
        regressions = self.regressor(fpn_features)

        anchors = self.anchors(batch_img)

        return fpn_features, regressions, classifications, anchors


def get_model(num_classes=80, backbone='resnet50', pretrained=True):
    model = RetinaNet(num_classes=num_classes, backbone=backbone)
    if pretrained:
        print('loading pretrained model...')
        model.load_state_dict(model_zoo.load_url(model_urls[backbone[:8]]), strict=False)
        print('pretrained loaded done.')
    return model


if __name__ == '__main__':
    image = torch.rand([2, 3, 768, 1024])
    retinanet = get_model(num_classes=80)
    out = retinanet(image)
    

        
        
