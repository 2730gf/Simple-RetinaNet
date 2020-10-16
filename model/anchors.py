import itertools
import torch
import torch.nn as nn
import numpy as np


class Anchor(nn.Module):
    def __init__(self, anchor_size=4, ratios=None, scales=None, pyramid_levels=None):
        super(Anchor, self).__init__()
        self.anchor_size = anchor_size

        if pyramid_levels == None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        
        if ratios == None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = ratios
    
        if scales == None:
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        else:
            self.scales = scales
        
        self.strides = [2 ** x for x in self.pyramid_levels]

        # if the next inputs with same shape with last image, just return last anchors
        self.last_anchors = {}   
        self.last_shape = None
    
    def forward(self, image):

        image_shape = image.shape[2:]
        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]
        
        if self.last_shape is None or image_shape != self.last_shape:
            self.last_shape = image_shape
        
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                base_anchor_size = self.anchor_size * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio / 2.0
                anchor_size_y_2 = base_anchor_size * (1.0 / ratio) / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # chages to [x1, y1, x2, y2]
                boxes = np.vstack((xv - anchor_size_x_2, yv - anchor_size_y_2, 
                                   xv + anchor_size_x_2, yv + anchor_size_y_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape N x A x 4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        # anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = np.concatenate(boxes_all, axis=0)
    
        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0).float()

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes


if __name__ == '__main__':
    Anchor = Anchor()
    image = torch.rand([2, 3, 800, 1024])
    anchors = Anchor(image)
    print(anchors.shape)





        

