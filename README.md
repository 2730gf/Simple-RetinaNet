![image](https://github.com/chencheng1203/Simple-RetinaNet/blob/master/results/demo.png)

### simple implement of RetinaNet
The pytorch re-implement of [retinanet](https://arxiv.org/abs/1708.02002), this implement is purely for learing and referenced the elegant code of [zylo117
Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch). I add additional [CBAM](https://arxiv.org/abs/1807.06521v2) attention module for resnet. The det may not achieve the paper's pefemence, but can be a good demo for learning one-stage object detector.

### requirements
- pytorch >= 1.3 & torchvision
- pycocotools
- opencv-python


### note
For the convenience of debugging, argparse is not used, you can config you settings in config.py 

### training
Before training on yourself dataset, convert data to coco format first, and change the config in config.py

```
python train.py
```

### inference
This implement provide two ways to use the trained model, for single image test and web-camera, you alse can visualize the model output of class logits and bbox regression in the notebook <kbd>scrip/inference.ipynb</kbd>

```
python predict.py  # change test img path .py file
or
python web_camera.py  # change carema_ip
```
