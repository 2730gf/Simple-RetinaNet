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
We provide two ways to use the trained model, for single image test and web-camera, you alse can visualize the model output of class logits and bbox regression in the notebook <kbd>scrip/inference.ipynb</kbd>

```
python predict.py  # change test img path first
or
python web_camera.py  # change carema_ip
```

### results
we train this model from scratch with backbone resnet50, but can not reach paper's results, here is the metric we got.
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.310
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.157
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.481
```
if you want the state_dict to learn how the model work, click this [link](https://pan.baidu.com/s/1cTQO1GokShrQClHAMEQmKg)(passwd:bupt)
