class get_args:
    def __init__(self):
        # model config
        self.anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.anchors_ratios = [0.5, 1, 2]

        # training config
        self.optim = 'adam'
        self.lr = 1e-5
        self.epoches = 100
        self.lr_patience = 2  # if get patience time continuous bad mean loss, then change the lr
        self.val_interval = 4
        self.start_epoch = 41
        self.GPU_nums = 2

        # dataset config
        self.num_classes = 80
        self.resize_wh = (1024, 768)
        self.train_set = 'train2017'
        self.val_set = 'val2017'
        self.num_workers = 12
        self.batch_size = 9
        # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # path config
        self.project_name = 'coco'
        self.data_path = '/home/workspace/chencheng/Learning/ObjectDetection/Datasets/CoCodataset/'
        self.log_path = 'logs/'
        self.load_weights = None
        self.save_path = 'checkpoints/'
        self.load_from = True
        self.load_from_path = '/home/workspace/chencheng/Learning/ObjectDetection/retinanet.pytorch/checkpoints/coco/epoch_40_loss_0.52891.pth'

        # inference config
        self.use_soft_nms = False
        self.nms_thresh = 0.3
        self.score_thresh = 0.25
        self.results_save_path = 'results/'
        self.inference_resize_wh = (1024, 768)
        self.nms_with_cls = False

        # evaluate config
        self.eval_json_save_path = '/home/workspace/chencheng/Learning/ObjectDetection/retinanet.pytorch/results/eval'

        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
                        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', \
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', \
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', \
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', \
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', \
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', \
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', \
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', \
                        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', \
                        'toothbrush']