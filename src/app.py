import torch, torchvision
from torchvision.utils import save_image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
# import skimage
import skimage.io


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

torch.cuda.set_device(0)

im = cv2.imread("../data/frames/out1080.mp4-00000.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) if you're not running a model in detectron2's core library
cfg.merge_from_file("/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
# https://dl.fbaipublicfiles.com/detectron2/TensorMask/tensormask_R_50_FPN_1x/152549419/model_final_8f325c.pkl
predictor = DefaultPredictor(cfg)



# outputs = predictor(im)

# save_image(mask.float(), 'mask.png')

# import pdb;pdb.set_trace()
# ###

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite('out3.png', v.get_image()[:, :, ::-1])


with open('/detectron-api/data/frames.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    previous_img=None 
    for img_path in content:
        print(img_path)
        image = skimage.io.imread('/detectron-api/data/frames/{}'.format(img_path))

        # Run detection
        results = predictor(image)
        if len(results["instances"].pred_classes)>0:
            
            if int(results["instances"].pred_classes[0])==0:
                mask=results["instances"].pred_masks[0].float()

                save_image(mask, '/detectron-api/data/masks/{}'.format(img_path))

                previous_img = mask
            else:
                print('no human detected!')
                cv2.imwrite('masks/{}'.format(img_path),previous_img)
        else:
            print('no object detected!')
            cv2.imwrite('masks/{}'.format(img_path),previous_img)