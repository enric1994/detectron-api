# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, torchvision
from torchvision.utils import save_image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
from flask import Flask, request, send_file
import numpy as np
import cv2
import random
import skimage.io

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

torch.cuda.set_device(0)

cfg = get_cfg()
cfg.merge_from_file("/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
predictor = DefaultPredictor(cfg)

app = Flask(__name__)
app.debug = True

@app.route('/', methods=['POST'])
def process_files():

	received_images = []
	images = request.files.to_dict()
	# import pdb;pdb.set_trace()
	for image in images:
		file_name = images[image].filename
		npimg = np.fromfile(images[image], np.uint8)
		decoded_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
		# imageRGB = cv2.cvtColor(decoded_image , cv2.COLOR_BGR2RGB)
		received_images.append(decoded_image)

	# Do some operation using the received files and the parameters
	# imsave('/detectron-api/tmp/tmp_result.png', received_images[0])
	# image = skimage.io.imread('/detectron-api/data/frames/{}'.format(img_path))
	results = predictor(received_images[0])
	if len(results["instances"].pred_classes)>0:
			
		if int(results["instances"].pred_classes[0])==0:
			mask=results["instances"].pred_masks[0].float()
			save_image(mask, '/detectron-api/tmp/tmp_result.png')

			# Return one file
			resp = send_file('/detectron-api/tmp/tmp_result.png', mimetype='image/png')
			return resp
		else:
			return '404'
	else:
		return '404'

app.run(host="0.0.0.0", port=5000)