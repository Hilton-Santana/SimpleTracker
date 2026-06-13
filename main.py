'''
Copyright (c) 2023 Hilton-Santana <https://my.github.com/Hilton-Santana>

Created Date: Friday, June 12th 2023, 10:16:18 pm
Author: Hilton-Santana

Description:
HISTORY:
Date      	By	Comments
----------	---	----------------------------------------------------------
'''

from utils_tracker import *

from symbol import classdef
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import tensorflow as tf
import cv2 
from copy import deepcopy

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

# import yolov5
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from matplotlib import path
from matplotlib import pyplot as plt
from skimage import color as ski_c
from collections import namedtuple 

# define detectro predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE='cpu'
predictor = DefaultPredictor(cfg)

#yolov predictor
device = select_device('')
model = DetectMultiBackend(ROOT / 'yolov5s.pt', device=device, dnn=False, data=ROOT/ 'indoor.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # check image size


#-------------Initialization------------#
filename_dir = 'dir_quadro_'+str(0)+ '.jpg'
players_frame = []
im_dir = cv2.imread(filename_dir)
#initialize players colors
base_dict = GetBaseColor(filename_dir,predictor)
players = Process(predictor, im_dir,base_dict)
dets = det(players)
players_frame.append(players)
size_track = len(players)

# initialize court
ax, court_x_min, court_x_max, court_y_min, court_y_max = ShowCourt()

# Define homography matrix (obtained with a previous calibration step)
H = np.zeros((3,3))
H[0,:] = np.array([0.000302139956643,0.076194203611221,-34.398757666341062])
H[1,:] = np.array([0.020250767381669,0.015966246977228,-12.103605353296231])
H[2,:] = np.array([-0.000018109494451,0.001315802531690,0.443977294605643])

#----------Tracker-----------------#
vid_capture = cv2.VideoCapture('dir.mov')
n_frames = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
n = 5000 # number of frames 
#-----------------#------------------------------------#

step = 3
count = 1
n = 500
roi_up = 350
roi_down = 350
model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

for i in range(1,n):
  ret, im_dir = vid_capture.read()
  shape = im_dir.shape
  im_dir_mod = np.zeros((shape[0] - roi_up - roi_down,shape[1] ,shape[2]))
  im_dir_mod = im_dir[ roi_up : shape[0] - roi_down, : ,:]
  #im_dir = im_dir_mod
  if i % step == 0:
    #-------------yolo inference ------#
     with dt[0]:
      im = torch.from_numpy(im).to(model.device)
      im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
      im /= 255  # 0 - 255 to 0.0 - 1.0
      if len(im.shape) == 3:
          im = im[None]  # expand for batch dim

        # Inference
     with dt[1]:
      pred = model(im, augment=False, visualize=False)

        # NMS
     with dt[2]:
      pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

     filename = 'track'+str(count - 1)+ '.jpg'
     count = count + 1

     dets = Process(predictor, im_dir_mod,base_dict)  
     cv2.imshow("Show",im_dir_mod)
     cv2.waitKey(1) 
     if (IsValid(dets)):
      players = dets

     #debug
     #print(det(dets))
     #print('\n')
     #cv2.imshow("Show",im_dir)
     #cv2.waitKey(1) 
     #enddebug
     #predict
     trks = np.zeros((len(players), 5))
     ret = []
     for t, trk in enumerate(trks):
      bb = players[t].kalman_tracker.get_state()[0]
      pos = players[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      #print(pos)
      #cv2.rectangle(im_dir,(int(pos[0]),int(pos[1])),(int(pos[2]),int(pos[3])),(0,255,0),2)
     #cv2.imshow("Show",im_dir)
     #cv2.waitKey(1) 
     #new_players = deepcopy(players_frame[count - 1])
     matched, unmatched_dets, unmatched_trks = CalculateIOUMatrix(dets,trks)
     pose = [0,0]
     for m in matched:
      players[m[1]].update(dets[m[0]].getBoundingBox())
     hs = []
     for player in players:
      #print(pos)    
      color = player.GetColorG()
      p_2_ground = player.GetGroundPixel()
      p_pixel = np.array([p_2_ground[0],p_2_ground[1],1.0]) 
      p_homo = np.dot(H,p_pixel)
      pose[0] = p_homo[0]/p_homo[2]
      pose[1] = p_homo[1]/p_homo[2]
      pose_x = np.clip(pose[0],court_x_min,court_x_max)
      pose_y = np.clip(pose[1],court_y_min,court_y_max)
      h = ax.plot(pose_x,pose_y,'.',color = color , markersize='20' , alpha=0.9 )
      hs.append(h)
      pos = player.getBoundingBox()
      if count > 1:
        cv2.rectangle(im_dir,(int(pos[0]),int(pos[1])),(int(pos[2]),int(pos[3])),(0,255,0),2)
     if count > 1:
      cv2.imshow("Show",im_dir)
      cv2.waitKey(1) 

     plt.savefig(filename)
     for h in hs:
      p = h.pop(0)
      p.remove()

import imageio
import os

# Build GIF
with imageio.get_writer('mygif.gif', mode='I',fps=5) as writer:
  for i in range(0,count-1):
    filename = 'track'+str(i)+ '.jpg'
    image = imageio.imread(filename)
    writer.append_data(image)