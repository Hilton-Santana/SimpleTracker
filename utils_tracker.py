from matplotlib import path
from matplotlib import pyplot as plt
from skimage import color as ski_c
from collections import namedtuple 
import cv2 
import numpy as np
import pandas as pd

from kalman_tracker import *

BndBox = namedtuple('BB', ['x', 'y'])

def insideCourt(box):
  a = (box[0],box[1])
  b = (box[2],box[3])
  if (~isInside(a) and ~isInside(b)):
    return False
  return True
def isInside(pt):
  A0 = (242,455)
  S0 = (1629,450)
  A8 = (24,727)
  S8 = (1894,708)
  p = path.Path([A0,S0,S8,A8])
  res = p.contains_points([pt])
  return res
class Player:
  def __init__(self,bb,im,mask,base):
    self.BB = BndBox([bb[0],bb[2]], [bb[1],bb[3]])
    self.color = [0,255,0]
    self.mean_color = []
    self.confiance = []
    self.BB_twin = [] #twin bb of another camera 
    self.matchColor(im,mask,base)
    self.kalman_tracker = KalmanBoxTracker([bb[0],bb[1],bb[2],bb[3]])
  def getBoundingBox(self):
    return np.array([self.BB.x[0],self.BB.y[0],self.BB.x[1],self.BB.y[1]])
  def getArea(self):
    area = (self.BB.x[1] - self.BB.x[0]) * (self.BB.y[1] - self.BB.y[0])
    return area
  def getWidth(self):
    w = self.BB.x[1] - self.BB.x[0]
    return w	
  def getHeight(self):
    h = self.BB.y[1] - self.BB.y[0]
    return h
  def matchColor(self, im,mask,base):
    color = []
    mean_color = cv2.mean(im, mask=mask)[:3]
    lab = np.zeros((1, 1, 3), dtype="uint8")
    lab[0,:,:] = mean_color
    mean_color_lab = ski_c.rgb2lab(lab).flatten()
    minDist = np.inf
    for key, value in base.items():
      base_color = value
      diff = (mean_color_lab - base_color)
      err = np.dot(diff,diff)
      if err < minDist:
        minDist = err
        color = key
    if color == "blue":
      self.color = [255,0,0]
    elif color == "black":
      self.color = [0,0,0]
    else:
      self.color = [255,255,255]
    self.confiance = np.sqrt(minDist)
    self.mean_color = mean_color_lab
  def Show(self):
    print(self.mean_color)
    print(self.confiance)
    print(self.BB)
    print(self.BB_twin)
  def GetBB(self):
    return self.BB
  def GetColor(self):
    return self.color
  def GetColorG(self):
    if self.color == [255,0,0]:
      return [0,0,1]
    elif self.color == [0,0,0]:
      return [0.5,0,0]
    else:
      return [0.6,0.6,0.4]
  def IsOverlapping(self,BB):
    b_x = BB.x
    b_y = BB.y
    if b_x[0] > self.BB.x[1] or b_x[1] < self.BB.x[0]:
      return False
    elif b_y[0] > self.BB.y[1] or b_y[1] < self.BB.y[0]:
      return False
    else:
      #print(BB)
      return True
  def GetGroundPixel(self):
    p_x = int((self.BB.x[0] + self.BB.x[1]) * 0.5)
    return [p_x, self.BB.y[1]]
  def GetMiddlePixel(self,flag):
    if flag == 1:
      p_x = ((self.BB.x[0] + self.BB.x[1]) * 0.5)
      p_y = ((self.BB.y[0] + self.BB.y[1]) * 0.5)
      return [p_x, p_y]
    elif flag == 2:
      p_x = ((self.BB_twin.x[0] + self.BB_twin.x[1]) * 0.5)
      p_y = ((self.BB_twin.y[0] + self.BB_twin.y[1]) * 0.5)
      return [p_x, p_y]
  def OverlappingArea(self,BB):
    x1 = max(self.BB.x[0],BB.x[0])
    x2 = min(self.BB.x[1],BB.x[1])
    y1 = max(self.BB.y[0],BB.y[0])
    y2 = min(self.BB.y[1],BB.y[1])
    area = max(0,(x2 - x1)) * max(0,(y2 - y1))
    return area
    if not(self.IsOverlapping(BB)):
      return 0.0
    base = 0.0
    height = 0.0
    #base
    if BB.x[0] > self.BB.x[0]:
        if BB.x[1] < self.BB.x[0]:
         base = BB.x[1] - BB.x[0]
        else:
         base = self.BB.x[1] - BB.x[0]
    else:
        if self.BB.x[1] < BB.x[1]:
          base = self.BB.x[1] - self.BB.x[0]
        else:
          base = BB.x[1] - self.BB.x[0]
    #height
    if BB.y[0] > self.BB.y[0]:
        if BB.y[1] < self.BB.y[0]:
         height = BB.y[1] - BB.y[0]
        else:
         height = self.BB.y[1] - BB.y[0]
    else:
        if self.BB.y[1] < BB.y[1]:
          height = self.BB.y[1] - self.BB.y[0]
        else:
          height = BB.y[1] - self.BB.y[0]
    area = base * height
    return area
  def IOU(self,player):
    BB = player.BB
    area_1 = self.getArea()
    area_2 = player.getArea()
    intersection_area = self.OverlappingArea(BB)
    iou = intersection_area / (area_1 + area_2 - intersection_area)
    return iou
  def MatchOtherBB(self,players):
    max_area = -1
    for player in players:
      color = player.GetColor()
      if self.color != color:
        continue
      BB = player.GetBB()
      if self.IsOverlapping(BB):
        #print(BB)
        area = self.OverlappingArea(BB)
        if area > max_area:
          max_area = area
          self.BB_twin = BB
    if max_area == -1:
        return False
    return True
  def update(self,z):
    self.kalman_tracker.update(z)
    bb = self.kalman_tracker.get_state()[0]
    self.BB = BndBox([bb[0],bb[2]], [bb[1],bb[3]])
    return (bb)
  def predict(self):
    bb =  (self.kalman_tracker.predict())
    bb_i = bb[0]
    self.BB = BndBox([bb_i[0],bb_i[2]], [bb_i[1],bb_i[3]])
    return (bb)

def Process(predictor,im,base_dict):
  players = []
  outputs = predictor(im)
  b = outputs["instances"].pred_boxes
  masks = outputs["instances"].pred_masks
  c = outputs["instances"].pred_classes
  count = 0
  for (box,mask,cl) in zip(b,masks,c):
    if cl != 0:
      continue
    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), [255,0,0], 2)
    if insideCourt(box):
      count = count + 1
      mask = mask.cpu().data.numpy().astype(np.uint8)
      mask = cv2.erode(mask, None, iterations=2)
      player = Player(box,im,mask,base_dict)
      players.append(player)
      cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), player.color, 2)
  return players
def Overlap(players,im):
	for player in players:
		box = player.GetBB()
		cv2.rectangle(im, (box.x[0], box.y[0]), (box.x[1], box.y[1]), player.color, 2)
def GetBaseColor(filename,predictor):
	im = cv2.imread(filename)
	chans = cv2.split(im)
	colors = ("b","g","r")
	outputs = predictor(im)
	b = outputs["instances"].pred_boxes
	masks = outputs["instances"].pred_masks
	base_ids = [1,16,17] #juiz, black_team, white_team
	keys = ["blue", "black", "white" ]
	lab = np.zeros((len(colors), 1, 3), dtype="uint8")
	base_dict = {}
	for i in range(3):
		box = b[base_ids[i]].tensor
		box = [int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])]
		mask = masks[base_ids[i]].cpu().data.numpy().astype(np.uint8)
		mask = cv2.erode(mask, None, iterations=2) 
		#cv2.imshow('mask',im[:,:,1]*mask)
		mean_color = cv2.mean(im, mask=mask)[:3]
		lab[i,:,:] = mean_color
		#show histograms
		fig = plt.figure()
		for (chan, color) in zip (chans,colors):
			hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])        
			#plt.plot(hist, color = color)
			#plt.xlim([0, 256])
	#print(lab)
	lab = ski_c.rgb2lab(lab)
	for i in range(3):
		base_dict[keys[i]] = lab[i,:,:].flatten()
	return base_dict


"""# Create Color base for players"""
def det(players):
  bb_ti = np.zeros((len(players),10))
  for i in range(0,len(players)):
    bb_ti[i,2] = players[i].BB.x[0]
    bb_ti[i,3] = players[i].BB.y[0]
    bb_ti[i,4] = players[i].BB.x[1] - players[i].BB.x[0]
    bb_ti[i,5] = players[i].BB.y[1] - players[i].BB.y[0]
  return bb_ti


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def	CalculateIOUMatrix(players_ti,bb_tj,iou_threshold = 0.3):
  bb_ti = np.zeros((len(players_ti),4))
  for i in range(0,len(players_ti)):
    bb_ti[i,0] = players_ti[i].BB.x[0]
    bb_ti[i,1] = players_ti[i].BB.y[0]
    bb_ti[i,2] = players_ti[i].BB.x[1]
    bb_ti[i,3] = players_ti[i].BB.y[1]
  trackers = bb_tj
  detections = bb_ti
  bb_ti = np.expand_dims(bb_ti, 1) # detection
  bb_tj = np.expand_dims(bb_tj, 0) # tracker
  xx1 = np.maximum(bb_ti[..., 0], bb_tj[..., 0])
  yy1 = np.maximum(bb_ti[..., 1], bb_tj[..., 1])
  xx2 = np.minimum(bb_ti[..., 2], bb_tj[..., 2])
  yy2 = np.minimum(bb_ti[..., 3], bb_tj[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  iou_matrix = wh / ((bb_ti[..., 2] - bb_ti[..., 0]) * (bb_ti[..., 3] - bb_ti[..., 1])                                      
    + (bb_tj[..., 2] - bb_tj[..., 0]) * (bb_tj[..., 3] - bb_tj[..., 1]) - wh)    

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))
  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)
#filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]< iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
      matches = np.empty((0,2),dtype=int)
  else:
      matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def IsValid(dets):
  size = len(dets)
  count_blue = 0
  count_white = 0
  count_black = 0
  for det in dets:
    color = det.GetColor()
    if color == [255,0,0]:
      count_blue += 1
    if color == [0,0,0]:
      count_white += 1
    if color == [255,255,255]:
      count_black += 1
  
  flag = (size == 12) and  (count_blue == 2) and (count_white == 5) and (count_black == 5)
  return flag

def ShowCourt():
    frame = pd.read_excel('data/court_geometry.xlsx', header=0)

    court_x_min = 0.0
    court_x_max = 15.0
    court_y_min = 0.0
    court_y_max = 28.0

    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([court_x_min, court_x_max])
    ax.set_ylim([court_y_min, court_y_max])

    def pt(name):
        row = frame.loc[frame['Pts'] == name, ['Xg', 'Yg']].astype(np.float64)
        if row.empty:
            raise ValueError(f"Point {name} not found in court_geometry.xlsx")
        return row.iloc[0].to_numpy()

    def line(p1, p2, linewidth=1.5):
        a = pt(p1)
        b = pt(p2)
        ax.plot([a[0], b[0]], [a[1], b[1]], color='black', linewidth=linewidth)

    # Four outer lines
    line("A0", "A8", linewidth=2.5)
    line("A8", "S8", linewidth=2.5)
    line("S8", "S0", linewidth=2.5)
    line("S0", "A0", linewidth=2.5)

    # Garrafão esquerdo
    line("A2", "F2")
    line("F2", "F6")
    line("F6", "A6")
    line("A6", "A2")

    # Garrafão direito
    line("N2", "S2")
    line("S2", "S6")
    line("S6", "N6")
    line("N6", "N2")

    # Circle
    j4 = pt("J4")
    k4 = pt("K4")

    R = np.linalg.norm(k4 - j4)
    theta = np.linspace(0.0, 2 * np.pi, num=30)

    circle_x = j4[0] + R * np.sin(theta)
    circle_y = j4[1] + R * np.cos(theta)

    ax.plot(circle_x, circle_y, color='black', linewidth=1.5)

    #debug
    #plt.show()
    #enddebug
    return ax, court_x_min, court_x_max, court_y_min, court_y_max