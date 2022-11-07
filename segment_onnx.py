from dataclasses import dataclass
from importlib.resources import path
import json
from unittest import result
import numpy as np
import onnx
import onnxruntime
import cv2
from onnx import numpy_helper
import sys
from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
import glob


#non max suppression 
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
def xywh2xyxy(x):    
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        other_box_ids = order[1:]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[other_box_ids])
        yy1 = np.maximum(y1[i], y1[other_box_ids])
        xx2 = np.minimum(x2[i], x2[other_box_ids])
        yy2 = np.minimum(y2[i], y2[other_box_ids])

        # print(list(zip(xx1, yy1, xx2, yy2)))

        w = np.maximum(0.0, xx2 - xx1 + 1e-9)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1e-9)  # maxiumum height
        inter = w * h

        ovr = inter / (areas[i] + areas[other_box_ids] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)



def non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.45, max_det=1000):
    output = [np.zeros((0, 6))] * prediction.shape[0]
    if prediction.size == 0:
        return output

    xc = prediction[..., 4] > conf_thres  # candidate
    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    nc = 80
    mi = 5 + nc

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])  #Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] (line 912/general.py)
        mask = x[:, mi:]
        # Detections matrix nx6 (xyxy, conf, cls)
        conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:mi], axis=1).reshape(conf.shape)
        x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.flatten() > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
