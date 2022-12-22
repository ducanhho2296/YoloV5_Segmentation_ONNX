import cv2
import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageDraw

class Segmentator:
    
    def __init__(self, conf_thresh=0.25, iou_thresh=0.5, max_det=300):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.inference_time = None
        self.nms_time = None
        self.interpreter = None
        self.is_inititated = False
        

    def xywh2xyxy(self, x):    
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

