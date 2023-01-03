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

    def nms(self, dets, scores, thresh):
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



    def non_max_suppression(self, prediction, conf_thres=0.5, iou_thres=0.45, max_det=1000):
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
            box = self.xywh2xyxy(x[:, :4])  #Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] (line 912/general.py)
            mask = x[:, mi:]
            # Detections matrix nx6 (xyxy, conf, cls)
            conf = np.amax(x[:, 5:mi], axis=1, keepdims=True)
            j = np.argmax(x[:, 5:mi], axis=1).reshape(conf.shape)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[conf.flatten() > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

            i = self.nms(boxes, scores, iou_thres)  # NMS
            # if i.shape[0] > max_det:  # limit detections
            #     i = i[:max_det]
            output[xi] = x[i]

        return output

    #Function to calculate masks
    def crop_mask(self, masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape #n = 6 (yolov5s-seg.onnx)
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
        """

        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        # masks = (masks_in @ protos.astype(float).view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
        mask_protos = np.reshape(protos, (c, -1))
        matmulres = np.matmul(masks_in, mask_protos)
        masks = np.reshape(matmulres, (masks_in.shape[0], mh, mw))

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        # if upsample:
        #     masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        masks_gt = np.greater(masks, 0.5)
        masks_gt = masks_gt.astype(float)
        # return masks.gt_(0.5)
        return masks_gt

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        # if isinstance(boxes, torch.Tensor):  # faster individually
        #     boxes[:, 0].clamp_(0, shape[1])  # x1
        #     boxes[:, 1].clamp_(0, shape[0])  # y1
        #     boxes[:, 2].clamp_(0, shape[1])  # x2
        #     boxes[:, 3].clamp_(0, shape[0])  # y2
        # else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def is_ascii(self, s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
        s = str(s)  # convert list, tuple, None, etc. to str
        return len(s.encode().decode('ascii', 'ignore')) == len(s)

    def scale_image(self, im1_shape, masks, im0_shape, ratio_pad=None):
        """
        img1_shape: model input shape, [h, w]
        img0_shape: origin pic shape, [h, w, 3]
        masks: [h, w, num] -> in onnx numpy: [n, w, h] ##(6, 160, 160)
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
