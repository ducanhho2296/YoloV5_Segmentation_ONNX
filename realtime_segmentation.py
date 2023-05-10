import numpy as np
import cv2
import onnxruntime as ort
from functions import *


class YOLOv5Segmentation:
    def __init__(self, model_path):
        self.labelMap = [...]
        self.model = model_path
        self.load_model()
        self.colors = Colors()

    def load_model(self):
        cuda = True
        providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(self.model, providers=providers, sess_options=so)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name1 = self.session.get_outputs()[0].name
        self.output_name2 = self.session.get_outputs()[1].name

    def process_frame(self, frame):
        segmentation = Segmentator()
        img = segmentation.letterbox(frame)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = np.array(img)
        img = img.astype(np.float32)
        img /= 255

        if len(img.shape) == 3:
            img = img[None]

        pred_onnx = self.session.run([self.output_name1], {self.input_name: img})
        proto_onnx = self.session.run([self.output_name2], {self.input_name: img})
        pred = pred_onnx[0]
        proto_mask = proto_onnx[0]

        pred = segmentation.non_max_suppression(pred)
        im0 = frame

        for i, det in enumerate(pred):
            annotator = Annotator(im0, line_width=3)

            if len(det):
                masks = segmentation.process_mask(proto_mask[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                det[:, :4] = segmentation.scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                annotator.masks(masks, colors=[self.colors(x, True) for x in det[:, 5]], im_gpu=None)

                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)
                    cls_name = self.labelMap[c]
                    label = f'{cls_name} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=self.colors(c, True))

            im0 = annotator.result()

        return im0

if __name__ == "__main__":
    model_path = "path-to-model"
    yolov5_segmentation = YOLOv5Segmentation(model_path)

    cam = VideoCameraAPI()
    cam.open(cameraID=0)
    cam.start()

    while True:
        frame = cam.read()
        im0 = yolov5_segmentation.process_frame(frame)
        cv2.imshow('frame', im0)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
