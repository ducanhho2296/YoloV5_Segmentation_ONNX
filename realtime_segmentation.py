from functions import *

colors = Colors() #call colors for coloring masks

labelMap = [
"person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
"truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
"bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
"bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
"suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
"baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
"fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
"orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
"chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
"laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
"toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
"teddy bear",     "hair drier", "toothbrush"
]



#model directory
model_dir = "./yolov5"
model  = "path-to-model"

#load onnx model
cuda = True
providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
so = ort.SessionOptions()
so.log_severity_level = 3
print(ort.get_device())
session = ort.InferenceSession(model, providers=providers, sess_options=so)
input_name = session.get_inputs()[0].name
output_name1 = session.get_outputs()[0].name
output_name2 = session.get_outputs()[1].name

#calculate FPS
import time

#Using camera
cam = VideoCameraAPI()
cam.open(cameraID=0)
cam.start()

while True:
    # Capture frame-by-frame
    frame = cam.read()        
    time_start = time.time()

    #call class
    segmentation = Segmentator()
    img = segmentation.letterbox(frame)[0]  # padded resize
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img= np.ascontiguousarray(img)
    img = np.array(img)
    img = img.astype(np.float32) #convert img to float 32

    img /= 255
    if len(img.shape) == 3:
        img = img[None]

    pred_onnx = session.run([output_name1], {input_name: img})
    proto_onnx = session.run([output_name2], {input_name: img})
    pred = pred_onnx[0]
    proto_mask = proto_onnx[0]

    #calculate non_max_suppression for bbox
    pred = segmentation.non_max_suppression(pred)
    # pred = non_max_suppression_fast(pred, overlapThresh=0.5)
    #prediction
    im0 = frame

    for i, det in enumerate(pred):
        annotator = Annotator(im0, line_width=3)
        
        if len(det):
            masks = segmentation.process_mask(proto_mask[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
            det[:, :4] = segmentation.scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()


        # Mask plotting
            annotator.masks(masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=None)


        #write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                c = int(cls)  # integer class
                # label = f'{[c]} {conf:.2f}'
                cls_name = labelMap[c]
                label = f'{cls_name} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))


        im0 = annotator.result()
    time_stop = time.time()
    fps = 1 / (time_stop - time_start)
    print("fps: ", fps)
    cv2.imshow('frame', im0)

    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()   

