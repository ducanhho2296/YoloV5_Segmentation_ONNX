import cv2
import os, sys
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions import VideoCameraAPI
from realtime_segmentation import YOLOv5Segmentation

app = FastAPI()

model_path = "path-to-model"
yolov5_segmentation = YOLOv5Segmentation(model_path)

def gen_frames():
    cam = VideoCameraAPI()
    cam.open(cameraID=0)
    cam.start()

    while True:
        frame = cam.read()
        im0 = yolov5_segmentation.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/")
def index(request: Request):
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
