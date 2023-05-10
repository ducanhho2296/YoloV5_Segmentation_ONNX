import cv2
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions import VideoCameraAPI
from realtime_segmentation import YOLOv5Segmentation

app = FastAPI()

model_path = "path-to-model"
yolov5_segmentation = YOLOv5Segmentation(model_path)


@app.get("/")
async def get(request: Request):
    return FileResponse("templates/index.html", media_type="text/html")

async def gen_frames():
    cam = VideoCameraAPI()
    cam.open(cameraID=0)
    cam.start()

    while True:
        frame = cam.read()
        im0 = yolov5_segmentation.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for frame in gen_frames():
        await websocket.send_bytes(frame)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
