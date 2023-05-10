# Real-time Instance Segmentation using YOLOv5 and ONNX Runtime

This is a approach for real-time instance segmentation using YOLOv5 and ONNX Runtime. The project uses YOLOv5 to detect objects in the input video stream and then performs instance segmentation to create a binary mask for each detected object. The resulting masks are then overlaid on the original video frames to highlight the detected objects.

## Requirements

To run this project, you need the following libraries installed:

- PyTorch
- OpenCV
- NumPy
- ONNX Runtime

The following command:

```bash
pip install torch opencv-python numpy onnxruntime
```

## Usage

1. Clone this repository

2. Download the YOLOv5 segmentation model

This project uses a custom dataset trained on YOLOv5 from Ultralytics. You can download the YOLOv5 model checkpoints from the [official repository](https://github.com/ultralytics/yolov5#pretrained-checkpoints).

3. Update the `model` variable in the `realtime_segmentation.py` file with the path to your downloaded YOLOv5 model.

```python
model = "path/to/yolov5.pt"
```

4. Run the `realtime_segmentation.py` file.

```bash
python realtime_segmentation.py
```

5. The program will open the default camera on your computer and start detecting and segmenting objects in real-time. You can press the 'q' key to quit the program.

## Result:

![download](https://user-images.githubusercontent.com/92146886/219335892-86fc877f-8526-4ce0-beab-c36e391a2dc6.jpeg)

## Credits

This project is inspired by the [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) repository. The project uses their implementation of YOLOv5 for object detection, instance segmentation and their color mapping function for coloring the object masks. The project also uses the [ONNX Runtime](https://github.com/microsoft/onnxruntime) library for inference.

