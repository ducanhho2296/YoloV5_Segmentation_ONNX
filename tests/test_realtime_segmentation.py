import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock

import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from realtime_segmentation import YOLOv5Segmentation

def mock_inference_session():
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name='input')]
    mock_session.get_outputs.return_value = [MagicMock(name='output1'), MagicMock(name='output2')]
    return mock_session

@pytest.fixture
def mock_yolov5_segmentation():
    yolo = YOLOv5Segmentation(None)
    yolo.session = mock_inference_session()
    yolo.input_name = 'input'
    yolo.output_name1 = 'output1'
    yolo.output_name2 = 'output2'
    return yolo

def test_load_model(mock_yolov5_segmentation):
    yolo = mock_yolov5_segmentation
    assert yolo.session is not None
    assert yolo.input_name == 'input'
    assert yolo.output_name1 == 'output1'
    assert yolo.output_name2 == 'output2'


if __name__ == "__main__":
    pytest.main([__file__])
