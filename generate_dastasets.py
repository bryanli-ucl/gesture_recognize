import os
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['__VK_LAYER_NV_optimus'] = 'NVIDIA_only'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# EGL Config
os.environ['EGL_PLATFORM'] = 'surfaceless' # or device
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
import numpy as np

import cv2

base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task',
    delegate=python.BaseOptions.Delegate.GPU
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

detector = vision.HandLandmarker.create_from_options(options)

while True:
    _, gbr = cap.read()    
    rgb = cv2.cvtColor(gbr, cv2.COLOR_BGR2RGB)

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
    det_res = detector.detect_for_video(img, timestamp_ms)
    
    cv2.imshow("IMAGE", gbr)
    if cv2.waitKey(5) & 0xFF == 27: # ESC
        break
