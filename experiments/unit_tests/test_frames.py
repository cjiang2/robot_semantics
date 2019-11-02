import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.rs_rgbd as rs_rgbd

videos = rs_rgbd.load_videos(dataset_path=os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD'),
                             folder='Test')

# Fetch all frames
frames = []
for video_name in videos.keys():
    frames += videos[video_name]

# Create the dataset object
frame_dataset = rs_rgbd.FrameDataset(frames, transform=rs_rgbd.transforms_data)

print()
# Test torch dataloader object
for i, frame in enumerate(frame_dataset):
    frame = (frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    plt.imshow(frame)
    plt.pause(0.01)