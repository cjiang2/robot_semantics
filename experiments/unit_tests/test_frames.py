import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.rs_rgbd as rs_rgbd

# Frame-only pytorch dataset object
class FrameDataset(data.Dataset):
    """Create an instance of RS-RGBD dataset with all the frames only.
    """
    def __init__(self, 
                 frames,
                 transform=None):
        self.frames = frames    # Load all frame images
        self.transform = transform

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self._imread(self.frames[idx])
        return img

videos = rs_rgbd.load_videos(dataset_path=os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD'),
                             folder='Grasp_Pour')

# Fetch all frames
frames = []
for video_name in videos.keys():
    frames += videos[video_name]

# Create the dataset object
frame_dataset = FrameDataset(frames, transform=rs_rgbd.transforms_data)

print()
# Test torch dataloader object
for i, frame in enumerate(frame_dataset):
    frame = (frame.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    plt.imshow(frame)
    plt.pause(0.001)