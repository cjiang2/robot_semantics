import os

import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

import rs.utils as utils

# ------------------------------
# TorchVision Utilities
# ------------------------------

# Parameter settings, from PyTorch deafult
TARGET_IMAGE_SIZE = (224, 224)
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]

# Transformers
transforms_data = transforms.Compose([transforms.Resize(TARGET_IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
                                     ])

# Video io
def _vidread(paths,
             transform=transforms_data):
    """Helper function to read a sequence of images.
    """
    frames = []
    for path in paths:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if transform:
            img = transform(img)
        frames.append(img)
    return frames

# (Clip, target) pytorch dataset object
class ClipDataset(data.Dataset):
    """Create an instance of generic clip dataset with pre-processed (clip_path, target).
    """
    def __init__(self, 
                 clips,
                 targets):
        self.clips, self.targets = clips, targets    # Load all (clip, target) pairs

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        Xv = np.load(self.clips[idx])
        clip_name = self.clips[idx].split(os.sep)[-1]
        S = self.targets[idx]
        return Xv, S, clip_name


# ------------------------------
# RS-RGBD Dataset Specifics
# ------------------------------

class RSRGBD_VideoDataset(data.Dataset):
    """Create an instance of RS-RGBD dataset for videos only.
    """
    def __init__(self, 
                 videos,
                 transform=transforms_data):
        self.videos = videos    # Load all frame images
        self.videos_str = list(videos.keys())
        self.transform = transform

    def __len__(self):
        return len(self.videos_str)

    def __getitem__(self, idx):
        video_str = self.videos_str[idx]
        frames = _vidread(self.videos[video_str], self.transform)
        if self.transform:
            frames = torch.stack(frames)
        return frames, video_str