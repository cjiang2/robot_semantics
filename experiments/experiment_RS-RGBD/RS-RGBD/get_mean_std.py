import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.rs_rgbd as rs_rgbd

def calculate_mean_std(frame_dataset):
    """Calculate per-channel mean over the entire dataset.
    Should get similar values comapred to ImageNet mean and std.
    """
    mean = 0.
    std = 0.
    for i, frame in enumerate(frame_dataset):
        frame = frame.view(frame.size(0), -1)
        print('Processing image', i, frame.shape)
        mean += frame.mean(1)
        std += frame.std(1)

    mean /= len(frame_dataset)
    std /= len(frame_dataset)
    return mean, std

if __name__ == '__main__':
    videos = rs_rgbd.load_videos(dataset_path=os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD'),
                                 folder='Grasp_Pour')

    # Fetch all frames
    frames = []
    for video_name in videos.keys():
        frames += videos[video_name]

    # Create the dataset object
    transforms_unnormalized = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                 ])
    frame_dataset = rs_rgbd.FrameDataset(frames, transform=transforms_unnormalized)
    mean, std = calculate_mean_std(frame_dataset)
    print('mean: {}, std: {}'.format(mean, std))