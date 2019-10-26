import os
import sys

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import v2c.utils as utils
from v2c.config import *
import datasets.rs_rgbd as rs_rgbd

# Configuration for hperparameters
class FEConfig(Config):
    """Configuration for feature extraction with RS-RGBD.
    """
    NAME = 'FE_RS-RGBD'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
    SETTINGS = ['Grasp_Pour']
    MAXLEN = 10
    WINDOW_SIZE = 30
    STEP = 15

def save_clips(clips, 
               targets,
               names,
               config,
               save_path):
    # Create the save path if non-existing
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save all clips
    for i in range(len(clips)):
        clip, target, name = clips[i], targets[i], names[i]
        video_name = '_'.join(name.split('_')[:-1])
        print(video_name)
        print('{}\n[{}, {}]\n{}\n'.format(name, clip[0].split('/')[-1], clip[-1].split('/')[-1], target))
        
    return

if __name__ == '__main__':
    config = FEConfig()
    config.display()

    folder = 'Grasp_Pour'
    clips, targets, names = rs_rgbd.generate_clips(config.DATASET_PATH, config.SETTINGS, config.WINDOW_SIZE)
    print('Number of clips:', len(clips), len(targets), len(names))

    save_clips(clips, targets, names, config, os.path.join(config.DATASET_PATH, 'clips'))