import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import v2c.utils as utils
from v2c.config import *
from v2c.model import *
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
    MAXLEN = 10
    WINDOW_SIZE = 30
    STEP = 15
    BACKBONE = {'resnet50': 2048}
    SETTINGS = ['WAM_Grasp_Pour', 'Grasp_Pour', 'WAM_Evaluation', 'Evaluation']

def extract(dataset_path,
            dataset,
            model_name,
            folder):
    """Wrapper function to extract CNN features and save both features
    and captions as single numpy files.
    """
    # Create output directory
    output_path = os.path.join(dataset_path, model_name, folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare pre-trained model
    print('Loading pre-trained model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNNWrapper(backbone=model_name,
                       checkpoint_path=os.path.join(ROOT_DIR, 'checkpoints', 'backbone', 'resnet50.pth'))
    model.eval()
    model.to(device)
    print('Done loading.')

    # Feature extraction
    for i, (Xv, S, clip_name) in enumerate(dataset):
        with torch.no_grad():
            print('-'*30)
            print('Processing clip {}...'.format(clip_name))
            #print(imgs_path, clip_name)
            #assert len(imgs_path) == 30
            # Get features
            Xv = Xv.to(device)
            outputs = model(Xv)

            # Save into clips
            feature_fpath = os.path.join(output_path, clip_name+'_clip.npy')
            np.save(feature_fpath, outputs.cpu().numpy())

            # Save caption
            caption_fpath = os.path.join(output_path, clip_name+'_caption.npy')
            np.save(caption_fpath, S)
            
            print('{}: {}'.format(clip_name+'_clip.npy', S))
            print('Shape: {}\nFeature saved to {}\nCaption saved to {}.'.format(outputs.shape, 
                                                                                feature_fpath,
                                                                                caption_fpath))
    del model
    return

if __name__ == '__main__':
    config = FEConfig()
    config.display()

    for folder in config.SETTINGS:
        print('-'*30)
        print('Settings {}...'.format(folder))
        clips, targets = rs_rgbd.generate_clips(config.DATASET_PATH, folder, config.WINDOW_SIZE)
        print('Number of clips:', len(clips), len(targets))
        clip_dataset = rs_rgbd.Frames2ClipDataset(clips, targets, transform=rs_rgbd.transforms_data)

        extract(config.DATASET_PATH, clip_dataset, list(config.BACKBONE.keys())[0], folder)
        print('-'*30)