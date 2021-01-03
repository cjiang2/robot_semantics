"""
RS Concepts
Script for Feature Extraction.
"""
import os
import sys

import numpy as np
import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import RS
sys.path.append(ROOT_DIR)  # To find local version of the library
from rs.config import Config
from rs.models import CNNExtractor
from rs.datasets import rs_rgbd
from rs.datasets import loader

class RSRGBDConfig(Config):
    """Configuration for UnitTest
    """
    NAME = 'RS_RGBD_feat_ext'
    ROOT_DIR = ROOT_DIR
    DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'RS-RGBD')
    BACKBONE = 'resnet50'
    WEIGHTS_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'weights', '{}.pth'.format(BACKBONE))
    BATCH_SIZE = 256

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(dataset, config):
    """Extract CNN visual features and save to disk as numpy files.
    """
    # Prepare output folder
    output_path = os.path.join(config.DATASET_PATH, config.BACKBONE)
    for task in config.TASKS:
        create_folder(os.path.join(output_path, task))

    # Prepare pre-trained model
    print('Loading pre-trained model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNNExtractor(backbone=config.BACKBONE,
                         weights_path=config.WEIGHTS_PATH,
                         save_grad=False)
    model.eval()
    model.to(device)
    print('Done, starting feature extraction...')

    # Feature extraction
    for _, (frames, video_str) in enumerate(dataset):
        with torch.no_grad():
            print('-'*30)
            print('Processing video: {}...'.format(video_str))
            frames = frames.to(device)
            outputs = []
            for i in range(0, frames.shape[0], config.BATCH_SIZE):
                inputs = frames[i:i+config.BATCH_SIZE]
                out = model(inputs)
                outputs.append(out)
            outputs = torch.cat(outputs, dim=0)

            # Save into clips
            task_folder, video_name = video_str.split(os.sep)
            outfile_path = os.path.join(output_path, task_folder, video_name+'.npy')
            np.save(outfile_path, outputs.cpu().numpy())
            print('{}/{}'.format(task_folder, video_name+'.npy'))
            print('Shape: {}, saved to {}.'.format(outputs.shape, outfile_path))

    del model
    return

if __name__ == '__main__':
    config = RSRGBDConfig()
    config.display()

    # Load video dataset
    videos = rs_rgbd.load_videos(config.TASKS, 
                                 config.DATASET_PATH)
    video_dataset = loader.RSRGBD_VideoDataset(videos=videos)

    extract(video_dataset, config)
