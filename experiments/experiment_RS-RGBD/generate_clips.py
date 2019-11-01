import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

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
    SETTINGS = ['Grasp_Pour']
    MAXLEN = 10
    WINDOW_SIZE = 30
    STEP = 15

def extract(dataset_path,
            dataset,
            model_name):
    # Create output directory
    output_path = os.path.join(dataset_path, model_name)
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
            Xv = Xv.to(device)
            print('-'*30)
            print('Processing clip {}...'.format(clip_name))
            #print(imgs_path, clip_name)
            #assert len(imgs_path) == 30
            outputs = model(Xv)
            outputs = outputs.view(outputs.shape[0], -1)

            # Save into clips
            outfile_path = os.path.join(output_path, clip_name+'_clip.npy')
            np.save(outfile_path, outputs.cpu().numpy())
            # Save caption
            outfile_path = os.path.join(output_path, clip_name+'_caption.npy')
            np.save(outfile_path, S)
            print('{}: {}'.format(clip_name+'_clip.npy', S))
            print('Shape: {}, saved to {}.'.format(outputs.shape, outfile_path))
    del model
    return

if __name__ == '__main__':
    config = FEConfig()
    model_names = ['resnet50']
    config.display()

    folder = 'Grasp_Pour'
    clips, targets = rs_rgbd.generate_clips(config.DATASET_PATH, folder, config.WINDOW_SIZE)
    print('Number of clips:', len(clips), len(targets))
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    clip_dataset = rs_rgbd.ClipDataset(clips, targets, transform=transform)

    extract(config.DATASET_PATH, clip_dataset, model_names[0])