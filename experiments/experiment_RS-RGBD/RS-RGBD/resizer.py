import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.config import *
import datasets.rs_rgbd as rs_rgbd

# Configuration for hperparameters
class ResizerConfig(Config):
    """Configuration for feature extraction with RS-RGBD.
    """
    NAME = 'Re_RS-RGBD'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
    MAXLEN = 10
    WINDOW_SIZE = 30
    STEP = 15
    BACKBONE = {'resnet50': 2048}
    SIZE = 224
    ALG_VAL = Image.BOX
    SETTINGS = ['WAM_Grasp_Pour', 'Grasp_Pour', 'WAM_Evaluation', 'Evaluation', 'Human_Intention', 'WAM_Intention']

if __name__ == '__main__':
    config = ResizerConfig()
    config.display()

    # Save path for resized images
    SAVE_PATH = os.path.join(config.DATASET_PATH, 'resized')
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for folder in config.SETTINGS:
        print('-'*30)
        print('Settings {}...'.format(folder))
        videos = rs_rgbd.load_videos(config.DATASET_PATH, folder)

        videos_name = list(videos.keys())
        for video in videos_name:
            print('Processing video: "{}"...'.format(video))
            VIDEO_PATH = os.path.join(SAVE_PATH, video)
            if not os.path.exists(VIDEO_PATH):
                os.makedirs(VIDEO_PATH)

            for img_path in videos[video]:
                img = Image.open(img_path)
                img_fname = img_path.split('/')[-1]
                if img.mode != "RGB":
                    img = img.convert(mode="RGB")

                img_resized = img.resize((config.SIZE, config.SIZE), config.ALG_VAL)
                img_resized.save(os.path.join(VIDEO_PATH, img_fname))

            print('Done.')