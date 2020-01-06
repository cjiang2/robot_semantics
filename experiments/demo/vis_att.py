"""
Robot Semantics
Visualize one single attention file.
"""
import os
import sys
import pickle

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from v2c import utils
from v2c import visualize
from v2c.config import *
from datasets import rs_rgbd

# Configuration for hperparameters
class InferenceConfig(Config):
    """Configuration for training with RS-RGBD.
    """
    NAME = 'v2c_RS-RGBD'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
    SETTINGS = ['Evaluation']
    SAVE_PATH = os.path.join(CHECKPOINT_PATH, 'attention', SETTINGS[0])
    # unknown_water_bottle1_mug5_270_299_att.npy
    ATT_FILE = 'unknown_beverage_plastic_bottle5_mug5_75_104_att.npy'

def retrieve_video_info(fname):
    fname = fname.split('_')
    start_frame_no, end_frame_no = int(fname[-3]), int(fname[-2])
    frames = []
    for i in range(start_frame_no, end_frame_no + 1, 1):
        frames.append('{}_rgb.png'.format(str(i)))
    video_folder = '_'.join(fname[:-3])
    return video_folder, frames

def main():
    # --------------------
    # Setup configuration class
    config = InferenceConfig()
    # Setup vocabulary
    vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
    config.VOCAB_SIZE = len(vocab)

    # Retrieve frame paths given the attention weight fname
    video_folder, frames = retrieve_video_info(config.ATT_FILE)
    frames = [os.path.join(config.DATASET_PATH, config.SETTINGS[0], video_folder, video_folder, x) for x in frames]
    
    # Load back attention
    alphas = np.load(os.path.join(config.SAVE_PATH, video_folder, config.ATT_FILE))
    alphas = np.squeeze(alphas, axis=2)
    
    # Visualize
    visualize.visualize_region_atts(frames, alphas)

if __name__ == '__main__':
    main()