"""
Robot Semantics
Visualize one single attention file.
"""
import os
import glob
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
    SETTINGS = ['Evaluation', 'WAM_Evaluation']
    VIDEO_NAME = 'wam_unknown_milk_milkcan2_mug6'


def load_attention_files(atts_path, video_path):
    """Load attention files and all frame path for a video file.
    """
    # Need to manually sort attention files by order due to system
    att_files = glob.glob(os.path.join(atts_path, '*.npy'))
    indices = np.argsort([int(x.split('_')[-3]) for x in att_files])
    att_files = np.array(att_files)[indices]

    # Get all video frames
    imgs_path = []
    num_images = len(glob.glob(os.path.join(video_path, '*.png')))
    for i in range(num_images):
        imgs_path.append(os.path.join(video_path, '{}_rgb.png'.format(i)))
    
    # Load all attention files
    # Newer attention weights in the newer clip will replace the older ones
    att_shape = np.load(att_files[0]).shape
    att_weights = np.zeros((num_images, att_shape[1]))
    for att_file in att_files:
        ind_att_weights = np.load(att_file)
        ind_att_weights = ind_att_weights.squeeze(2)
        start_frame_no, end_frame_no = int(att_file.split('_')[-3]), int(att_file.split('_')[-2])
        for i in range(ind_att_weights.shape[0]):
            att_weights[i+start_frame_no,:] = ind_att_weights[i,:]

    return imgs_path, att_weights


def main():
    # --------------------
    # Setup configuration class
    config = InferenceConfig()
    # Setup vocabulary
    vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
    config.VOCAB_SIZE = len(vocab)

    # Locate the exact setting of the video
    vname2settings = rs_rgbd.map_video_settings(config.DATASET_PATH, config.SETTINGS)
    # Path to video frames and attention files
    video_path = os.path.join(config.DATASET_PATH, vname2settings[config.VIDEO_NAME], config.VIDEO_NAME)
    atts_path = os.path.join(config.CHECKPOINT_PATH, 'attention', vname2settings[config.VIDEO_NAME])

    # Load attention files and map to every frames
    imgs_path, att_weights = load_attention_files(atts_path, video_path)
    
    # Visualize
    plots = visualize.visualize_region_atts_v2(imgs_path, att_weights)

if __name__ == '__main__':
    main()