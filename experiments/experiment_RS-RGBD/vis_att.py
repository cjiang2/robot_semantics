import os
import sys
import pickle

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.transform

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from v2c import utils
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

def retrieve_video_info(fname):
    fname = fname.split('_')
    start_frame_no, end_frame_no = int(fname[-3]), int(fname[-2])
    frames = []
    for i in range(start_frame_no, end_frame_no + 1, 1):
        frames.append('{}_rgb.png'.format(str(i)))
    video_folder = '_'.join(fname[:-3])
    return video_folder, frames

def visualize_region_atts(frames_path, 
                          alphas, 
                          smooth=True):
    """Visualizes region attention weight for all frames.
    """
    for t in range(len(frames_path)):
        frame = Image.open(frames_path[t])
        frame = np.asarray(frame.resize([7 * 24, 7 * 24], Image.BILINEAR))
        
        plt.subplot(np.ceil(len(frames_path) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (str(t)), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(frame)
        alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha.reshape(7,7), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha.reshape(7,7), [7 * 24, 7 * 24])
        plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
    

def main():
    # --------------------
    # Setup configuration class
    config = InferenceConfig()
    # Setup vocabulary
    vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
    config.VOCAB_SIZE = len(vocab)
    
    # Setup save path
    SAVE_PATH = os.path.join(config.CHECKPOINT_PATH, 'attention', config.SETTINGS[0])
    # unknown_water_bottle1_mug5_270_299_att.npy
    ATT_FILE = 'unknown_beverage_plastic_bottle5_mug5_75_104_att.npy'
    

    video_folder, frames = retrieve_video_info(ATT_FILE)
    frames = [os.path.join(config.DATASET_PATH, config.SETTINGS[0], video_folder, video_folder, x) for x in frames]
    alphas = np.load(os.path.join(SAVE_PATH, video_folder, ATT_FILE))
    alphas = np.squeeze(alphas, axis=2)
    
    visualize_region_atts(frames, alphas)

if __name__ == '__main__':
    main()