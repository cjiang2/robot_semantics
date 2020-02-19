import os
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

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
    SETTINGS = ['Evaluation', 'WAM_Evaluation', 'WAM_Intention']
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(95))

def init_model(config, 
               vocab,
               CHECKPOINT_FILE):
    # --------------------
    # Setup and build video2command training inference
    v2c_model = Video2Command(config, vocab)
    v2c_model.build(None)
    # Safely create prediction dir if non-exist
    if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
        os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))
    # Load back weights
    v2c_model.load_weights(CHECKPOINT_FILE)
    print('Model loading success.')
    return v2c_model

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

    # --------------------
    # Init test dataset
    clips, targets, _, _ = rs_rgbd.parse_clip_paths_and_captions(config, vocab)
    config.display()
    print('No. clips to evaluate:', len(clips), len(targets))

    # Dataset obj
    test_dataset = rs_rgbd.ClipDataset(clips, targets)
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  shuffle=False, 
                                  num_workers=1)
    
    # Setup all possible save paths
    for folder in config.SETTINGS:
        save_path = os.path.join(config.CHECKPOINT_PATH, 'attention', folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Get mapping
    vname2settings = rs_rgbd.map_video_settings(config.DATASET_PATH, config.SETTINGS)

    # --------------------
    # Init model
    if os.path.exists(config.CHECKPOINT_FILE):
        print('Loading saved model {}...'.format(config.CHECKPOINT_FILE))
        v2c_model = init_model(config, vocab, config.CHECKPOINT_FILE)

        # Evaluate
        y_pred, y_true, fnames, all_alphas = v2c_model.evaluate(test_loader)

        # Save all attention weights
        fnames = [x.replace('clip', 'att') for x in fnames]
        for i in range(len(fnames)):
            alphas, fname = all_alphas[i], fnames[i]
            video_folder, _ = retrieve_video_info(fname)

            video_path = os.path.join(config.CHECKPOINT_PATH, 'attention', vname2settings[video_folder])
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            
            np.save(os.path.join(video_path, fname), alphas)
            print('-'*30)
            print('Saved to {}'.format(os.path.join(video_path, fname)))
            print('Shape:', alphas.shape)

    print('Done.')

if __name__ == '__main__':
    main()