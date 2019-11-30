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
class TrainConfig(Config):
    """Configuration for training with RS-RGBD.
    """
    NAME = 'v2c_RS-RGBD'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')

# Setup configuration class
config = TrainConfig()
# Setup tf.dataset object
clips, targets, vocab, config = rs_rgbd.parse_clip_paths_and_captions(config)
config.display()
train_dataset = rs_rgbd.ClipDataset(clips, targets)
train_loader = data.DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=config.WORKERS)
bias_vector = vocab.get_bias_vector() if config.USE_BIAS_VECTOR else None

# Setup and build video2command training inference
v2c_model = Video2Command(config)
v2c_model.build(bias_vector)

# Save vocabulary at last
with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

# Start training
v2c_model.train(train_loader)