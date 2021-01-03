"""
RS Concepts
Training script.

"""
import os
import sys
import pickle

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import rs utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from rs.config import Config
from rs.models import *
from rs import utils
from rs.datasets import rs_rgbd
from rs.datasets import loader

# Configuration for hperparameters
class TrainConfig(Config):
    """Configuration for training with RS-RGBD.
    """
    NAME = 'v2l_RS-RGBD'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'RS-RGBD')
    TASKS = ['human_grasp_pour', 
             'wam_grasp_pour']

# Setup configuration class
config = TrainConfig()

# Load RS-RGBD clip dataset
clips_fpath, targets, vocab, config = utils.prepare_data(config)
config.display()

# Clip dataset object and pytorch loader
train_dataset = loader.ClipDataset(clips_fpath, targets)
train_loader = data.DataLoader(train_dataset, 
                               batch_size=config.BATCH_SIZE, 
                               shuffle=True, 
                               num_workers=config.WORKERS)
bias_vector = vocab.get_bias_vec()

# Setup and build video2lang training inference
v2l_model = Video2Lang(config, vocab)
v2l_model.build(bias_vector)

# Save vocabulary at last
with open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'wb') as f:
    pickle.dump(vocab, f)

# Start training
v2l_model.train(train_loader)