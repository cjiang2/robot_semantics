import os
import sys

import numpy as np
import torch
from torch.utils import data

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

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
    MAXLEN = 10

# Setup configuration class
config = TrainConfig()
# Setup tf.dataset object
clips, targets, vocab, config = rs_rgbd.parse_clip_paths_and_captions(config)
config.display()
train_dataset = rs_rgbd.ClipDataset(clips, targets)

print('No. clips:', len(train_dataset))
print('Vocabulary:', len(vocab))
print(vocab.word_counts)