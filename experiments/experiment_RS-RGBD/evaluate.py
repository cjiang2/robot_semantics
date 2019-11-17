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
    MAXLEN = 10
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(150))
    SETTINGS = {'test': ['Evaluation']}

def init_model():
    # --------------------
    # Setup configuration class
    config = InferenceConfig()
    # Setup vocabulary
    vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))
    config.VOCAB_SIZE = len(vocab)

    # --------------------
    # Setup and build video2command training inference
    v2c_model = Video2Command(config)
    v2c_model.build(None)

    # Safely create prediction dir if non-exist
    if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
        os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))

    # Load back weights
    v2c_model.load_weights(config.CHECKPOINT_FILE)
    print('Model loading success.')
    return v2c_model, vocab, config

def main():
    v2c_model, vocab, config = init_model()
    clips, targets, _, _ = rs_rgbd.parse_clip_paths_and_captions(config, vocab)
    config.display()
    print('No. clips to evaluate:', len(clips), len(targets))

    test_dataset = rs_rgbd.ClipDataset(clips, targets)
    test_loader = data.DataLoader(test_dataset, 
                                batch_size=config.BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=1)

    y_pred, y_true, names = v2c_model.evaluate(test_loader, vocab)

    # Save to evaluation file
    f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction', 'prediction_{}.txt'.format(150)), 'w')

    for i in range(len(y_pred)):
        #print(y_pred[i])
        pred_command = utils.sequence_to_text(y_pred[i], vocab)
        #print(y_true[i])
        true_command = utils.sequence_to_text(y_true[i], vocab)
        f.write('------------------------------------------\n')
        f.write(names[i] + '\n')
        # Prediction
        # Ground truth
        f.write(pred_command + '\n')
        f.write(true_command + '\n')

    print('Ready for cococaption.')

if __name__ == '__main__':
    main()