"""
RS Concepts
Evaluation script.

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
class InferenceConfig(Config):
    """Configuration for training with RS-RGBD.
    """
    NAME = 'v2l_RS-RGBD'
    MODE = 'eval'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'RS-RGBD')

def init_model(config, 
               vocab,
               CHECKPOINT_FILE):
    # --------------------
    # Setup and build video2lang testing inference
    v2l_model = Video2Lang(config, vocab)
    v2l_model.build(None)

    # Safely create prediction dir if non-exist
    if not os.path.exists(os.path.join(config.CHECKPOINT_PATH, 'prediction')):
        os.makedirs(os.path.join(config.CHECKPOINT_PATH, 'prediction'))

    # Load back weights
    v2l_model.load_weights(CHECKPOINT_FILE)
    return v2l_model

def main():
    # --------------------
    # Setup configuration class
    config = InferenceConfig()
    # Setup vocabulary
    vocab = pickle.load(open(os.path.join(config.CHECKPOINT_PATH, 'vocab.pkl'), 'rb'))

    # --------------------
    # Init test dataset
    # Load RS-RGBD clip dataset
    clips_fpath, targets, _, config = utils.prepare_data(config, 
                                                         vocab=vocab)
    config.display()
    print('No. clips to evaluate:', len(clips_fpath), len(targets))

    # Dataset obj
    eval_dataset = loader.ClipDataset(clips_fpath, targets)
    eval_loader = data.DataLoader(eval_dataset, 
                                  batch_size=config.BATCH_SIZE, 
                                  shuffle=False, 
                                  num_workers=1)

    # --------------------
    # Init model
    for idx in range(0, config.NUM_EPOCHS + 1, config.SAVE_EVERY):
        CHECKPOINT_FILE = os.path.join(config.CHECKPOINT_PATH, 'saved', 'v2l_epoch_{}.pth'.format(idx))
        if os.path.exists(CHECKPOINT_FILE):
            print('Loading saved model {}...'.format(CHECKPOINT_FILE))
            v2l_model = init_model(config, vocab, CHECKPOINT_FILE)

            # Evaluate
            y_pred, y_true, fnames, _ = v2l_model.evaluate(eval_loader)

            # Save to evaluation file
            f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction', '{}_prediction.txt'.format(idx)), 'w')
            for i in range(len(y_pred)):
                pred_command = utils.sequence_to_text(y_pred[i], vocab)
                true_command = utils.sequence_to_text(y_true[i], vocab)
                f.write('------------------------------------------\n')
                f.write(fnames[i] + '\n')
                f.write(pred_command + '\n')    # Prediction
                f.write(true_command + '\n')    # Ground truth
                #print('------------------------------------------')
                #print('Clip Name:', fnames[i])
                #print('Pred:', pred_command)
                #print('True:', true_command)

    print('Ready for cococaption.')

if __name__ == '__main__':
    main()