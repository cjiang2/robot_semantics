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
    SETTINGS = ['Evaluation']

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

    # --------------------
    # Init model
    for idx in range(0, config.NUM_EPOCHS + 1, config.SAVE_EVERY):
        CHECKPOINT_FILE = os.path.join(config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(idx))
        if os.path.exists(CHECKPOINT_FILE):
            print('Loading saved model {}...'.format(CHECKPOINT_FILE))
            v2c_model = init_model(config, vocab, CHECKPOINT_FILE)

            # Evaluate
            y_pred, y_true, fnames = v2c_model.evaluate(test_loader, vocab)

            # Save to evaluation file
            f = open(os.path.join(config.CHECKPOINT_PATH, 'prediction', 'prediction_{}.txt'.format(idx)), 'w')
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