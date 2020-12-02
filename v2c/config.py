"""
Robot Semantics
Base configuration class.
Reference from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
"""

import numpy as np
import os
import multiprocessing

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # --------------------
    # Basics of the configurations.
    NAME = None  # Override in sub-classes
    MODE = 'train'  # Mode (train/eval)
    ROOT_DIR = None     # Root project directory

    # --------------------
    # Training Parameters
    # Learning rate
    LEARNING_RATE = 1e-4

    # Batch size
    BATCH_SIZE = 16

    # Epochs
    NUM_EPOCHS = 50

    # Saved model path
    CHECKPOINT_PATH = os.path.join('checkpoints')

    # Display every # steps
    DISPLAY_EVERY = 20

    # Save model every ? epoch
    SAVE_EVERY = 1

    # --------------------
    # Model hyperparameters
    # Backbone & num_features used for feature extraction
    BACKBONE = {'resnet50': 2048}

    # Unit size for LSTM, Dense
    UNITS = 256

    # Embedding size
    EMBED_SIZE = 300

    # Size for Vocabulary
    VOCAB_SIZE = None

    # Teacher-forcing Ratio, Randomly choose between using 
    # gt target vs. using pred target during training
    # This will slow down the convergence rate but force exploration 
    # against exploitation.
    TEACHER_FORCING_RATIO = 1.0 

    # --------------------
    # Parameters for dataset configuration
    # Path to currently used dataset
    DATASET_PATH = os.path.join('datasets') # Override in sub-classes

    # All settings to be used
    SETTINGS = ['Grasp_Pour', 'WAM_Grasp_Pour', 'Human_Intention']

    # Maximum command sentence length
    MAXLEN = 15

    # --------------------
    # Parameters for NLP parsing settings
    # Word frequency
    FREQUENCY = None

    # Whether to use bias vector, which is related to the log 
    # probability of the distribution of the labels (words) and how often they occur.
    USE_BIAS_VECTOR = True

    # Special tokens to be added into vocabulary
    START_WORD = '<sos>'
    END_WORD = '<eos>'
    UNK_WORD = None

    # --------------------
    # Parameters for video streamline configuration
    # Size for video observation window
    WINDOW_SIZE = 30

    # Maximum number of frame needed to append to the queue for a clip retrieval
    RETRIEVAL_LIMIT = 15

    def __init__(self):
        """Set values of computed attributes."""
        # Workers used for dataset object...
        if os.name is 'nt':
            self.WORKERS = 0
        else:
            self.WORKERS = multiprocessing.cpu_count()

        # Some forced assertion
        assert self.MODE in ['train', 'test']

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        print("-"*30)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()