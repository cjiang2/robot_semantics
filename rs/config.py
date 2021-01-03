"""
RS Concepts
Base configuration class.

Reference from: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
"""

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
    ROOT_DIR = None     # Root project directory, Override in sub-classes
    CHECKPOINT_PATH = os.path.join('checkpoints')   # Saved run path
    DATASET_PATH = os.path.join('data')    # Path to dataset files

    # --------------------
    # Parameters for NLP
    # Size for Vocabulary
    VOCAB_SIZE = None  # Override in sub-classes

    # Special tokens to be added into vocabulary
    START_WORD = '<sos>'
    END_WORD = '<eos>'
    UNK_WORD = None

    # Maximum sequence length allowed
    MAXLEN = 10

    # Word frequency
    FREQUENCY = None

    # --------------------
    # Hyperparameters
    # Batch size
    BATCH_SIZE = 16

    # Epochs
    NUM_EPOCHS = 50

    # Learning rate
    LEARNING_RATE = 1e-4

    # Learning rate decay, decay by 0.1 every ? epoch
    LR_DECAY_EVERY = [5, 40]

    # Whether to clip gradient or not
    CLIP_NORM = 5   # None

    # Unit size for LSTM, Dense
    UNITS = 512

    # Word embedding size, default to 300 matching word2vec
    EMBED_DIM = 512

    # Backbone & num_features used for feature extraction
    BACKBONE = 'resnet50'

    # If true, load full CNN model
    LOAD_CNN = False  

    # --------------------
    # Training Parameters
    # Display every # steps
    DISPLAY_EVERY = 20

    # Save model every ? epoch
    SAVE_EVERY = 1

    # --------------------
    # Parameters for RS-RGBD dataset configuration
    # Manipulation task divisions to be used
    TASKS = ['human_grasp_pour', 
             'wam_grasp_pour',
             'eval_human_grasp_pour',
             'eval_wam_grasp_pour',
             'human_point_and_intend',
             'wam_point_and_intend',
             'eval_wam_grasp_pour_complex']

    # Semantic annotation to use for learning
    ANNOT_TO_USE = 'command'

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
        assert self.MODE in ['train', 'eval']

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        print("-"*30)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print()