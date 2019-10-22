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
    # Name the configurations.
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
    NUM_EPOCHS = 150

    # Saved model path
    CHECKPOINT_PATH = os.path.join('checkpoints')

    # Display every # steps
    DISPLAY_EVERY = 20

    # Save model every ? epoch
    SAVE_EVERY = 5

    # --------------------
    # Model hyperparameters
    # Backbone & num_features used for feature extraction
    BACKBONE = {'resnet50': 2048}

    # Unit size for LSTM, Dense
    UNITS = 512

    # Embedding size
    EMBED_SIZE = 512

    # Size for Vocabulary
    VOCAB_SIZE = None

    # Size for video observation window
    WINDOW_SIZE = 30

    # --------------------
    # Parameters for dataset, tf.dataset configuration
    # Path to currently used dataset
    DATASET_PATH = os.path.join('datasets') # Override in sub-classes

    # Maximum command sentence length
    MAXLEN = 10

    # Buffer size for tf.dataset shuffling
    BUFFER_SIZE = 1000

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