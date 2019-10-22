import os
import sys

import numpy as np
import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *

BACKBONE_NAME = 'resnet50'
POOLING = None

cnn_wrapper = CNNWrapper(BACKBONE_NAME, POOLING)
img = torch.ones((1, 3, 224, 224))
print(img.shape)
out = cnn_wrapper(img)
print(out.shape)