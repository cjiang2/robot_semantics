import os
import sys

import numpy as np
import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *

# Hyperparameters
MAXLEN = 30
BATCH_SIZE = 16

# ----------------------------------------
# UnitTest on video encoder
# ----------------------------------------
video_encoder = VideoEncoder(2048, 512)

Xv = torch.ones((BATCH_SIZE, 30, 2048), dtype=torch.float32)
Xv, states = video_encoder(Xv)

# ----------------------------------------
# UnitTest on command decoder
# ----------------------------------------
command_decoder = CommandDecoder(units=512, vocab_size=150, embed_dim=512)
S = torch.ones((BATCH_SIZE, MAXLEN), dtype=torch.long)

# ----------------------------------------
# Test encoding on video features
# ----------------------------------------
timestep = 4
probs, states = command_decoder(S[:,timestep], states)
target = S[:,timestep]
# Randomly set 0 padding entries
S[:,13] = 0
S[:,10] = 0

# UnitTest on loss objective
loss_objective = CommandLoss()
loss = loss_objective(probs, S[:,timestep])
print(loss)

target = S[:,timestep]
target[13] = 0
target[15] = 0
num_nonzeros = (target != 0).sum()
S_mask = (target != 0)
print(target, target.shape)
print(num_nonzeros)
print(S_mask.shape)