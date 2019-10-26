import os
import sys

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from datasets.rs_rgbd import *

# Test dataset reading
dataset_path = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
annotations = load_annotations(dataset_path, 'Grasp_Pour')
summary(annotations)

# Test accessing individual annotation
annotations_by_video = annotations['mug1_bottle1']
print(annotations_by_video)

# Test video parsing
videos = load_videos(dataset_path, 'Grasp_Pour')
print(videos['mug1_bottle1'])