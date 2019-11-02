import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.rs_rgbd as rs_rgbd
from v2c.config import *

# Test dataset reading
dataset_path = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
print(dataset_path)
annotations = rs_rgbd.load_annotations(dataset_path, 'Grasp_Pour')
rs_rgbd.summary(annotations)
print()

# Test accessing individual annotation
annotations_by_video = annotations['mug1_bottle1']
#print(annotations_by_video)

# Test video parsing
videos = rs_rgbd.load_videos(dataset_path, 'Grasp_Pour')
#print(videos['mug1_bottle1'])

# Test clip generation method
clips, targets = rs_rgbd.generate_clips(dataset_path, 'Grasp_Pour', 30)
print('Number of clips:', len(clips), len(targets))

"""
# Print all clips
for i in range(len(clips)):
    name = list(clips[i].keys())[0]
    clip, target = clips[i][name], targets[i]
    video_name = '_'.join(name.split('_')[:-1])
    print(video_name)
    print('{}\n[{}, {}]\n{}\n'.format(name, clip[0].split('/')[-1], clip[-1].split('/')[-1], target))
"""

# Test pytorch dataset object
frames2clip_dataset = rs_rgbd.Frames2ClipDataset(clips, targets, transform=rs_rgbd.transforms_data)

print()
# Test torch dataloader object
for i, (imgs, caption, clip_name) in enumerate(frames2clip_dataset):
    print(clip_name)
    print(len(imgs))
    print(caption)
    print()
    img0 = (imgs[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img1 = (imgs[-1].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    plt.subplot(2,1,1)
    plt.imshow(img0)
    plt.subplot(2,1,2)
    plt.imshow(img1)
    #plt.pause(0.0001)
    plt.show()
    break

class TrainConfig(Config):
    """Configuration for training with RS-RGBD.
    """
    NAME = 'v2c_RS-RGBD'
    MODE = 'train'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'RS-RGBD')
    MAXLEN = 10

config = TrainConfig()
clips, targets, vocab, config = rs_rgbd.parse_clip_paths_and_captions(config)
clip_dataset = rs_rgbd.ClipDataset(clips, targets, transform=rs_rgbd.transforms_data)
print(vocab.word2idx)
print(vocab.word_counts)

print()
# Test torch dataloader object
for i, (Xv, S, clip_name) in enumerate(clip_dataset):
    print(Xv.shape, S.shape, clip_name)
    break