import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.rs_rgbd as rs_rgbd

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
clip_dataset = rs_rgbd.ClipDataset(clips, targets)

print()
# Test torch dataloader object
for i, (imgs, caption, clip_name) in enumerate(clip_dataset):
    print(clip_name)
    print(len(imgs))
    print(caption)
    print()
    plt.subplot(2,1,1)
    plt.imshow(np.asarray(imgs[0]))
    plt.subplot(2,1,2)
    plt.imshow(np.asarray(imgs[-1]))
    plt.show()