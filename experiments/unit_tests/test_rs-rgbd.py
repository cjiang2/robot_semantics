import os
import sys

import numpy as np

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
print()
# Save all clips
for i in range(len(clips)):
    name = list(clips[i].keys())[0]
    clip, target = clips[i][name], targets[i]
    video_name = '_'.join(name.split('_')[:-1])
    print(video_name)
    print('{}\n[{}, {}]\n{}\n'.format(name, clip[0].split('/')[-1], clip[-1].split('/')[-1], target))