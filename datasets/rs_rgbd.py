import os
import sys
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils import data

# ----------------------------------------
# Functions for RS-RGBD Database Integration
# ----------------------------------------

def load_annotations(dataset_path=os.path.join('datasets', 'RS-RGBD'),
                     folder='Grasp_Pour'):
    """Helper function to parse RS-RGBD videos with annotations.
    For traning purpose mainly.
    """
    def read_caption_file(path):
        """Parse individual annotation text file.
        """
        timestamps = []
        captions = []
        caption_name = path.split('/')[-1][:-4]
        f = open(path, 'r')
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            start_frame, end_frame = lines[i].strip().split(', ')
            start_frame, end_frame = int(start_frame), int(end_frame)
            caption = lines[i+1].strip().split(', ')
            #print(start_frame, end_frame, caption)
            timestamps.append([start_frame, end_frame])
            captions.append(caption)
        f.close()
        return caption_name, timestamps, captions

    def get_frames_no(timestamp):
        frames = []
        for i in range(timestamp[0], timestamp[1] + 1, 1):
            frames.append(i)
        return frames

    # Paths
    folder_path = os.path.join(dataset_path, folder)
    videos_name = os.listdir(folder_path)

    # Collection annotation for each video
    annotations = {}
    for video_name in videos_name:
        #print('-'*30)
        annotation_by_video = {}
        video_path = os.path.join(folder_path, video_name)
        # Read each type of caption annotation
        captions_path = glob.glob(os.path.join(video_path, 'semantics', '*.txt'))
        for caption_path in captions_path:
            caption_name, timestamps, captions = read_caption_file(caption_path)
            annotation_by_video[caption_name] = [timestamps, captions]
            #print(timestamps, len(timestamps))
            #print(captions, len(captions))
            #print()
        annotations[video_name] = annotation_by_video
        #print()

    return annotations

def load_videos(dataset_path=os.path.join('datasets', 'RS-RGBD'),
                folder='Grasp_Pour'):
    """Parse RS-RGBD dataset, only frame images. Specifically used
    for evaluation folders.
    """
    # Paths
    folder_path = os.path.join(dataset_path, folder)
    videos_name = os.listdir(folder_path)

    # Collect videos
    videos = {}
    for video_name in videos_name:
        # Safely parse all frame images sorted
        images_path = []
        num_images = len(glob.glob(os.path.join(folder_path, video_name, video_name, '*.png')))
        for i in range(num_images):
            images_path.append(os.path.join(folder_path, video_name, video_name, '{}_rgb.png'.format(i)))
        videos[video_name] = images_path
    return videos

def summary(annotations):
    """Helper function for RS-RGBD dataset summary.
    """
    num_frames = 0
    num_lefthand = 0
    num_righthand = 0
    num_static = 0
    for video_name in annotations.keys():
        annotations_by_video = annotations[video_name]
        for annotation_type in annotations_by_video.keys():
            if annotation_type == 'lefthand':
                if len(annotations_by_video[annotation_type][0]) != 1:
                    num_lefthand += len(annotations_by_video[annotation_type][0])
            elif annotation_type == 'righthand':
                if len(annotations_by_video[annotation_type][0]) != 1:
                    num_righthand += len(annotations_by_video[annotation_type][0])
            else:
                num_static += len(annotations_by_video[annotation_type][0])
        
    print('# videos in total:', len(annotations))
    print('# lefthand manipulator captions in total:', num_lefthand)
    print('# righthand manipulator captions in total:', num_righthand)
    print('# manipulator captions in total:', num_lefthand + num_righthand)
    print('# static captions in total:', num_static)