"""
RS Concepts
Script for Offline Clip Sampling and Generation.
"""
import os
import sys

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../")

# Import RS
sys.path.append(ROOT_DIR)  # To find local version of the library
from rs import utils
from rs.config import Config
from rs.datasets import rs_rgbd

class RSRGBDConfig(Config):
    """Configuration for UnitTest
    """
    NAME = 'RS_RGBD_clip_gen'
    ROOT_DIR = ROOT_DIR
    CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
    DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'RS-RGBD')
    BACKBONE = 'resnet50'

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_label(label, 
                  config):
    if config.ANNOT_TO_USE == 'action':
        # NOTE: For action label, use the first action command annotion
        # Matching Robot_Semantics Experiment
        label = label[0]
    elif config.ANNOT_TO_USE == 'kb':
        # NOTE: Separate knowledge base triples by a comma
        label = ' <eot> '.join(label)
        #label = label.replace('_', ' ')
        #print(label)
    elif config.ANNOT_TO_USE == 'command':
        label = label[0]
        #label = label.replace('wam_robot', 'wam')
        #label = label.replace('_', ' ')

    return label

def generate_clips(features, 
                   annotations,
                   config):
    """Helper to sample clips for model training and evaluating.
    """
    # Load feature by video 1st
    for video_str in features:
        # Save path
        if 'eval' not in video_str:
            output_path = os.path.join(config.CHECKPOINT_PATH, 'train')
        else:
            output_path = os.path.join(config.CHECKPOINT_PATH, 'eval')
        create_folder(output_path)

        # Separate video_name from video_str
        video_name = video_str.split(os.sep)[1]

        # Prepare video stream object
        stream = utils.StreamSimple(window_size=config.WINDOW_SIZE)

        # Load saved numpy feature
        feat = np.load(features[video_str])
        num_frames = feat.shape[0]

        # Generate frame indices for all possible clips
        clips = []
        frame_ranges = []
        for i in range(num_frames):
            stream.add_frame(i)   # Can add either a real frame or just an integer index
            indices = stream.get_clip()
            if indices is not None:
                start_idx, end_idx = indices[0], indices[-1]
                clip = feat[start_idx:end_idx+1]
                clips.append(clip)
                frame_ranges.append((start_idx, end_idx))     # Keep a list of references for clip indices

        # Force retrieve one last clip
        indices = stream.get_clip(forced_retrieve=True)
        start_idx, end_idx = indices[0], indices[-1]
        clip = feat[start_idx:end_idx+1]
        clips.append(clip)
        frame_ranges.append((start_idx, end_idx))
                
        # Collect labels into pairs
        annots_video = annotations[video_str]
        timestamps, labels = annots_video[config.ANNOT_TO_USE]  # NOTE: Use the specified annotations only
        pairs = []
        for i in range(len(timestamps)):
            # Get the current segment
            timestamp = timestamps[i]

            # Process the label accordingly by types of experiment
            target = process_label(labels[i], config)

            print('[{}, {}] {}'.format(timestamp[0], timestamp[-1], target))
            pairs.append([timestamp, target])

        # Assign annotation for each clip, determined by the last frame_no of the clip
        for i, clip in enumerate(clips):
            end_frame_no = frame_ranges[i][-1]
            for pair in pairs:
                if end_frame_no in pair[0]:
                    # Save the (clip, target) pair to local hard drive now
                    clip_name = '{}_{}_{}'.format(video_name, frame_ranges[i][0], frame_ranges[i][1])
                    print('-'*30)
                    print('Clip Name:', clip_name)
                    print('Clip Range: "{}-{}"\nTarget: "{}"'.format(frame_ranges[i][0], 
                                                                     frame_ranges[i][1],
                                                                     pair[1]))
                    # Save into clips
                    feature_fpath = os.path.join(output_path, clip_name+'_clip.npy')
                    np.save(feature_fpath, clip)

                    # Save caption
                    target_fpath = os.path.join(output_path, clip_name+'_target.npy')
                    target = np.array(pair[1])
                    np.save(target_fpath, target)

                    print('Shape: {}\nClip saved to {}\nTarget saved to {}.'.format(clip.shape, 
                                                                                   feature_fpath,
                                                                                   target_fpath))
                    break

    return

if __name__ == '__main__':
    config = RSRGBDConfig()
    config.display()

    # Collect all extracted features
    FEAT_PATH = os.path.join(config.DATASET_PATH, config.BACKBONE)
    features = rs_rgbd.collect_features(config.TASKS, FEAT_PATH)
    annotations = rs_rgbd.load_semantic_annotations(config.TASKS, config.DATASET_PATH)

    # Sample by video stream
    generate_clips(features, annotations, config)