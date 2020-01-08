import os
import sys
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import v2c.utils as utils

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
        caption_name = path.split(os.sep)[-1][:-4]
        f = open(path, 'r')
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            start_frame, end_frame = lines[i].strip().split(', ')
            start_frame, end_frame = int(start_frame), int(end_frame)
            caption = lines[i+1].strip().split(', ')
            #print(start_frame, end_frame, caption)
            #timestamps.append([start_frame, end_frame])
            timestamps.append(get_frames_no([start_frame, end_frame]))
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


# ----------------------------------------
# Functions for RS-RGBD Manual Feature Extraction
# ----------------------------------------

def generate_clips(dataset_path,
                   folder,
                   window_size,
                   annotations_type=['lefthand', 'righthand']):
    """Generate clips for training purposes through simulating video streamline queue.
    """
    def get_full_paths(frames_no,
                       folder,
                       dataset_path,
                       video_name):
        frames_path = []
        for frame_no in frames_no:
            frames_path.append(os.path.join(dataset_path, folder, video_name, video_name, '{}_rgb.png'.format(frame_no)))
        return frames_path

    all_clips, all_captions = [], []
    # Get all timestamps and annotation captions 1st
    annotations = load_annotations(dataset_path, folder)
    videos = load_videos(dataset_path, folder)

    # Generate clips
    for video_name in annotations.keys():
        annotations_by_video = annotations[video_name]
        num_frames = len(videos[video_name])
        print(video_name, num_frames)

        # Host a streamline queue
        stream_queue = utils.StreamlineVideoQueue(window_size=window_size)

        # Generate frame indices for all possible clips
        clips = []
        frame_ranges = []
        for i in range(num_frames):
            stream_queue.update(i)
            clip = stream_queue.retrieve_clip()
            if clip is not None:
                clips.append(clip)
                frame_ranges.append((i - window_size + 1, i))

        # Extract the main annotated manipulator captions, ignore the other manipulator with empty('none') annotations
        main_annotations = []
        for annotation_type in annotations_type:
            if len(annotations_by_video[annotation_type][0]) > 1:
                timestamps, captions = annotations_by_video[annotation_type]
                for i in range(len(timestamps)):
                    # Get the current segment
                    timestamp, caption = timestamps[i], captions[i][0]      # NOTE: Choose the first caption (Highest-level action)
                    print('[{}, {}] {}'.format(timestamp[0], timestamp[-1], caption))
                    main_annotations.append([timestamp, caption])

        # Search for annotation, determined by the last frame_no of the clip
        for i, clip in enumerate(clips):
            end_frame_no = clip[-1]
            for main_annotation in main_annotations:
                #print(main_annotation)
                if end_frame_no in main_annotation[0]:
                    # Keep note of the frame_range where the clip belongs
                    all_clips.append({'{}_{}_{}'.format(video_name, frame_ranges[i][0], frame_ranges[i][1]): get_full_paths(clip, folder, dataset_path, video_name)})
                    all_captions.append(main_annotation[1])
                    break
        print()
        
    return all_clips, all_captions

def override_caption(caption):
    """NOTE: NOT DESIRABLE.
    Manually override some specific annotation tokens.
    """
    caption = caption.replace('lefthand', 'humanhand')
    caption = caption.replace('righthand', 'humanhand')
    #caption = caption.replace('_', ' ')
    return caption

def process_caption(captions,
                    config,
                    vocab=None):
    """Helper function to process captions into sequences.
    """
    # Add start and end tokens
    captions = ['{} {} {}'.format(config.START_WORD, override_caption(caption), config.END_WORD) for caption in captions]

    # Build vocabulary
    if vocab is None:
        vocab = utils.build_vocab(captions, 
                                  frequency=config.FREQUENCY,
                                  start_word=config.START_WORD,
                                  end_word=config.END_WORD,
                                  unk_word=config.UNK_WORD)
    # Reset vocab_size
    config.VOCAB_SIZE = len(vocab)

    maxlen = utils.get_maxlen(captions)
    #print('Maximum length:', maxlen)
    if (config.MAXLEN is None) or (config.MAXLEN < maxlen):
        config.MAXLEN = maxlen
    #print('Current Maximum length settings:', config.MAXLEN)
            
    # Process text tokens
    targets = utils.texts_to_sequences(captions, vocab)
    targets = utils.pad_sequences(targets, config.MAXLEN, padding='post')
    targets = targets.astype(np.int64)

    return targets, vocab, config

def parse_clip_paths_and_captions(config,
                                  vocab=None,
                                  video_name=None):
    """Helper function to parse paths to clips, load and process captions and 
    return (clip, target) pairs.
    Added option to load specified clips for one video if needed.
    """
    # Load paths to caption and clips
    # TODO: Only accept one folder from the config.SETTINGS
    feature_path = os.path.join(config.DATASET_PATH, 
                                list(config.BACKBONE.keys())[0], 
                                config.SETTINGS[0])
    if video_name is not None:
        clips = sorted(glob.glob(os.path.join(feature_path, '{}_*_clip.npy'.format(video_name))))        
    else:
        clips = sorted(glob.glob(os.path.join(feature_path, '*_clip.npy')))

    captions = ['{}'.format(str(np.load(x.replace('_clip', '_caption')))) for x in clips]

    # Sentences to Sequences
    targets, vocab, config = process_caption(captions, 
                                             config=config, 
                                             vocab=vocab)
    return clips, targets, vocab, config


# ----------------------------------------
# Functions for PyTorch Dataset object, transformers
# ----------------------------------------

# Parameter settings, from PyTorch deafult
TARGET_IMAGE_SIZE = (224, 224)
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]

# Transformers
transforms_data = transforms.Compose([transforms.Resize(TARGET_IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
                                     ])

# (frames, target) pytorch dataset object
class Frames2ClipDataset(data.Dataset):
    """Create an instance of RS-RGBD dataset with (frames, clip_names, captions).
    """
    def __init__(self, 
                 clips,
                 captions,
                 transform=None):
        self.clips, self.captions = clips, captions     # Load annotations
        self.transform = transform

    def parse_clip(self, 
                   clip):
        """Helper function to parse images {clip_name: imgs_path} into a clip. 
        """
        imgs = []
        clip_name = list(clip.keys())[0]
        imgs_path = clip[clip_name]
        for img_path in imgs_path:
            img = self._imread(img_path)
            imgs.append(img)
        return torch.stack(imgs, dim=0), clip_name

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        imgs, clip_name = self.parse_clip(self.clips[idx])
        caption = self.captions[idx]
        return imgs, caption, clip_name

# (Clip, target) pytorch dataset object
class ClipDataset(data.Dataset):
    """Create an instance of RS-RGBD dataset with pre-processed (clip_path, target).
    """
    def __init__(self, 
                 clips,
                 targets):
        self.clips, self.targets = clips, targets    # Load all (clip, target) pairs

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        Xv = np.load(self.clips[idx])
        clip_name = self.clips[idx].split(os.sep)[-1]
        S = self.targets[idx]
        return Xv, S, clip_name

# Frame-only pytorch dataset object
class FrameDataset(data.Dataset):
    """Create an instance of RS-RGBD dataset with all the frames only.
    """
    def __init__(self, 
                 frames,
                 transform=None):
        self.frames = frames    # Load all frame images
        self.transform = transform

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self._imread(self.frames[idx])
        return img