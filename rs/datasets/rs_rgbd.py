import os
import glob

# ------------------------------
# RS-RGBD Dataset Integration
# ------------------------------

def load_semantic_annotations(tasks,
                              dataset_path=os.path.join('datasets', 'RS-RGBD')):
    """Parse RS-RGBD semantic annotations.
    tasks: all tasks/folders to be parsed.
    dataset_path: path of root dataset folder.
    Return:
        annotations: a dict. annotations[video_name] = {ANNOTATION_TYPE: [timestamps, texts], ...}
    """
    def read_annotation_file(path):
        """Parse individual annotation text file.
        """
        timestamps = []
        texts = []
        annotation_name = path.split(os.sep)[-1][:-4]
        f = open(path, 'r')
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            start_frame, end_frame = lines[i].strip().split(', ')
            start_frame, end_frame = int(start_frame), int(end_frame)
            text = lines[i+1].strip().split(', ')
            #print(start_frame, end_frame, text)
            #timestamps.append([start_frame, end_frame])
            timestamps.append(get_frames_no([start_frame, end_frame]))
            texts.append(text)
        f.close()
        return annotation_name, timestamps, texts

    def get_frames_no(timestamp):
        frames = []
        for i in range(timestamp[0], timestamp[1] + 1, 1):
            frames.append(i)
        return frames

    # Paths
    # Collection annotation for each video
    annotations = {}
    for task in tasks:
        folder_path = os.path.join(dataset_path, task)
        videos_name = os.listdir(folder_path)

        for video_name in videos_name:
            #print('-'*30)
            annotation_by_video = {}
            video_path = os.path.join(folder_path, video_name)
            # Read each type of text annotation
            annotations_path = glob.glob(os.path.join(video_path, 'semantics', '*.txt'))
            for annotation_path in annotations_path:
                annotation_name, timestamps, texts = read_annotation_file(annotation_path)
                annotation_by_video[annotation_name] = [timestamps, texts]

            annotations[os.path.join(task, video_name)] = annotation_by_video

    return annotations

def load_videos(tasks,
                dataset_path=os.path.join('datasets', 'RS-RGBD'),
                file_type='_rgb.png'):
    """Parse RS-RGBD dataset, only RGB frame images. 
    tasks: all tasks/folders to be parsed.
    dataset_path: path of root dataset folder.
    file_type: format of image file to be parsed.
    Return:
        videos: a dict. annotations[video_name] = [img_path, ...]
    """
    videos = {}
    # Paths
    for task in tasks:
        folder_path = os.path.join(dataset_path, task)
        videos_name = os.listdir(folder_path)

        # Collect videos
        for video_name in videos_name:
            # Safely parse all frame images sorted
            images_path = []
            num_images = len(glob.glob(os.path.join(folder_path, video_name, video_name, '*{}'.format(file_type))))
            for i in range(num_images):
                images_path.append(os.path.join(folder_path, video_name, video_name, '{}{}'.format(i, file_type)))
            videos[os.path.join(task, video_name)] = images_path
    return videos

def collect_features(tasks,
                     feat_path=os.path.join('datasets', 'RS-RGBD', 'wideresnet34')):
    """Parse the paths of extracted RS-RGBD features. 
    tasks: all tasks/folders to be parsed.
    feat_path: path of root feature folder.
    Return:
        videos: a dict. annotations[video_name] = feat_fpath
    """
    features = {}
    # Paths
    for task in tasks:
        folder_path = os.path.join(feat_path, task)
        feats_fpath = sorted(glob.glob(os.path.join(folder_path, '*.npy')))
        for feat_fpath in feats_fpath:
            video_name = feat_fpath.split(os.sep)[-1][:-4]
            video_str = '{}{}{}'.format(task, os.sep, video_name)
            features[video_str] = feat_fpath
    return features