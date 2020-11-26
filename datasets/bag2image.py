import argparse
import os
import glob
import numpy as np
import cv2
from PIL import Image
import pyrealsense2 as rs

def streaming_from_file(filepath):
    """Helper function to read rosbag file from realsense.
    """
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(filepath, repeat_playback=False)
    return pipe, cfg

def post_processing(frame,
                    enable_spatial=True,
                    enable_temporal=True,
                    enable_hole=True,
                    spatial_params=[(rs.option.filter_magnitude, 5), 
                                    (rs.option.filter_smooth_alpha, 1),
                                    (rs.option.filter_smooth_delta, 50),
                                    (rs.option.holes_fill, 3)],
                    temporal_params=[],
                    hole_params=[]):
    """Filters to cleanup depth maps.
    """
    # Filters and settings
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # Depth to disparity before spatial and temporal filters
    frame = depth_to_disparity.process(frame)

    # Spatial filter
    if enable_spatial:
        # Settings
        spatial = rs.spatial_filter()
        for spatial_param in spatial_params:
            spatial.set_option(spatial_param[0], spatial_param[1])

        # Apply on frame
        frame = spatial.process(frame)

    # Temporal filter
    if enable_temporal:
        temporal = rs.temporal_filter()
        for temporal_param in temporal_params:
            temporal.set_option(temporal_param[0], temporal_param[1])
        frame = temporal.process(frame)

    # Back to depth
    frame = disparity_to_depth.process(frame)

    # Hole filling
    if enable_hole:
        hole_filling = rs.hole_filling_filter()
        for hole_param in hole_params:
            hole_filling.set_option(hole_param[0], hole_param[1])
        frame = hole_filling.process(frame)

    return frame

def extract_frames(pipe, 
                   cfg, 
                   save_path,
                   resize=None,
                   post_processing=False,
                   save_colorize=True,
                   save_pc=False,
                   visualize=True):
    """Helper function to align and extract rgb-d image from bagfile.
       Check more saving options in arguments.
    Args:
        pipe: pyrealsense2 pipeline.
        cfg: pyrealsense2 pipeline configuration.
        save_path: Path to save the extracted frames.
        save_colorize: Save colorized depth maps visualization.
        save_pc: Save point cloud data.
        visualize: Visualize the video frames during saving.
    """
    # Configurations
    if save_colorize:
        colorizer = rs.colorizer()
    if save_pc:
        pc = rs.pointcloud()
        points = rs.points()
    # Save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Start the pipe
    i = 0
    profile = pipe.start(cfg)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False) # Make sure this is False or frames get dropped
    while True:
        try:
            # Wait for a conherent pairs of frames: (rgb, depth)
            pairs = pipe.wait_for_frames()

            # Align depth image to rgb image first
            align = rs.align(rs.stream.color)
            pairs = align.process(pairs)

            color_frame = pairs.get_color_frame()
            depth_frame = pairs.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # Post-processing
            if post_processing:
                depth_frame = post_processing(depth_frame)

            # Get rgb-d images
            color_img = np.asanyarray(color_frame.get_data())
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            depth_img = np.asanyarray(depth_frame.get_data())
            print('Frame {}, Depth Image {}, Color Image {}'.format(i+1, depth_img.shape, color_img.shape))
            
            # If resize image
            if resize:
                height, width = resize, resize  # Always assume square
                # Use PIL to resize image
                img_temp = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                img_temp = Image.fromarray(img_temp)
                img_temp = img_temp.resize((width, height))
                color_img = cv2.cvtColor(np.array(img_temp), cv2.COLOR_RGB2BGR)

                # Use nearest neighbor interpolation to resize depth image
                depth_img = cv2.resize(depth_img, (height, width), interpolation=cv2.INTER_NEAREST)

            # Save as loseless formats
            cv2.imwrite(os.path.join(save_path, '{}_rgb.png'.format(i)), color_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            np.save(os.path.join(save_path, '{}_depth.npy'.format(i)), depth_img)
            
            if save_colorize:
                # Save colorized depth map
                depth_img_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                cv2.imwrite(os.path.join(save_path, '{}_depth_colorized.jpg'.format(i)), depth_img_colorized)   # No need for lossless here
            
            if save_pc:
                # NOTE: Point cloud calculation takes longer time.
                #pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                points.export_to_ply(os.path.join(save_path, '{}_pc.ply'.format(i)), color_frame)
            
            i += 1

            if visualize:
                # Stack both images horizontally
                images = np.vstack((color_img, depth_img_colorized))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
                cv2.imshow('RealSense', images)
                cv2.waitKey(1)
            
        except Exception as e:
            print(e)
            break

    # Clean pipeline
    pipe.stop()
    print('{} frames saved in total.'.format(i))

    return

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Convert bagfile to images.')
    parser.add_argument('task', help='Path to task folder.')
    parser.add_argument('-resize', '--resize', type=int, nargs=1, default=False,
                         help='Resize image to specified shape (int, int).')
    parser.add_argument('-post_processing','--post_processing', nargs='?', type=bool, default=False, 
                        help='Enable depth post processing.')
    parser.add_argument('-save_colorize','--save_colorize', nargs='?', type=bool, default=False, 
                        help='Save colorized depth map.')
    parser.add_argument('-save_pc','--save_pc', nargs='?', type=bool, default=False, 
                        help='Save point cloud data.')
    parser.add_argument('-visualize','--visualize', nargs='?', type=bool, default=False, 
                        help='Visualize while saving.')
    args = parser.parse_args()

    videos = glob.glob(os.path.join(args.task, '*'))
    for video in videos:
        video_name = video.split(os.sep)[-1]
        bagfile = os.path.join(os.path.curdir, args.task, video_name, video_name+'.bag')
        save_path = os.path.join(os.path.curdir, args.task, video_name, video_name)
        print('Extracting {} and saving to {}...'.format(bagfile, save_path))

        # Save things here
        pipe, cfg = streaming_from_file(bagfile)
        extract_frames(pipe, cfg, 
                       resize=args.resize,
                       save_path=save_path, 
                       post_processing=args.post_processing,
                       save_colorize=args.save_colorize, 
                       save_pc=args.save_pc,
                       visualize=args.visualize)