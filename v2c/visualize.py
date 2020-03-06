"""
Robot Semantics
Generic functions for visualization purpose.
"""
import os

import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.transform
import graphviz

# ------------------------------------------------------------
# Functions for graphviz visualization
# ------------------------------------------------------------

def convert_to_graphviz(graph, name='KG', engine='fdp'):
    """Helper function to convert edge graph into graphviz.Digraph.
    """
    e = graphviz.Digraph(name, engine=engine)
    e.attr('node', shape='box')
    for edge in graph:
        e.attr('node', shape='box')
        e.node(edge[0])
        e.node(edge[2])
        e.edge(edge[0], edge[2], label=edge[1])
    return e


# ------------------------------------------------------------
# Functions for region attention visualization
# ------------------------------------------------------------

def visualize_region_atts(frames_path, 
                          alphas, 
                          smooth=True):
    """Visualizes region attentions given corresponding attention weights and
    paths to frame.
    """
    for t in range(len(frames_path)):
        frame = Image.open(frames_path[t])
        frame = np.asarray(frame.resize([7 * 24, 7 * 24], Image.BILINEAR))
        
        plt.subplot(np.ceil(len(frames_path) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (str(t)), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(frame)
        alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha.reshape(7,7), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha.reshape(7,7), [7 * 24, 7 * 24])
        plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

def visualize_region_atts_v2(frames_path,
                             alphas,
                             smooth=True, 
                             base_size=7,
                             upscale=64,
                             show_plot=True):
    """Visualizes region attentions given corresponding attention weights and
    paths to frame.
    Version 2: Support loading for all frames for a video. Use OpenCV color map for 
    visualization.
    """
    plots = []
    for t in range(len(frames_path)):
        frame = cv2.imread(frames_path[t])
        alpha = alphas[t, :].reshape(7, 7)

        # Resize frame to match alpha size
        frame = cv2.resize(frame, (base_size*upscale, base_size*upscale))

        # Process alpha into a binary mask
        alpha = alpha / alpha.max()
        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha, upscale=upscale, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha, [base_size*upscale, base_size*upscale])

        # Convert to mask, then to heatmap
        mask = np.uint8(255*alpha)
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # Overlay
        vis = np.float32(heatmap) + np.float32(frame)
        vis = vis / np.max(vis)

        # Scale-up visualization
        vis = np.uint8(255.*vis)

        plots.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    # Show everything
    if show_plot:
        for t, vis in enumerate(plots):
            cv2.imwrite('save/{}.png'.format(t), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            plt.imshow(vis)
            plt.pause(0.001)
            plt.clf()
            
    return plots