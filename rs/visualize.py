"""
Robot Semantics
Generic functions for visualization purpose.
"""
import os
import sys
import random
import colorsys

import graphviz
import cv2
import skimage.transform
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
import matplotlib.pyplot as plt

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

def normalize_alpha(alpha):
    """Normalize attention weights.
    """
    ma, mi = alpha.max(), alpha.min()
    alpha = (alpha-mi) / (ma-mi)
    return alpha

def upsample_attention(alpha,
                       image_shape,
                       method='pyramid',
                       input_size=224,
                       base_size=7,
                       sigma=8):
    """Use skimage pyramid to upsample attention map.
    """
    if method == 'pyramid':
        upscale = int(input_size / base_size)
        alpha = skimage.transform.pyramid_expand(alpha, upscale=upscale, sigma=sigma)
    # If not using pyramid, directly upsample into image shape
    alpha = skimage.transform.resize(alpha, image_shape)

    # Normalize alpha
    alpha = normalize_alpha(alpha)

    # Get heatmap and binary mask
    mask = np.uint8(255*alpha)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    return mask, heatmap

def heatmap_overlay(img, 
                    heatmap):
    """Overlay heatmap on top of an image.
    """
    # Overlay
    vis = np.float32(heatmap) + np.float32(img)
    vis = vis / np.max(vis)

    # Scale-up visualization
    vis = np.uint8(255.*vis)
    return vis

def display_video(frames,
                  texts,
                  alphas,
                  input_size=224,
                  base_size=7,
                  sigma=8,
                  show_plot=True,
                  save_path=None,
                  upsample_method='pyramid',
                  to_rgb=True):
    """Visualizes region attentions given corresponding attention weights and
    paths to frame.
    For offline demo only.
    """
    plots = []
    for t in range(len(frames)):
        frame, alpha = frames[t], alphas[t]

        # Two methods to visualize attention map
        _, heatmap = upsample_attention(alpha, 
                                        frame.shape[:2], 
                                        method=upsample_method,
                                        input_size=input_size, 
                                        base_size=base_size,
                                        sigma=sigma)
        vis = heatmap_overlay(frame, heatmap)
        if to_rgb:
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plots.append(vis)

    # Show everything
    if show_plot:
        for t, vis in enumerate(plots):
            if save_path is not None:
                if to_rgb:
                    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path, '{}.png'.format(t)), vis)
            plt.imshow(vis)
            plt.title(texts[t])
            plt.pause(0.001)
            plt.clf()
            
    return plots