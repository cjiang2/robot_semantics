"""
Robot Semantics
Generic functions for visualization purpose.
"""
import os

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.transform
import graphviz

# ------------------------------------------------------------
# Functions for graphviz visualization
# ------------------------------------------------------------

def convert_to_graphviz(graph, name='KG'):
    """Helper function to convert edge graph into graphviz.Digraph.
    """
    e = graphviz.Digraph(name)
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