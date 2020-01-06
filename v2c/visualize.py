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
# Functions for knowledge graph visualization
# ------------------------------------------------------------

def visualize_graph(kg):
    """Helper function to construct graphviz.Digraph visualizing 
    knowledge graph.
    """
    e = graphviz.Digraph('ER', filename='er.gv')
    e.attr('node', shape='box')
    for graph in kg:
        e.attr('node', shape='box')
        e.node(graph[0].name)
        e.node(graph[1].name)
        e.edge(graph[0].name, graph[1].name, label=graph[2].replace(',', ', '))
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