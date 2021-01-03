import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from rs.models import resnet

# ----------------------------------------
# Functions for CNN Video Feature Extraction
# ----------------------------------------

BACKBONE_TO_IN_SIZE = {'resnet50': 2048,
                       'resnet34': 512,
                       'resnet18': 512}

class CNNExtractor(nn.Module):
    """Wrapper module to extract features from image using
    pre-trained CNN.
    """
    def __init__(self,
                 backbone,
                 weights_path,
                 save_grad=True):
        super(CNNExtractor, self).__init__()
        self.backbone = backbone
        self.save_grad = save_grad
        self.model = self.init_backbone(weights_path)

        # Grad-CAM specifics
        if self.save_grad:
            self.gradients = []
            self.activations = []

    def save_gradient(self, 
                      grad):
        """Gradient hook.
        """
        self.gradients.append(grad)

    def reset_cam(self):
        self.gradients = []
        self.activations = []

    def forward(self,
                x):
        x = self.model(x)

        # Save gradients and activations for Grad-CAM
        if self.save_grad:
            x.register_hook(self.save_gradient)
            self.activations.append(x)
        
        return x

    def init_backbone(self,
                      weights_path):
        """Helper to initialize a pretrained pytorch model.
        """
        # ResNet
        # --------------------
        if self.backbone == 'resnet18':
            model = models.resnet18(pretrained=True)   
            modules = list(model.children())[:-2]

        elif self.backbone == 'resnet34':
            model = models.resnet34(pretrained=True)
            modules = list(model.children())[:-2]
        
        # Use Caffe ResNet50 instead of the default one
        elif self.backbone == 'resnet50':
            model = resnet.resnet50(pretrained=False)
            model.load_state_dict(torch.load(weights_path))
            modules = list(model.children())[:-2]

        # Repack model to remove the last classifier layer
        model = nn.Sequential(*modules)
        
        return model