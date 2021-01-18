"""
RS Concepts
Model codebase.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from rs.models.extractor import *
from rs.models.encoder import *
from rs.models.decoder import *

class NLLLoss(nn.Module):
    """Calculate Cross-entropy loss per word.
    """
    def __init__(self):
        super(NLLLoss, self).__init__()
        self.cross_entropy = nn.NLLLoss(reduction='none')

    def forward(self, 
                input, 
                target,
                mask):
        loss = self.cross_entropy(input, target)
        loss_m = loss * mask
        return loss_m.sum()

class Video2Lang():
    """Train/Eval inference class for V2L model.
    """
    def __init__(self,
                 config,
                 vocab):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vocab = vocab

    def build(self,
              bias_vector=None):
        # Initialize Encode & Decode models here
        self.video_encoder = VideoEncoder(in_size=BACKBONE_TO_IN_SIZE[self.config.BACKBONE],
                                          hidden_size=self.config.UNITS)
        self.lang_decoder = LangDecoder(units=self.config.UNITS,
                                        vocab_size=self.config.VOCAB_SIZE,
                                        embed_dim=self.config.EMBED_DIM,
                                        bias_vector=bias_vector)
        self.video_encoder.to(self.device)
        self.lang_decoder.to(self.device)
    
        # Loss function
        self.criterion = NLLLoss()
        self.criterion.to(self.device)

        # Setup parameters and optimizer
        self.params = list(self.video_encoder.parameters()) + \
                      list(self.lang_decoder.parameters())
        self.optimizer = torch.optim.AdamW(self.params, 
                                           lr=self.config.LEARNING_RATE,
                                           weight_decay=self.config.WEIGHT_DECAY)

        # Online mode, initialize CNN feature extractor here as well
        if self.config.LOAD_CNN:
            self.cnn_extractor = CNNExtractor(backbone=self.config.BACKBONE,
                                              weights_path=self.config.WEIGHTS_PATH,
                                              save_grad=False)
            self.cnn_extractor.eval()
            self.cnn_extractor.to(self.device)

        # Save configuration
        # Safely create checkpoint dir if non-exist
        if not os.path.exists(os.path.join(self.config.CHECKPOINT_PATH, 'saved')):
            os.makedirs(os.path.join(self.config.CHECKPOINT_PATH, 'saved'))

    def train(self,
              train_loader):
        """Train the Video2Lang model.
        """
        def train_step(X, S):
            """One train step.
            """
            loss = 0.0
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # --------------------
            # Forward pass
            # Calculate mask against zero-padding
            S_mask = S != 0

            # Decoding loop
            Xs = S[:,0]     # First word is always START_WORD

            # Video encoding
            _, states = self.video_encoder(X)

            # Language Decoding
            for timestep in range(self.config.MAXLEN - 1):
                probs, states = self.lang_decoder(Xs, states)
                
                # Calculate loss per word
                loss += self.criterion(probs, S[:,timestep+1], S_mask[:,timestep+1])

                # Teacher-Forcing for command decoder
                Xs = S[:,timestep+1]

            loss = loss / S_mask.sum()     # Average loss per word
            # Gradient backward
            loss.backward()
            if self.config.CLIP_NORM:
                nn.utils.clip_grad_norm_(self.video_encoder.parameters(), self.config.CLIP_NORM)
                nn.utils.clip_grad_norm_(self.lang_decoder.parameters(), self.config.CLIP_NORM)
            self.optimizer.step()
            return loss

        # Training epochs
        self.video_encoder.train()
        self.lang_decoder.train()
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            for i, batch in enumerate(train_loader):
                # Mini-batch
                Xv, S, clip_names = batch[0], batch[1], batch[2]
                Xv, S = Xv.to(self.device), S.to(self.device)
                # Train step
                loss = train_step(Xv, S)
                total_loss += loss
                # Display
                if i % self.config.DISPLAY_EVERY == 0:
                    print('Epoch {}, Iter {}, Loss {:.6f}'.format(epoch+1, 
                                                                  i,
                                                                  loss))
            # End of epoch, save weights
            print('Total loss for epoch {}: {:.6f}'.format(epoch+1, total_loss / (i + 1)))
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_weights(epoch + 1)

            # If LR decay is needed
            if epoch + 1 in self.config.LR_DECAY_EVERY:
                self.config.LEARNING_RATE = self.config.LEARNING_RATE / 10
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.LEARNING_RATE
                print('Decaying LR, current LR:', self.config.LEARNING_RATE)

        return

    def predict(self, 
                X):
        """Run the prediction pipeline given one sample.
        """
        self.video_encoder.eval()
        self.lang_decoder.eval()
        
        with torch.no_grad():
            # Initialize S with '<sos>'
            batch_size = 1 if self.config.LOAD_CNN else X.shape[0]
            S = torch.zeros((batch_size, self.config.MAXLEN), dtype=torch.long)
            S[:,0] = self.vocab.word_to_idx('<sos>')
            S = S.to(self.device)

            # If online mode, perform CNN feature extraction 1st
            if self.config.LOAD_CNN:
                X = self.cnn_extractor(X)
                X = X.unsqueeze(0)

            # Video encoding
            _, states = self.video_encoder(X)

            # Semantics Decoding
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:,timestep]

                probs, states = self.lang_decoder(Xs, states)
                
                # Collect prediction
                preds = torch.argmax(probs, dim=1)
                S[:,timestep+1] = preds

        return S

    def evaluate(self,
                 eval_loader):
        """Run the evaluation pipeline over the eval dataset.
        """
        assert self.config.MODE == 'eval'
        y_pred, y_true, fnames, alphas = [], [], [], []
        # Evaluation over the entire test dataset
        for i, batch in enumerate(eval_loader):
            # Mini-batch
            Xv, S_true, clip_names = batch[0], batch[1], batch[2]

            Xv, S_true = Xv.to(self.device), S_true.to(self.device)
            S_pred = self.predict(Xv)
            y_pred.append(S_pred)
            y_true.append(S_true)
            fnames += clip_names
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return y_pred.cpu().numpy(), y_true.cpu().numpy(), fnames, None

    def save_weights(self, 
                     epoch):
        """Save the current weights and record current training info 
        into tensorboard.
        """
        # Save the current checkpoint
        torch.save({
                    'VideoEncoder_state_dict': self.video_encoder.state_dict(),
                    'LangDecoder_state_dict': self.lang_decoder.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.config.CHECKPOINT_PATH, 'saved', 'v2l_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    def load_weights(self,
                     save_path):
        """Load pre-trained weights by path.
        """
        print('Loading...')
        checkpoint = torch.load(save_path, map_location=self.device)
        self.video_encoder.load_state_dict(checkpoint['VideoEncoder_state_dict'])
        self.lang_decoder.load_state_dict(checkpoint['LangDecoder_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')