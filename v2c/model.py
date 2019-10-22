import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from v2c.backbone import resnet

# ----------------------------------------
# Functions for Video Feature Extraction
# ----------------------------------------

class CNNWrapper(nn.Module):
    """Wrapper module to extract features from image using
    pre-trained CNN.
    """
    def __init__(self,
                 backbone,
                 checkpoint_path):
        super(CNNWrapper, self).__init__()
        self.backbone = backbone
        self.model = self.init_backbone(checkpoint_path)

    def forward(self,
                x):
        with torch.no_grad():
            x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return x

    def init_backbone(self,
                      checkpoint_path):
        """Helper to initialize a pretrained pytorch model.
        """
        if self.backbone == 'resnet50':
            model = resnet.resnet50(pretrained=False)   # Use Caffe ResNet instead
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet101':
            model = resnet.resnet101(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet152':
            model = resnet.resnet152(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'vgg16':
            model = models.vgg16(pretrained=True)

        elif self.backbone == 'vgg19':
            model = models.vgg19(pretrained=True)

        # Remove the last classifier layer
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        
        return model

# ----------------------------------------
# Functions for V2CNet
# ----------------------------------------

class VideoEncoder(nn.Module):
    """Module to encode pre-extracted features coming from 
    pre-trained CNN.
    """
    def __init__(self,
                 in_size,
                 units):
        super(VideoEncoder, self).__init__()
        self.linear = nn.Linear(in_size, units)
        self.lstm = nn.LSTM(units, units, batch_first=True)
        self.reset_parameters()

    def forward(self, 
                Xv):
        # Phase 1: Encoding Stage
        # Encode video features with one dense layer and lstm
        # State of this lstm to be used for lstm2 language generator
        Xv = self.linear(Xv)
        #print('linear:', Xv.shape)
        Xv = F.relu(Xv)

        Xv, (hi, ci) = self.lstm(Xv)
        Xv = Xv[:,-1,:]     # Only need the last timestep
        hi, ci = hi[0,:,:], ci[0,:,:]
        #print('lstm:', Xv.shape, 'hi:', hi.shape, 'ci:', ci.shape)
        return Xv, (hi, ci)

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


class CommandDecoder(nn.Module):
    """Module to decode features and generate word for captions
    using RNN.
    """
    def __init__(self,
                 units,
                 vocab_size,
                 embed_dim,
                 bias_vector=None):
        super(CommandDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, units)
        self.logits = nn.Linear(units, vocab_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters(bias_vector)

    def forward(self, 
                Xs, 
                states):
        # Phase 2: Decoding Stage
        # Given the previous word token, generate next caption word using lstm2
        # Sequence processing and generating
        #print('sentence decoding stage:')
        #print('Xs:', Xs.shape)
        Xs = self.embed(Xs)
        #print('embed:', Xs.shape)

        hi, ci = self.lstm_cell(Xs, states)
        #print('out:', hi.shape, 'hi:', states[0].shape, 'ci:', states[1].shape)

        x = self.logits(hi)
        #print('logits:', x.shape)
        x = self.softmax(x)
        #print('softmax:', x.shape)
        return x, (hi, ci)

    def init_hidden(self, 
                    batch_size):
        """Initialize a zero state for LSTM.
        """
        h0 = torch.zeros(batch_size, self.units)
        c0 = torch.zeros(batch_size, self.units)
        return (h0, c0)

    def reset_parameters(self,
                         bias_vector):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        nn.init.uniform_(self.embed.weight.data, -0.05, 0.05)
        if bias_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_vector).float()


class CommandLoss(nn.Module):
    """Calculate Cross-entropy loss per word.
    """
    def __init__(self, 
                 ignore_index=0):
        super(CommandLoss, self).__init__()
        self.cross_entropy = nn.NLLLoss(reduction='sum', 
                                        ignore_index=ignore_index)

    def forward(self, 
                input, 
                target):
        return self.cross_entropy(input, target)


class Video2Command():
    """Train/Eval inference class for V2C model.
    """
    def __init__(self,
                 config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def build(self,
              bias_vector=None):
        # Initialize Encode & Decode models here
        self.video_encoder = VideoEncoder(in_size=list(self.config.BACKBONE.values())[0],
                                          units=self.config.UNITS)
        self.command_decoder = CommandDecoder(units=self.config.UNITS,
                                              vocab_size=self.config.VOCAB_SIZE,
                                              embed_dim=self.config.EMBED_SIZE,
                                              bias_vector=bias_vector)
        self.video_encoder.to(self.device)
        self.command_decoder.to(self.device)
    
        # Loss function
        self.loss_objective = CommandLoss()
        self.loss_objective.to(self.device)

        # Setup parameters and optimizer
        self.params = list(self.video_encoder.parameters()) + \
                      list(self.command_decoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, 
                                          lr=self.config.LEARNING_RATE)

        # Save configuration
        # Safely create checkpoint dir if non-exist
        if not os.path.exists(os.path.join(self.config.CHECKPOINT_PATH, 'saved')):
            os.makedirs(os.path.join(self.config.CHECKPOINT_PATH, 'saved'))

    def train(self, 
              train_loader):
        """Train the Video2Command model.
        """
        def train_step(Xv, S):
            """One train step.
            """
            loss = 0.0
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Video feature extraction 1st
            Xv, states = self.video_encoder(Xv)

            # Calculate mask against zero-padding
            S_mask = S != 0

            # Teacher-Forcing for command decoder
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:,timestep]
                probs, states = self.command_decoder(Xs, states)
                # Calculate loss per word
                loss += self.loss_objective(probs, S[:,timestep+1])
            loss = loss / S_mask.sum()     # Loss per word

            # Gradient backward
            loss.backward()
            self.optimizer.step()
            return loss

        # Training epochs
        self.video_encoder.train()
        self.command_decoder.train()
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            for i, (Xv, S, clip_names) in enumerate(train_loader):
                # Mini-batch
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
        return

    def evaluate(self,
                 test_loader,
                 vocab):
        """Run the evaluation pipeline over the test dataset.
        """
        assert self.config.MODE == 'test'
        y_pred, y_true = [], []
        # Evaluation over the entire test dataset
        for i, (Xv, S_true, clip_names) in enumerate(test_loader):
            # Mini-batch
            Xv, S_true = Xv.to(self.device), S_true.to(self.device)
            S_pred = self.predict(Xv, vocab)
            y_pred.append(S_pred)
            y_true.append(S_true)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return y_pred.cpu().numpy(), y_true.cpu().numpy()

    def predict(self, 
                Xv,
                vocab):
        """Run the prediction pipeline given one sample.
        """
        self.video_encoder.eval()
        self.command_decoder.eval()

        with torch.no_grad():
            # Initialize S with '<sos>'
            S = torch.zeros((Xv.shape[0], self.config.MAXLEN), dtype=torch.long)
            S[:,0] = vocab('<sos>')
            S = S.to(self.device)

            # Start v2c prediction pipeline
            Xv, states = self.video_encoder(Xv)

            #states = self.command_decoder.reset_states(Xv.shape[0])
            #_, states = self.command_decoder(None, states, Xv=Xv)   # Encode video features 1st
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:,timestep]
                probs, states = self.command_decoder(Xs, states)
                preds = torch.argmax(probs, dim=1)    # Collect prediction
                S[:,timestep+1] = preds
        return S

    def save_weights(self, 
                     epoch):
        """Save the current weights and record current training info 
        into tensorboard.
        """
        # Save the current checkpoint
        torch.save({
                    'VideoEncoder_state_dict': self.video_encoder.state_dict(),
                    'CommandDecoder_state_dict': self.command_decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(self.config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    def load_weights(self,
                     save_path):
        """Load pre-trained weights by path.
        """
        print('Loading...')
        checkpoint = torch.load(save_path)
        self.video_encoder.load_state_dict(checkpoint['VideoEncoder_state_dict'])
        self.command_decoder.load_state_dict(checkpoint['CommandDecoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')