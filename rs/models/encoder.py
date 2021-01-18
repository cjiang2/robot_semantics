import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class VideoEncoder(nn.Module):
    """Module to encode pre-extracted features coming from 
    pre-trained CNN.
    """
    def __init__(self,
                 in_size,
                 hidden_size):
        super(VideoEncoder, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        self.proj = nn.Conv2d(in_size, hidden_size, 1)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

        self.reset_parameters()

    def forward(self,
                Xv):
        # Encode video features with one dense layer and lstm
        # State of this lstm to be used for lstm2 language generator
        # Xv: (batch_size, num_clips, in_size, h, w)

        # Encode video feature using LSTM
        hi, ci = None, None
        for timestep in range(Xv.shape[1]):
            xv = Xv[:,timestep,:,:,:]

            # LSTM encoding
            hi, ci = self.step(xv, hi, ci)

        return hi, (hi, ci)

    def step(self, 
             xv, 
             hi,
             ci):
        # Initialize hidden state for LSTM if needed
        if hi is None or ci is None:
            hi, ci = self.init_hidden(xv)

        # Use 1x1 Conv for Projection
        x = self.proj(xv)

        # Global average pooling over spatial resolution
        x = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=-1)

        # Encode context vector using LSTM
        hi, ci = self.lstm_cell(x, (hi, ci))

        return hi, ci

    def init_hidden(self,
                    input):
        """Return a initial state for LSTM.
        """
        state_size = [int(input.shape[0]), self.hidden_size]
        h0 = torch.zeros(state_size, 
                         device=input.device, dtype=input.dtype)
        c0 = torch.zeros(state_size, 
                         device=input.device, dtype=input.dtype)
        return (h0, c0)

    def reset_parameters(self):
        for i in range(4):
            nn.init.orthogonal_(self.lstm_cell.weight_hh.data[self.hidden_size*i:self.hidden_size*(i+1)])
        nn.init.zeros_(self.lstm_cell.bias_ih)
        nn.init.zeros_(self.lstm_cell.bias_hh)


if __name__ == '__main__':
    # ----------
    # Test for encoder
    batch_size = 16
    units = 512
    hidden_size = 256
    h, w = 7, 7

    encoder = VideoEncoder(units, hidden_size)
    x = torch.ones(batch_size, 30, units, h, w).float()
    encoder(x)

    for n, p in encoder.named_parameters():
        print(n, p.data.dtype, p.shape)