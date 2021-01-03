import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

# ----------------------------------------
# Additive Attention
# ----------------------------------------

class BahdanauAttention(nn.Module):
    """Bahdanau Additive Attention.
    """
    def __init__(self,
                 in_size,
                 hidden_size,
                 units,
                 bias=True,
                 save_grad=False):
        super(BahdanauAttention, self).__init__()
        self.save_grad = save_grad
        self.W = nn.Linear(in_size, units, bias=False)
        self.U = nn.Linear(hidden_size, units, bias=False)
        self.V = nn.Linear(units, 1, bias=False)
        self.bias = Parameter(torch.Tensor(units)) if bias else None
        self.reset_parameters()

    def forward(self, 
                x, 
                ht):
        # x: (batch_size, in_size, h, w), h: (batch_size, hidden_units)
        x_ = x.view(x.size(0), x.size(1), -1)
        x_ = x_.permute(0, 2, 1)
        ht = ht.unsqueeze(1)

        query = self.W(x_)
        keys = self.U(ht)

        # Additive attention gate
        score = query + keys
        if self.bias is not None:
            score += self.bias
        score = self.V(torch.tanh(score))

        # score shape == (batch_size, h*w, 1)
        alpha = F.softmax(score.reshape(score.size(0), -1), dim=1)
        alpha = alpha.reshape(alpha.size(0), 1, x.size(2), x.size(3))

        # context == (batch_size, hidden_units)
        x_attn = alpha * x

        return x_attn, alpha.squeeze(1)

    def reset_parameters(self):
        nn.init.zeros_(self.bias.data)

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

        self.proj = nn.Conv2d(in_size, hidden_size, (1, 1))
        self.attention = BahdanauAttention(hidden_size, hidden_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)

        self.reset_parameters()

    def forward(self,
                Xv):
        # Encode video features with one dense layer and lstm
        # State of this lstm to be used for lstm2 language generator
        # Xv: (batch_size, num_clips, in_size, h, w)

        # Encode video feature using LSTM
        alphas = []
        hi, ci = None, None
        for timestep in range(Xv.shape[1]):
            xv = Xv[:,timestep,:,:,:]

            # LSTM encoding
            hi, ci, alpha = self.step(xv, hi, ci)
            alphas.append(alpha)
        
        # Stack everything
        alphas = torch.stack(alphas, dim=1)

        return hi, (hi, ci), alphas

    def step(self, 
             xv, 
             hi,
             ci):
        # Initialize hidden state for LSTM if needed
        if hi is None or ci is None:
            hi, ci = self.init_hidden(xv)

        # Use 1x1 Conv for Projection
        x = F.relu(self.proj(xv))

        # Attention over spatial resolution
        x, alpha = self.attention(x, hi)

        # Pool over spatial resolution
        x = x.sum(dim=(2, 3))

        # Encode context vector using LSTM
        hi, ci = self.lstm_cell(x, (hi, ci))

        return hi, ci, alpha

    def init_hidden(self,
                    input):
        """Return a initial state for LSTM.
        """
        state_size = [int(input.shape[0]), self.hidden_size]
        h0 = torch.zeros(state_size, requires_grad=False,
                         device=input.device, dtype=input.dtype)
        c0 = torch.zeros(state_size, requires_grad=False,
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