import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------
# Language Decoder
# ----------------------------------------

class LangDecoder(nn.Module):
    """Module to decode features and generate word for sentence
    sequence using RNN.
    """
    def __init__(self,
                 units,
                 vocab_size,
                 embed_dim,
                 bias_vector=None):
        super(LangDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        self.initialized = False

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + units, units)
        self.logits = nn.Linear(units, vocab_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.reset_parameters(bias_vector)

    def forward(self, 
                Xs, 
                states, 
                Xv):
        # Phase 2: Decoding Stage
        # Given the previous word token, generate next caption word using lstm2

        # Not initialized, use Xv only:
        if not self.initialized:
            for timestep in range(Xv.shape[1]):
                Xv_step = Xv[:,timestep,:]
                Xs_shape = [int(Xv_step.shape[0]), self.embed_dim]
                x = torch.cat((Xv_step, torch.zeros(Xs_shape).to(Xv.device)), dim=-1)
                hi, ci = self.lstm_cell(x)
            x = None
            self.initialized = True

        else:
            # Sequence processing and generating
            #print('sentence decoding stage:')
            #print('Xs:', Xs.shape)
            Xs = self.embed(Xs)
            #print('embed:', Xs.shape)
            #print('Xv:', Xv.shape)
            x = torch.cat((Xv[:,-1,:], Xs), dim=-1)
            #print(x.shape)
            #exit()

            hi, ci = self.lstm_cell(x, states)
            #print('out:', hi.shape, 'hi:', states[0].shape, 'ci:', states[1].shape)

            x = self.logits(hi)
            #print('logits:', x.shape)
            x = self.softmax(x)
            #print('softmax:', x.shape)
        return x, (hi, ci)

    def reset_parameters(self,
                         bias_vector=None):
        for i in range(4):
            nn.init.orthogonal_(self.lstm_cell.weight_hh.data[self.units*i:self.units*(i+1)])
        nn.init.zeros_(self.lstm_cell.bias_ih)
        nn.init.zeros_(self.lstm_cell.bias_hh)
        if bias_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_vector).float()

        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def reset(self):
        self.initialized = False

if __name__ == '__main__':
    # ----------
    decoder = LangDecoder(256, 46, 300)
    for n, p in decoder.named_parameters():
        print(n, p.shape)