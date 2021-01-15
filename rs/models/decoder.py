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
                 bias_vector=None,
                 log_softmax=True):
        super(LangDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim
        self.log_softmax = log_softmax

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
        # Sequence processing and generating
        Xs = self.embed(Xs)
        x = torch.cat([Xs, Xv], dim=1)

        hi, ci = self.lstm_cell(x, states)

        x = self.logits(hi)
        x = self.softmax(x)
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

if __name__ == '__main__':
    # ----------
    decoder = LangDecoder(256, 46, 300)
    for n, p in decoder.named_parameters():
        print(n, p.shape)