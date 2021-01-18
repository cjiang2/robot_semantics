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
        Xs = self.embed(Xs)
        hi, ci = self.lstm_cell(Xs, states)
        x = self.logits(hi)
        x = self.softmax(x)
        return x, (hi, ci)

    def reset_parameters(self,
                         bias_vector=None):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        nn.init.uniform_(self.embed.weight.data, -0.1, 0.1)
        if bias_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_vector).float()

if __name__ == '__main__':
    # ----------
    decoder = LangDecoder(256, 46, 300)
    for n, p in decoder.named_parameters():
        print(n, p.shape)