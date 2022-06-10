from tqdm import tqdm
import math
import torch
import numpy as np
import torch.nn as nn


import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import args



class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Temporal_trans(nn.Module):
    def __init__(self):
        super(Temporal_trans, self).__init__()

        # temporal embdding
        self.temporal_embedding = PositionalEncoding(args().emb_size*args().emb_size)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args().emb_size*args().emb_size,nhead=args().trans_heads,
                                                             dim_feedforward=args().forward_dim,
                                                             batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=args().trans_layer)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, silde, 32, 32]
        '''
        
        # -> [b,s,32*32]
        enc_inputs = enc_inputs.view(-1,args().data_slide_len,args().emb_size*args().emb_size)

        temporal_enc_input = self.temporal_embedding(enc_inputs)
        # print('temporal_enc_input',temporal_enc_input.shape)

        trans_enc_output = self.encoder(temporal_enc_input)
        # print('trans_enc_output',trans_enc_output.shape)

        return trans_enc_output  # [batch_size, 256, 256]


class Spatial_trans(nn.Module):
    def __init__(self):
        super(Spatial_trans, self).__init__()

        # temporal embdding
        self.temporal_embedding = PositionalEncoding(args().emb_size)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args().emb_size,nhead=args().trans_heads,
                                                             dim_feedforward=args().forward_dim,
                                                             batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=args().trans_layer)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, 32, 32]
        '''

        temporal_enc_input = self.temporal_embedding(enc_inputs)
        # print('temporal_enc_input',temporal_enc_input.shape)

        trans_enc_output = self.encoder(temporal_enc_input)
        # print('trans_enc_output',trans_enc_output.shape)

        return trans_enc_output 


if __name__ == "__main__":
    tem_trans = Temporal_trans()
    spa_trans = Spatial_trans()

    import numpy as np
    tem_enc_input = torch.rand((2,8,32,32))

    tem_output = tem_trans(tem_enc_input)
    print('tem_output',tem_output.shape) # [1,8,1024]
    
    spa_enc_input = torch.rand((2,32,32))
    spa_output = spa_trans(spa_enc_input)
    print('spa_output',spa_output.shape) # [1,32,32] ~ [8,32,32]