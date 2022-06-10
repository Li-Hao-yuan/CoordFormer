from tqdm import tqdm
import math
import torch
import numpy as np
import torch.nn as nn


import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
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

class MaperLayer(nn.Module):
    BN_MOMENTUM = 0.1
    
    def __init__(self,in_channel,out_channel):
        super(MaperLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        out = out + residual
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class EncoderHead(nn.Module):
    def __init__(self):
        super(EncoderHead,self).__init__()
        self.layers = nn.ModuleList([
            MaperLayer(32,16),
            MaperLayer(16,4),
            MaperLayer(4,1),
        ])
        self.end = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Linear(64, 256, bias=False)  #[1,256,512]
        )
    
    def forward(self,x):# [b*3,32,128,128] -> [b*3,64,256]
        for layer in self.layers:
            x = layer(x)

        x = self.end(x)

        return x.squeeze(1)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.head = EncoderHead()

        # temporal embdding
        self.temporal_embedding = PositionalEncoding(args().emb_size)

        # encoder
        encoder_layer_w = nn.TransformerEncoderLayer(d_model=args().emb_size,nhead=args().trans_heads,
                                                             dim_feedforward=args().forward_dim,
                                                             batch_first=True)
        encoder_layer_l = nn.TransformerEncoderLayer(d_model=args().src_len, nhead=args().trans_heads,
                                                     dim_feedforward=args().forward_dim,
                                                     batch_first=True)
        self.encoderw = nn.TransformerEncoder(encoder_layer_w,num_layers=args().trans_layer//2)
        self.encoderl = nn.TransformerEncoder(encoder_layer_l,num_layers=args().trans_layer//2)

        # decoder
        decoder_layer_w = nn.TransformerDecoderLayer(d_model=args().emb_size,nhead=args().trans_heads,
                                                   dim_feedforward=args().forward_dim,
                                                   batch_first=True)
        decoder_layer_l = nn.TransformerDecoderLayer(d_model=args().tgt_len, nhead=args().trans_heads,
                                                     dim_feedforward=args().forward_dim,
                                                     batch_first=True)
        self.decoderw = nn.TransformerDecoder(decoder_layer_w,num_layers=args().trans_layer//2)
        self.decoderl = nn.TransformerDecoder(decoder_layer_l,num_layers=args().trans_layer // 2)


    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, 32, 128, 128]
        dec_inputs: [batch_size, 512, 1024]
        '''

        enc_inputs = self.head(enc_inputs) # [32,128,128] -> [1,64,256]
        # print('enc_inputs',enc_inputs.shape)

        temporal_enc_input = self.temporal_embedding(enc_inputs)
        # print('temporal_enc_input',temporal_enc_input.shape)

        trans_enc_output = self.encoderw(temporal_enc_input)
        trans_enc_output = trans_enc_output.view(-1,args().emb_size,args().src_len)
        trans_enc_output = self.encoderl(trans_enc_output)
        trans_enc_output = trans_enc_output.view(-1, args().src_len,args().emb_size)
        # print('trans_enc_output',trans_enc_output.shape)

        temporal_dec_input = self.temporal_embedding(dec_inputs)
        # print('dec_inputs',dec_inputs.shape)
        # print('temporal_dec_input',temporal_dec_input.shape)

        trans_dec_output = self.decoderw(trans_enc_output,temporal_dec_input)
        trans_dec_output = trans_dec_output.view(-1, args().emb_size, args().tgt_len)
        temporal_dec_input = temporal_dec_input.view(-1, args().emb_size, args().tgt_len)
        trans_dec_output = self.decoderl(trans_dec_output,temporal_dec_input)
        trans_dec_output = trans_dec_output.view(-1, args().tgt_len, args().emb_size)

        trans_dec_output = trans_dec_output.unsqueeze(0)
        # print('trans_dec_output',trans_dec_output.shape)
        # trans_dec_output = torch.zeros_like(trans_dec_output).cuda()

        return trans_dec_output #, camera_map, heatmap#,enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == "__main__":
    model = Transformer()

    import numpy as np
    enc_input = torch.rand((1,32,128,128))
    dec_input = torch.zeros((1,64,256))

    dec_output = model(enc_input,dec_input)[0]
    print('dec_output',dec_output.shape) # [256,512]