from tqdm import tqdm
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from config import args


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

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
        # print('x',x.shape)
        # print('pe',self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding2d(nn.Module):
    def __init__(self, src_len, dk, dropout=0.1, max_len=8):
        super(PositionalEncoding2d, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe2d = torch.zeros(max_len, src_len,dk)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, src_len, 2).float() * (-math.log(1000.0) / src_len))
        div_term_l = torch.exp(torch.arange(0, dk, 2).float() * (-math.log(1000.0) / dk))
        pe2d[:, 0::2, :] += torch.sin(position * div_term_w).unsqueeze(2)
        pe2d[:, 1::2, :] += torch.cos(position * div_term_w).unsqueeze(2)

        pe2d[:, :, 0::2] *= torch.sin(position * div_term_l).unsqueeze(1)
        pe2d[:, :, 1::2] *= torch.cos(position * div_term_l).unsqueeze(1)

        pe2d = pe2d.unsqueeze(0)
        self.register_buffer('pe2d', pe2d)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        # print('x',x.shape)
        # print('pe',self.pe2d[:,:x.size(1)].shape)
        x = x + self.pe2d[:,:x.size(1)]
        return self.dropout(x)

class NTAttention(nn.Module):
    def __init__(self, emb_size, dk=args().dk):
        super(NTAttention, self).__init__()
        self.emb_size = emb_size
        self.dk = dk

        self.temporal_embedding = PositionalEncoding2d(dk,dk) # 256 64

        self.k_conv = nn.Conv2d(args().trans_heads, args().trans_heads, 3, 1, 1)
        self.v_conv = nn.Parameter(torch.ones(args().trans_heads, args().trans_heads, 3, 3), requires_grad=False)

        self.W_Q = nn.Linear(emb_size, dk * args().trans_heads, bias=False)
        self.W_K = nn.Linear(emb_size, dk * args().trans_heads, bias=False)
        self.W_V = nn.Linear(emb_size, dk * args().trans_heads, bias=False)
        self.fc = nn.Linear(args().trans_heads * self.dk, emb_size, bias=False)
        self.layerNorm = nn.LayerNorm(self.emb_size)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, args().trans_heads, self.dk).transpose(1,
                                                                                          2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, args().trans_heads, self.dk).transpose(1,
                                                                                          2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, args().trans_heads, self.dk).transpose(1,
                                                                                          2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # make conv
        K = self.k_conv(K)
        V = F.conv2d(V, self.v_conv, None, 1, 1)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, args().trans_heads, 1, 1).to(
            torch.bool).cuda() # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # ScaledDotProductAttention
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        scores = ((Q * K) / np.sqrt(args().dk))
        scores.masked_fill_(attn_mask, -1e3)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.temporal_embedding(attn)  ###################################################
        # print('attn',attn.shape)
        # exit()
        context = attn * V

        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  args().trans_heads * self.dk)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return self.layerNorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, emb_size):
        super(PoswiseFeedForwardNet, self).__init__()
        self.emb_size = emb_size
        self.layerNorm = nn.LayerNorm(self.emb_size)
        self.fc = nn.Sequential(
            nn.Linear(emb_size, args().forward_dim, bias=False),
            nn.ReLU(),
            nn.Linear(args().forward_dim, emb_size, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layerNorm(output + residual)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, emb_size, dk=args().dk):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = NTAttention(emb_size, dk)
        self.pos_ffn = PoswiseFeedForwardNet(emb_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]


        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self, emb_size, dk=args().dk):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = NTAttention(emb_size, dk)
        self.dec_enc_attn = NTAttention(emb_size, dk)
        self.pos_ffn = PoswiseFeedForwardNet(emb_size)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class MaperLayer(nn.Module):
    '''
    [in_channel,w,h] -> [out_channel,w,h]
    '''
    BN_MOMENTUM = 0.1

    def __init__(self, in_channel, out_channel):
        super(MaperLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
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
    '''
    [b*3,32,128,128] -> [b*3,1,64,256]
    '''

    def __init__(self):
        super(EncoderHead, self).__init__()
        self.layers = nn.ModuleList([
            MaperLayer(64, 16),
            MaperLayer(16, 4),
            MaperLayer(4, 1),
        ])
        self.end = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Linear(64, 256, bias=False)  # [1,256,512]
        )

    def forward(self, x):  # [b*3,32,128,128] -> [b*3,64,256]
        for layer in self.layers:
            x = layer(x)

        x = self.end(x)

        return x.squeeze(1)

class Encoder(nn.Module):
    '''
    [b*3,64,256] -> [b*3,64,256]
    '''

    def __init__(self, emb_size=args().emb_size):
        super(Encoder, self).__init__()
        self.attention_on_width = (emb_size == args().emb_size)
        dk = args().dk if self.attention_on_width else args().emb_size

        self.pos_emb = PositionalEncoding(emb_size)
        self.layers = nn.ModuleList([EncoderLayer(emb_size, dk) for _ in range(args().trans_layer)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, emb_size]
        '''
        if not self.attention_on_width:
            enc_inputs = enc_inputs.view(-1, args().emb_size, args().src_len)

        batch_size = enc_inputs.shape[0]
        enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]

        # 定长
        enc_self_attn_mask = torch.zeros((batch_size, enc_inputs.shape[1], enc_inputs.shape[1])).to(bool)
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        if not self.attention_on_width:
            enc_outputs = enc_outputs.view(-1, args().src_len, args().emb_size)

        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self, emb_size=args().emb_size):
        super(Decoder, self).__init__()
        self.attention_on_width = (emb_size == args().emb_size)
        dk = args().dk if self.attention_on_width else args().emb_size

        self.pos_emb = PositionalEncoding(emb_size)
        self.layers = nn.ModuleList([DecoderLayer(emb_size, dk) for _ in range(args().trans_layer)])

    def forward(self, dec_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''

        if not self.attention_on_width:
            dec_inputs = dec_inputs.view(-1, args().emb_size, args().tgt_len)
            enc_outputs = enc_outputs.view(-1, args().emb_size, args().src_len)

        # dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        batch_size = dec_inputs.shape[0]
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]

        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.zeros((batch_size, dec_inputs.shape[1], dec_inputs.shape[1])).to(
            bool)  # [batch_size, tgt_len, tgt_len]

        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0)  # [batch_size, tgt_len, tgt_len]

        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]
        dec_enc_attn_mask = torch.zeros((batch_size, dec_inputs.shape[1], dec_inputs.shape[1])).to(
            bool)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        if not self.attention_on_width:
            dec_outputs = dec_outputs.view(-1, args().tgt_len, args().emb_size)

        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.head = EncoderHead()

        self.encoderw = Encoder()
        self.encoderl = Encoder(args().src_len)

        self.decoderw = Decoder()
        self.decoderl = Decoder(args().tgt_len)

    def forward(self, enc_inputs,dec_input):
        '''
        enc_inputs: [batch_size*3, 32, 128, 128]
        dec_inputs: [batch_size*3, 64, 256]
        '''
        # [b*3,32,128,128] -> [b*3,64,256]
        enc_inputs = self.head(enc_inputs)

        # [b*3,64,256] -> [b*3,64,256]
        enc_outputs, enc_self_attns = self.encoderw(enc_inputs)
        # enc_outputs, enc_self_attns = self.encoderl(enc_outputs)
        # print('enc_outputs',enc_outputs.shape)

        # [b*3,64,256],[b*3,64,256] -> [b,64,256]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoderw(enc_outputs,dec_input)
        # dec_outputs, dec_self_attns, dec_enc_attns = self.decoderl(enc_outputs, dec_outputs)
        dec_outputs = dec_outputs.unsqueeze(0)
        # print('dec_outputs',dec_outputs.shape)

        return dec_outputs


if __name__ == "__main__":
    model = Transformer()

    import numpy as np

    enc_input = torch.rand((3, 32, 128, 128))
    dec_input = torch.zeros((3, 64, 256))

    dec_output = model(enc_input,dec_input)
    print('dec_output', dec_output.shape)  # [b,64,256]