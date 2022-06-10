from tqdm import tqdm
import math
import torch
import numpy as np
import torch.nn as nn

'''
wait:
dec input的encoder
dec output的decoder
输出结果的转化
'''

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

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(args().dk) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e3) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.W_Q = nn.Linear(emb_size, args().dk * args().trans_heads, bias=False)
        self.W_K = nn.Linear(emb_size, args().dk * args().trans_heads, bias=False)
        self.W_V = nn.Linear(emb_size, args().dv * args().trans_heads, bias=False)
        self.fc = nn.Linear(args().trans_heads * args().dv, emb_size, bias=False)
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
        Q = self.W_Q(input_Q).view(batch_size, -1, args().trans_heads, args().dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, args().trans_heads, args().dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, args().trans_heads, args().dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, args().trans_heads, 1, 1).to(torch.bool).cuda() # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, args().trans_heads * args().dv) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return self.layerNorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,emb_size):
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
        return self.layerNorm(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,emb_size):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(emb_size)
        self.pos_ffn = PoswiseFeedForwardNet(emb_size)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self,emb_size):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(emb_size)
        self.dec_enc_attn = MultiHeadAttention(emb_size)
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
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class PointerLayer(nn.Module):
    BN_MOMENTUM = 0.1
    
    def __init__(self,in_channel,out_channel,upsample=True):
        super(PointerLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=0.1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2) if upsample else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        out += residual
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        return out
    
class EncoderHead(nn.Module):
    def __init__(self):
        super(EncoderHead,self).__init__()
        self.layer1 = PointerLayer(32,8,False)
        self.layer2 = PointerLayer(8,1,False)
        self.pool2d = nn.MaxPool2d(2)
        
        self.linear = nn.Sequential(
            nn.Linear(64*64,64*80),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        '''
        [b,32,128,128] -> [b,64,80]
        '''
        out = self.layer1(x) # -> [b,8,128,128]
        out = self.layer2(out) # -> [b,1,128,128]
        out = self.pool2d(out) # -> [b,1,64,64]
        out = out.view(-1,64*64)
        out = self.linear(out) # -> [b,64*79]
        out = out.view(-1,64,80) # -> [b,64,80]
        
        return out
    
class PreEncoder(nn.Module):
    def __init__(self):
        super(PreEncoder, self).__init__()
        self.src_emb = nn.Embedding(args().pre_src_len, args().pre_emb_size)
        self.pos_emb = PositionalEncoding(args().pre_emb_size)
        self.layers = nn.ModuleList([EncoderLayer(args().pre_emb_size) for _ in range(args().trans_layer)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        batch_size = enc_inputs.shape[0]
        enc_outputs = enc_inputs.view(batch_size,-1,args().pre_emb_size) # [batch,32,128,128] -> [batch,512,1024]
        
        
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        
        # 定长
        enc_self_attn_mask = torch.ones((batch_size,args().pre_src_len,args().pre_src_len))
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.head = EncoderHead()
        self.pos_emb = PositionalEncoding(args().emb_size)
        self.layers = nn.ModuleList([EncoderLayer(args().emb_size) for _ in range(args().trans_layer)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, pre_src_len, pre_emb_size]
        '''
        batch_size = enc_inputs.shape[0]
        enc_outputs = self.head(enc_inputs.view(batch_size,32,128,128))
        
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        
        # 定长
        enc_self_attn_mask = torch.ones((batch_size,args().src_len,args().src_len))
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(args().emb_size)
        self.layers = nn.ModuleList([DecoderLayer(args().emb_size) for _ in range(args().trans_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        # dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        batch_size = dec_inputs.shape[0]
        
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
        
        #dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.ones((batch_size,args().tgt_len,args().tgt_len)) # [batch_size, tgt_len, tgt_len]
        
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]

        #dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]
        dec_enc_attn_mask = torch.ones((batch_size,args().tgt_len,args().src_len)) # [batc_size, tgt_len, src_len]


        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Maper(nn.Module):
    def __init__(self,final_channel=1,upsample=True):
        super(Maper,self).__init__()
        self.upsample = upsample
        self.pool2d = nn.MaxPool2d(2)
        self.layers = nn.ModuleList([
            PointerLayer(32,16,upsample),
            PointerLayer(16,4,upsample),
            PointerLayer(4,final_channel,False)
            ])
        
        
    def forward(self,enc_outputs):
        '''
        enc_outputs: [batch_size, src_len, d_model] [256,512] -> [32,128,128] -> [1,512,512]
        '''
        batch_size = enc_outputs.shape[0]
        feature_map = enc_outputs.view(batch_size,32,128,128)
        for layer in self.layers:
            feature_map = layer(feature_map)
        
        if not self.upsample: # camera map
            feature_map = self.pool2d(feature_map)
        
        return feature_map

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.pre_encoder = PreEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.pointer = Maper()
        self.cameraer = Maper(3,upsample=False)
        self.projection = nn.Linear(args().emb_size, 132+10+2+1, bias=False)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        pre_enc_outputs, pre_enc_self_attns = self.pre_encoder(enc_inputs)

        # heatmap = self.pointer(pre_enc_outputs) # [1,512,512]
        camera_map = self.cameraer(pre_enc_outputs) # [3,64,64]
        
        enc_outputs,enc_self_attns = self.encoder(pre_enc_outputs)

        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        # normalize all channel
        dec_logits[:,:,:132]   = self.tanh(dec_logits[:,:,:132])
        dec_logits[:,:,132:142] = self.tanh(dec_logits[:,:,132:142])
        dec_logits[:,:,142]    = self.sig(dec_logits[:,:,142])
        dec_logits[:,:,143]    = self.sig(dec_logits[:,:,143])
        dec_logits[:,:,144]    = self.sig(dec_logits[:,:,144])
        
        
        return dec_logits, camera_map#,enc_self_attns, dec_self_attns, dec_enc_attns

if __name__ == "__main__":
    model = Transformer()

    import numpy as np
    enc_input = torch.rand((1,32,128,128))
    dec_input = torch.zeros((1,64,80))

    dec_output = model(enc_input,dec_input)[0] # [1,64,80]

    print( dec_output.shape )
    print( torch.max(dec_output[:,:,:66]) , torch.max(dec_output[:,:,66:76]) , torch.max(dec_output[:,:,76:78]) , torch.max(dec_output[:,:,78]) )
    print( torch.min(dec_output[:,:,:66]) , torch.min(dec_output[:,:,66:76]) , torch.min(dec_output[:,:,76:78]) , torch.min(dec_output[:,:,78]) )