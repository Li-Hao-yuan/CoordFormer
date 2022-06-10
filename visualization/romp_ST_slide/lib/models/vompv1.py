from ast import arg
import sys, os

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
import config
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.transformer import Temporal_trans,Spatial_trans
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from models.base import Base

class ParamLayer(nn.Module):
    ''' ResNet layer '''
    BN_MOMENTUM = 0.1
    
    def __init__(self,in_channel,out_channel):
        super(ParamLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        out += residual
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class EncoderHead(nn.Module):
    '''
    [b,slide,64,128,128] -> [b*s,64,128,128] -> [b*s,32,32]
    '''
    def __init__(self):
        super(EncoderHead,self).__init__()
        self.layers = nn.ModuleList([
            ParamLayer(64,16),
            ParamLayer(16,4),
            ParamLayer(4,1),
        ])
        self.pool2d = nn.MaxPool2d(2)
    
    def forward(self,x):# [b*s,32,128,128] -> [b*s,32,32]

        x = self.layers[0](x)
        x = self.pool2d(x)
        x = self.layers[1](x)
        x = self.pool2d(x)
        x = self.layers[2](x)

        return x.squeeze(1)

class ParamDecoder(nn.Module):
    '''
    [b,s,1024]->[b,1,s,1024]->[64,64,64]
    '''
    def __init__(self,):
        super(ParamDecoder,self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.head = nn.Linear(1024,128)
        self.projection = nn.Linear(128,64)
        self.layers = nn.ModuleList([
            ParamLayer(1,16),
            ParamLayer(16,32),
            ParamLayer(32,64)
            ])
        
        
    def forward(self,enc_outputs):
        feature_map = enc_outputs.view(-1,1,args().data_slide_len,args().emb_size*args().emb_size)
        feature_map = self.head(feature_map)

        for layer in self.layers:
            feature_map = layer(feature_map)
            feature_map = self.projection(feature_map)
            feature_map = self.up(feature_map)
        
        return self.projection(feature_map)

class MoCA(nn.Module):
    def __init__(self):
        super(MoCA,self).__init__()
        
        self.conv1 = nn.Conv2d(1,1,5,1,2)
        self.conv2 = nn.Conv2d(1,1,5,1,2)
        self.conv3 = nn.Conv2d(2,1,5,1,2)
        self.conv4 = nn.Conv2d(1,1,5,1,2)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self,x):
        '''
        x : [b,s,1024]
        '''
        xT = x.permute(0,2,1) # [2,1024,8]
        
        NSSM = self.softmax( torch.matmul(x,xT) ) # [b,s,s]
        # print('NSSM:',NSSM.shape)
        
        conv_x = self.conv1(x.unsqueeze(1)).squeeze(1)
        conv_xT = self.conv2(xT.unsqueeze(1)).squeeze(1)
        Attention_map = self.softmax( torch.matmul(conv_x,conv_xT) )
        # print('Attention_map:',Attention_map.shape)
        
        Moca_map = self.conv3( torch.cat((NSSM.unsqueeze(1),Attention_map.unsqueeze(1)),dim=1) ).squeeze(1)
        # print('Moca_map:',Moca_map.shape)
        
        conv_g = self.conv4(x.unsqueeze(1)).squeeze(1)
        Wz = torch.matmul(Moca_map,conv_g)
        
        return Wz + x
        
class VOMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(VOMP,self).__init__()
        
        self.head = EncoderHead()
        self.tem_trans = Temporal_trans()
        self.spa_trans = Spatial_trans()
        self.moca = MoCA()
        
        self.param_dec = ParamDecoder()
        self.center = nn.Sequential(nn.Conv2d(64, 1, 1, 1))
        self.camera = nn.Sequential(nn.Conv2d(64, 3, 1, 1))
        self.params = nn.Sequential(nn.Conv2d(64, 142, 1, 1))

        self.backbone = backbone
        self._result_parser = ResultParser()
        
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def head_forward(self,out):
        '''
        out : [batch_size*slide,64,128,128]
        '''
        # dec_output = [batch_size,1,64,256]
        # camera_map = [batch_size,3,64,64]
        # center_map = [batch_size,1,64,64]
        spatial_trans_input = self.head(out) # [b*s,64,128,128] -> [b*s,32,32]
        spatial_trans_output = self.spa_trans(spatial_trans_input) # -> [b*s,32,32]
        spatial_trans_output += spatial_trans_input
        # print('spatial_trans_output',spatial_trans_output.shape)
        
        # [b,s,32,32] -> [b,s,1024]
        temproal_trans_input = spatial_trans_output.view(-1,args().data_slide_len,args().emb_size*args().emb_size)
        temproal_trans_input = self.moca(temproal_trans_input)
        temproal_trans_output = self.tem_trans(temproal_trans_input) # -> [b,s,1024]
        temproal_trans_output += temproal_trans_input
        # print('temproal_trans_output',temproal_trans_output.shape)
        
        param_dec = self.param_dec(temproal_trans_output)
        param_output = self.params(param_dec)
        camera_map   = self.camera(param_dec)
        center_map   = self.center(param_dec)
        # print('param_output',param_output.shape)
        # print('camera_map',camera_map.shape)
        # print('center_map',center_map.shape)

        camera_map[:,0] = torch.pow(1.1,camera_map[:,0])

        param_output = torch.cat([camera_map, param_output], 1)

        output = {'params_maps':param_output.float(),'center_map':center_map.float()} 

        return output


if __name__ == "__main__":
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    args().predict = True
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({
        'image':torch.rand(8,512,512,3).cuda(),
        # 'other_kp2d':torch.rand(3,64,54,2).cuda(),
        # 'person_centers':torch.rand(2,64,2).cuda()
        })

    print("ok!")