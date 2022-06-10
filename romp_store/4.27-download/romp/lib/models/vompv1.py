import sys, os

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
import config
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.transformer import Transformer
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from models.base import Base

class ParamLayer(nn.Module):
    BN_MOMENTUM = 0.1
    
    def __init__(self,in_channel,out_channel,sample=None):
        super(ParamLayer,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=0.1)
        
        self.sample = sample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        out += residual
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.sample is not None:
            out = self.sample(out)

        return out

class ParamEncoder(nn.Module):
    '''
    [689x3,64,64] -> [1,256,512]
    '''
    def __init__(self,final_channel=1):
        super(ParamEncoder,self).__init__()
        self.head_linear = nn.Linear(689*3, 128, bias=False)
        self.sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layers = nn.ModuleList([
            ParamLayer(128,64,self.sample),
            ParamLayer(64,16,self.sample),
            ParamLayer(16,final_channel,None)
            ])
        self.end_linear = nn.Linear(256, 512, bias=False)
        
        
    def forward(self,enc_outputs):
        batch_size = enc_outputs.shape[0]
        feature_map = enc_outputs.view(batch_size,64,64,-1) # [64,64,6890x3]
        feature_map = self.head_linear(feature_map) # [64,64,128]
        feature_map = feature_map.view(batch_size,-1,64,64) # [128,64,64]
        
        for layer in self.layers: # -> [final_channel,256,256]
            feature_map = layer(feature_map)
        feature_map = self.end_linear(feature_map)
        
        return feature_map.view(batch_size,256,512)

class ParamDecoder(nn.Module):
    '''
    [1,256,512]->[142,64,64] -> [6890*3,64,64]
    '''
    def __init__(self,final_channel=142):
        super(ParamDecoder,self).__init__()
        self.layer_norm = nn.LayerNorm([64,64])
        self.projection = nn.Linear(142,689*3)
        self.linear = nn.Linear(512,256,bias=False)
        self.sample = nn.MaxPool2d(2)
        self.layers = nn.ModuleList([
            ParamLayer(1,32,self.sample),
            ParamLayer(32,64,self.sample),
            ParamLayer(64,final_channel,None)
            ])
        
        
    def forward(self,enc_outputs,get_params=True):

        batch_size = enc_outputs.shape[0]
        feature_map = enc_outputs.view(batch_size,1,256,512)
        feature_map = self.linear(feature_map) # [1,256,256]

        for layer in self.layers:
            feature_map = layer(feature_map)

        vertice_map = feature_map.view(batch_size,64,64,142)
        vertice_map = self.projection(vertice_map).view(batch_size,-1,64,64)

        if get_params:
            params_map = feature_map.view(batch_size,142,64,64)
            params_map = self.layer_norm(params_map)

            return params_map,vertice_map
        else:
            return vertice_map


class VOMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(VOMP,self).__init__()
        self.backbone = backbone
        self.param_enc = ParamEncoder()
        self.param_dec = ParamDecoder()
        self._result_parser = ResultParser()
        self.trans = Transformer()
        
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def train_forward(self,out,params_map):
        '''
        out : [batch_size,32,128,128]
        params_map : [batch_size,76,64,64]
        '''
        # dec_output = [batch_size,1,512,1024]
        # camera_map = [batch_size,3,64,64]
        # center_map = [batch_size,1,64,64]
        params_map = self.param_enc(params_map) # [689x3,64,64] -> [256,512]
        dec_output,camera_map,center_map = self.trans(out,params_map) 
        param_output,vertice_output = self.param_dec(dec_output) # [1,256,512]->[142,64,64] -> -> vertice

        param_output = torch.cat([camera_map, param_output], 1)

        output = {'params_maps':param_output.float(),'center_map':center_map.float(),'pred_vertice':vertice_output.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()

        return output

    def predict_forward(self,out):
        '''
        out : [batch_size,32,128,128]
        '''
        batch_size = out.shape[0]
        params_loop = torch.zeros(1,689*3,64,64).cuda()

        for frame in range(batch_size):
            params_map = self.param_enc(params_loop)

            if frame+1 < batch_size:
                dec_output = self.trans(out[:frame+1,:,:,:],params_map,last_frame=False)
                vertice_output = self.param_dec(dec_output,get_params=False)

                params_loop = torch.cat((params_loop,vertice_output),dim=0)
            else:
                dec_output,camera_map,center_map = self.trans(out[:frame+1,:,:,:],params_map,last_frame=True)
                param_output,vertice_output = self.param_dec(dec_output,get_params=True)

        param_output = torch.cat([camera_map, param_output], 1)
        output = {'params_maps':param_output.float(),'center_map':center_map.float(),'pred_vertice':vertice_output.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()
        return output

    def head_forward(self,out,params_map=None):
        if params_map == None:
           output = self.predict_forward(out)
        else:
            output = self.train_forward(out,params_map)
        return output


if __name__ == "__main__":
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    args().predict = True
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({
        'image':torch.rand(2,512,512,3).cuda(),
        # 'params':torch.rand(2,64,76).cuda(),
        # 'person_centers':torch.rand(2,64,2).cuda()
        })

    print("ok!")