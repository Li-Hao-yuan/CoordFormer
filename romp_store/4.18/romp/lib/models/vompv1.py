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
    [76,64,64] -> [1,256,1024]
    '''
    def __init__(self,final_channel=1):
        super(ParamEncoder,self).__init__()
        self.sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layers = nn.ModuleList([
            ParamLayer(76,32,self.sample),
            ParamLayer(32,16,self.sample),
            ParamLayer(16,final_channel,self.sample)
            ])
        
        
    def forward(self,enc_outputs):
        batch_size = enc_outputs.shape[0]
        feature_map = enc_outputs.view(batch_size,76,64,64)
        for layer in self.layers:
            feature_map = layer(feature_map)
        
        return feature_map.view(batch_size,256,1024)

class ParamDecoder(nn.Module):
    '''
    [1,256,1024] ->  [142,64,64]
    '''
    def __init__(self,final_channel=142):
        super(ParamDecoder,self).__init__()
        self.tanh = nn.Tanh()
        self.sample = nn.MaxPool2d(2)
        self.layers = nn.ModuleList([
            ParamLayer(1,32,self.sample),
            ParamLayer(32,64,self.sample),
            ParamLayer(64,final_channel,self.sample)
            ])
        
        
    def forward(self,enc_outputs):

        batch_size = enc_outputs.shape[0]
        feature_map = enc_outputs.view(batch_size,1,256,1024)

        for layer in self.layers:
            feature_map = layer(feature_map)

        feature_map = feature_map.view(batch_size,142,64,64)
        feature_map = self.tanh(feature_map)

        return feature_map


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

    def head_forward(self,out,params_map):
        '''
        out : [batch_size,32,128,128]
        params_map : [batch_size,76,64,64]
        '''
        # dec_output = [batch_size,1,512,1024]
        # camera_map = [batch_size,3,64,64]
        # center_map = [batch_size,1,64,64]
        params_map = self.param_enc(params_map)
        dec_output,camera_map,center_map = self.trans(out,params_map)
        param_output = self.param_dec(dec_output)

        param_output = torch.cat([camera_map, param_output], 1)

        output = {'params_maps':param_output.float(), 'center_map':center_map.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()

        return output


if __name__ == "__main__":
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({
        'image':torch.rand(2,512,512,3).cuda(),
        'params':torch.rand(2,64,76).cuda(),
        'person_centers':torch.rand(2,64,2).cuda()
        })

    print("ok!")