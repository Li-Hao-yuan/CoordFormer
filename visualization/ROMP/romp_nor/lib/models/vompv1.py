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
from models.transformer import Transformer
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from models.base import Base

class ParamLayer(nn.Module):
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

class ParamEncoder(nn.Module):
    '''
    [64,54,3] -> [1,64,256]
    '''
    def __init__(self):
        super(ParamEncoder,self).__init__()
        self.head_linear = nn.Linear(54*3, 256, bias=False)
        
        
    def forward(self,enc_outputs):
        # print('enc_outputs',enc_outputs.shape)
        enc_outputs = enc_outputs.view(-1,64,162)
        enc_outputs = self.head_linear(enc_outputs) # [b,64,108] -> [b,64,256]
        
        return enc_outputs.view(-1,1,64,256)

class ParamDecoder(nn.Module):
    '''
    [b,8,64,256]->[142,64,64]
    '''
    def __init__(self,):
        super(ParamDecoder,self).__init__()
        self.linear = nn.Linear(256,64,bias=False)
        self.layers = nn.ModuleList([
            ParamLayer(args().data_slide_len,16),
            ParamLayer(16,32),
            ParamLayer(32,64)
            ])
        
        
    def forward(self,enc_outputs):
        feature_map = self.linear(enc_outputs) #[1,64,64]
        feature_map = feature_map.view(-1,args().data_slide_len,64,64)

        for layer in self.layers:
            feature_map = layer(feature_map)

        return feature_map


class VOMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(VOMP,self).__init__()
        def regression_head(final_channel=1):
            return nn.Sequential(
                nn.Conv2d(64, final_channel, 1, 1),
                # nn.BatchNorm2d(1),
                # nn.Tanh() if tanh else nn.Sigmoid(),
                # nn.MaxPool2d(4),
            )

        self.backbone = backbone
        self.param_enc = ParamEncoder()
        self.param_dec = ParamDecoder()
        self.pointer = regression_head(1)
        self.camera = regression_head(3)
        self.params = regression_head(142)

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
        params_map : [batch_size,64,54,3]
        '''
        # dec_output = [batch_size,1,64,256]
        # camera_map = [batch_size,3,64,64]
        # center_map = [batch_size,1,64,64]
        params_map = self.param_enc(params_map).squeeze(1)
        dec_output = self.trans(out,params_map)
        param_dec = self.param_dec(dec_output)

        param_output = self.params(param_dec)
        camera_map = self.camera(param_dec)
        center_map = self.pointer(param_dec)
        # print('param_output',param_output.shape)
        # print('camera_map',camera_map.shape)
        # print('center_map',center_map.shape)
        # exit()

        camera_map[:,0] = torch.pow(1.1,camera_map[:,0])

        param_output = torch.cat([camera_map, param_output], 1)

        output = {'params_maps':param_output.float(),'center_map':center_map.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()

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

                params_loop = torch.cat((params_loop,vertice_output[-1:]),dim=0)
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
    args().predict = False
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({
        'image':torch.rand(3,512,512,3).cuda(),
        'other_kp2d':torch.rand(3,64,54,2).cuda(),
        # 'person_centers':torch.rand(2,64,2).cuda()
        })

    print("ok!")