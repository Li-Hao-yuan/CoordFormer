import sys, os

root_dir = os.path.join(os.path.dirname(__file__), '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
from config import args
from models.hrnet_32 import HigherResolutionNet
from models.transformer import Transformer
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from models.base import Base


class ParamLayer(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, in_channel, out_channel):
        super(ParamLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channel, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
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
    def __init__(self):
        super(EncoderHead, self).__init__()
        self.layers = nn.ModuleList([
            ParamLayer(64, 16),
            ParamLayer(16, 4),
            ParamLayer(4, 1),
        ])
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):  # [b*s,32,128,128] -> [b*s,256,256]
        for layer in self.layers:
            x = layer(x)

        x = self.up(x)

        return x.squeeze(1)


class CenterMap(nn.Module):
    def __init__(self):
        super(CenterMap, self).__init__()
        self.layers = nn.ModuleList([
            ParamLayer(args().data_slide_len, 16),
            ParamLayer(16, 8),
            ParamLayer(8, 1),
        ])
        # self.down = nn.Conv2d(16,1,1,1)

    def forward(self, x):  # [b*s,256,256] -> [b,256,256]
        x = x.view(-1, args().data_slide_len, 256, 256)
        for layer in self.layers:
            x = layer(x)
        # x = self.down(x)

        return x.squeeze(1)


class ParamDecoder(nn.Module):
    '''
    [b*s,256,256]->[32,64,64]
    '''

    def __init__(self, ):
        super(ParamDecoder, self).__init__()
        self.pool2d = nn.MaxPool2d(2)
        self.layers = nn.ModuleList([
            ParamLayer(args().data_slide_len, 16),
            ParamLayer(16, 32)
        ])

    def forward(self, enc_outputs):
        feature_map = enc_outputs.view(-1, args().data_slide_len, 256, 256)

        for layer in self.layers:
            feature_map = layer(feature_map)
            feature_map = self.pool2d(feature_map)

        return feature_map


class VOMP(Base):
    def __init__(self, backbone=None, **kwargs):
        super(VOMP, self).__init__()
        self.backbone = backbone
        self._result_parser = ResultParser()

        self.head = EncoderHead()
        self.tem_trans = Transformer()
        self.spa_trans = Transformer()

        self.centermap = CenterMap()

        self.param_dec = ParamDecoder()
        self.camera = nn.Sequential(nn.Conv2d(32, 3, 1, 1))
        self.params = nn.Sequential(nn.Conv2d(32, 142, 1, 1))
        self.center = nn.Sequential(nn.MaxPool2d(4))

        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def head_forward(self, out):
        '''
        out : [batch_size*slide,32,128,128]
        '''
        # dec_output = [batch_size,1,64,256]
        # camera_map = [batch_size,3,64,64]
        # center_map = [batch_size,1,64,64]
        temporal_trans_input = self.head(out)  # [b*s,32,128,128] -> [b*s,256,256]
        temproal_trans_output = self.tem_trans(temporal_trans_input)  # -> [b*s,256,256]
        centermap = self.centermap(temproal_trans_output)  # -> [b,256,256]
        temproal_trans_output = temproal_trans_output + temproal_trans_output*centermap.repeat(args().data_slide_len, 1, 1)
        spatial_trans_output = self.spa_trans(temproal_trans_output)  # -> [b*s,256,256]

        param_dec = self.param_dec(spatial_trans_output + temproal_trans_output)
        param_output = self.params(param_dec)
        camera_map = self.camera(param_dec)
        center_map = self.center(centermap)
        # print('param_output',param_output.shape)
        # print('camera_map',camera_map.shape)
        # print('center_map',center_map.shape)

        camera_map[:, 0] = torch.pow(1.1, camera_map[:, 0])

        param_output = torch.cat([camera_map, param_output], 1)

        output = {'params_maps': param_output.float(), 'center_map': center_map.unsqueeze(1).float()}

        return output


if __name__ == "__main__":
    args().configs_yml = 'configs/v1.yml'
    args().model_version = 1
    args().predict = True
    args().backbone = 'resnet'
    from models.build import build_model

    model = build_model().cuda()
    outputs = model.feed_forward({
        'image': torch.rand(8, 512, 512, 3).cuda(),
        # 'other_kp2d': torch.rand(3, 64, 54, 2).cuda(),
        # 'person_centers':torch.rand(2,64,2).cuda()
    })

    print(outputs.keys())
    print(outputs['params_maps'].shape)
    print(outputs['center_map'].shape)

    print("ok!")

    # model = VOMP()
    # fake_data = torch.randn((8,32,128,128))
    #
    # output = model.head_forward(fake_data)
    # print(output['params_maps'].shape)
    # print(output['center_map'].shape)