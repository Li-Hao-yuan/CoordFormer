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

class VOMP(Base):
    def __init__(self, backbone=None,**kwargs):
        super(VOMP,self).__init__()
        self.backbone = backbone
        # self.hrnet = HigherResolutionNet()
        self._result_parser = ResultParser()
        self.trans = Transformer()
        
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def translate_dec_output(self,dec_output,camera_map):
        # dec_output [batch_size,64,142+2+1]
        # camera_map = [batch_size,3,64,64]
        param_output = torch.zeros(dec_output.shape[0],142,64,64).cuda()
        center_map = torch.zeros(dec_output.shape[0],1,64,64).cuda()

        for item_count,item in enumerate(dec_output):
            # into one batch : [64,145]
            param_confidence = item[ item[:,-1]>args().paramsmap_conf_thresh ]
            for prediction in param_confidence:
                # into one prediction : [132+10+2+1]
                [x,y] = (prediction[-3:-1]*63).to(torch.int)
                
                param_output[item_count,:,x,y] = prediction[:142]
                center_map[item_count,0,x,y] = 1

        param_output = torch.cat([camera_map, param_output], 1)

        return param_output,center_map

    def head_forward(self,out,params_map):
        '''
        x : [batch_size,32,128,128]
        params_map : [batch_size,145,64,64]
        '''
        # dec_output [batch_size,64,145]
        # camera_map = [batch_size,3,64,64]
        dec_output,camera_map = self.trans(out,params_map)
        param_output,centermap = self.translate_dec_output(dec_output.cuda(),camera_map.cuda())

        output = {'params_maps':param_output.float(), 'center_map':centermap.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()

        return output


if __name__ == "__main__":
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({
        'image':torch.rand(1,512,512,3).cuda(),
        'params':torch.rand(1,64,76).cuda(),
        'person_centers':torch.rand(1,64,2).cuda()
        })

    print("ok!")