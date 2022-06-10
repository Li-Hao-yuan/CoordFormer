from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import torch
import torch.nn as nn

root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
from config import args
from utils import print_dict
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast
from models.smpl import SMPL

BN_MOMENTUM = 0.1
default_cfg = {'mode':'val', 'calc_loss': False}
smpl = SMPL(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
            batch_size=args().batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False,\
            ).cuda()

class Base(nn.Module):
    def forward(self, meta_data, **cfg):

        if cfg['mode'] == 'matching_gts':
            return self.matching_forward(meta_data, **cfg)
        elif cfg['mode'] == 'parsing':
            return self.parsing_forward(meta_data, **cfg)
        elif cfg['mode'] == 'forward':
            return self.pure_forward(meta_data, **cfg)
        else:
            raise NotImplementedError('forward mode is not recognized! please set proper mode (parsing/matching_gts)')

    def matching_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.matching_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        if cfg['calc_loss']:
            outputs.update(self._calc_loss(outputs))
        #print_dict(outputs)
        return outputs

    @torch.no_grad()
    def parsing_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
                outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        else:
            outputs = self.feed_forward(meta_data)
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)

        outputs['meta_data'] = meta_data
        #print_dict(outputs)
        return outputs

    def feed_forward(self, meta_data):
        
        x = self.backbone(meta_data['image'].contiguous().cuda())
        
        if args().video:
            if args().predict:
                outputs = self.head_forward(x)
                
                return outputs
            else:
                # 需要做位移！！！
                T_params_dict = {}
                T_params_dict['poses'] = torch.zeros(1,72)
                T_params_dict['betas'] = torch.zeros(1,10)
                T_smpl_outs = smpl(**T_params_dict, return_verts=True)
                template_vertices = T_smpl_outs['verts'][0][::10]
                template_vertices = template_vertices.contiguous().view(689*3)


                params = meta_data['params'] # [batch_size,64,76]
                params = torch.cat( (params[:,:,:66],torch.zeros(params.shape[0],64,6).cuda(),params[:,:,66:]),dim=2 ) # add hand pose
                person_centers = meta_data["person_centers"] #[batch_size,64,2]
                
                dec_input = torch.zeros((params.shape[0],689*3,64,64)).cuda()
                person_centers = ((person_centers)+1/2 *64 ).to(torch.int)
                for batch_count,center_predict in enumerate(person_centers):
                    person_real_centers = center_predict[ center_predict[:,0]>0 ]
                    
                    for center_count,center in enumerate(person_real_centers):
                        [center_x,center_y] = center

                        params_dict = {}
                        params_dict['poses'] = params[batch_count,center_count,:72].unsqueeze(dim=0)
                        params_dict['betas'] = params[batch_count,center_count,72:].unsqueeze(dim=0)

                        smpl_outs = smpl(**params_dict, return_verts=True)
                        sample_smpl = smpl_outs['verts'][0][::10].contiguous().view(689*3)

                        dec_input[batch_count,:,center_x,center_y] = sample_smpl - template_vertices

                tem_dec_input = dec_input

                first_dec_input = torch.zeros(1,689*3,64,64).cuda()
                dec_input = torch.cat((first_dec_input,dec_input),dim=0)[:-1]


                # dec_input [batch_size,6890*3,64,64]
                outputs = self.head_forward(x,dec_input)
                outputs['real_vertice'] = tem_dec_input

                return outputs
        else:
            outputs = self.head_forward(x)
            return outputs

    @torch.no_grad()
    def pure_forward(self, meta_data, **cfg):
        if args().model_precision=='fp16':
            with autocast():
                outputs = self.feed_forward(meta_data)
        else:
            outputs = self.feed_forward(meta_data)
        return outputs

    def head_forward(self,x):
        return NotImplementedError

    def make_backbone(self):
        return NotImplementedError

    def backbone_forward(self, x):
        return NotImplementedError

    def _build_gpu_tracker(self):
        self.gpu_tracker = MemTracker()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)