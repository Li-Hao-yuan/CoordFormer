import torch
from lib.models.build import build_model

model = build_model().cuda()
outputs=model.feed_forward({
    'image':torch.rand(1,512,512,3).cuda(),
    'params':torch.rand(1,64,76).cuda(),
    'person_centers':torch.rand(1,64,2).cuda()
    })

print("ok!")