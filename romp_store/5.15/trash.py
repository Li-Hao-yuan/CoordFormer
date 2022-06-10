import torch

a = torch.rand(3,8,256,256)
b = torch.rand(3,8,256,256)

c = a * b
print(a.shape)