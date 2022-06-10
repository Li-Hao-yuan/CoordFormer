import torch
import torch.nn as nn

class ParamLayer(nn.Module):
    def __init__(self,in_channel,out_channel,sample):
        super(ParamLayer,self).__init__()
        self.sample = sample
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,5,1,2),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)

    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out += residual

        out = self.conv2(x)

        out = self.relu(out)
        out = self.sample(out)

        return out

class ParamEncoder(nn.Module):
    # [batch_size,145,64,64] - > [batch_size,1,512,512]
    def __init__(self):
        downsample = nn.MaxPool2d(2)
        super(ParamEncoder,self).__init__()
        self.layers = nn.ModuleList([
            ParamLayer(145,64,downsample),
            ParamLayer(64,8,downsample),
            ParamLayer(8,1,downsample)
            ])

    def forward(self,x):
        for layer in self.layers:
            x= layer(x)
        return x

class ParamDecoder(nn.Module):
    # [batch_size,1,512,512] -> [batch_size,145,64,64]
    def __init__(self):
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        super(ParamDecoder,self).__init__()
        self.layers = nn.ModuleList([
            ParamLayer(1,8,upsample),
            ParamLayer(8,64,upsample),
            ParamLayer(64,145,upsample)
            ])

    def forward(self,x):
        for layer in self.layers:
            x= layer(x)
        return x
    
if __name__ == "__main__":
    encoder = ParamEncoder()
    data = torch.randn((2,145,64,64))
    print( encoder(data).shape )