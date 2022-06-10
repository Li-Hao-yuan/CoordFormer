from regex import P
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

################# 预定义函数 #################

number=['0','1','2','3','4','5','6','7','8','9']

# 生成随机四位数
def random_captcha_text(charset=number,captchasize=4):
    captchatext=[]
    for i in range(captchasize):
        c=random.choice(charset)
        captchatext.append(c)
    return captchatext

# 根据随机数生成图片
def gen_captcha_text_and_image():
    image=ImageCaptcha()

    captchatext=random_captcha_text()
    captcha=image.generate(''.join(captchatext)) #->>

    captchaimage=Image.open(captcha)
    captchaimage=np.array(captchaimage)
    return captchatext,captchaimage

# 生成一批数据
def gen_batch_data(batch_size):
    # 循环压入数据
    data = []
    label = []
    
    for _ in range(batch_size):
        text,image = gen_captcha_text_and_image() # text ['1','2','3','4'] , image [60,160,3]
        
        # text
        tem_data = []
        for text_count in range(4):
            tem_data.append(int(text[text_count]))
        
        # image
        image = convert2gray(image)
        image = cv2.resize(image,(64,160))
        
        # into
        label.append(tem_data) # [batch,4,10]
        data.append(image) # [bacth,64,160]
        
    return data,label
    

# 图像转灰度是有公式的 -> Gray = R*0.299 + G*0.587 + B*0.114
def convert2gray(img):
    if len(img.shape)>2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    else:
        return img
        
def test_for_data_gen():
    text,image = gen_captcha_text_and_image()
    print(text)
    cv2.imshow("image",convert2gray(image))
    cv2.waitKey(0)            

################# 定义模型 #################
EPOCH = 10
BATCH_SIZE = 64
LR = 0.001          # 学习率

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 160)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters,滤波器的个数
                kernel_size=5,      # filter size,长宽
                stride=1,           # filter movement/step,步长
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                #围上一圈零
            ),      # output shape (16, 64, 160)
            #16来自滤波器个数,padding使得长宽不变
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 32, 80)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 32, 80)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 16, 40)
        )
        self.conv3 = nn.Sequential(  # input shape (16, 16, 40)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (32, 16, 40)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 8, 20)
        )
        self.conv4 = nn.Sequential(  # input shape (64, 8, 20)
            nn.Conv2d(64, 128, 3, 1, 1),  # output shape (128, 8, 20)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (128, 4, 10)
        )
        self.out = nn.Linear(128 * 4 * 10, 40)   # fully connected layer, output 10 classes 

    def forward(self, x):
        '''
        x : [batch_size,1,64,160]
        '''
        x = self.conv1(x)
        x = self.conv2(x)    #保留batch纬度,其次全部变成一起
        x = self.conv3(x)    
        x = self.conv4(x)    
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 128 * 4 * 20)
        output = self.out(x)  # output : [batch_size,10] 一个数字
        return output



if __name__=='__main__':
    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    
    
    accuracy_on_data = [] # 统计正确率

    # training and testing
    for epoch in tqdm(range(EPOCH)):
        cnn.train()
        data, label = gen_batch_data(BATCH_SIZE)   # 分配 batch data, normalize x when iterate train_loader
        data = torch.tensor(data).to(torch.float32).unsqueeze(1)
        label = torch.tensor(label)
        
        output = cnn(data)               # cnn output
        
        loss = loss_func(output[:,:10], label[:,0] ) + loss_func(output[:,10:20],label[:,1] )+ \
            loss_func(output[:,20:30], label[:,2]) + loss_func(output[:,30:40],label[:,3])# cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        # print("mse loss:%.2f"%(loss.data.numpy()))

        # 测试
        if (epoch+1) % 2 == 0:
            cnn.eval()
            with torch.no_grad():
                tem_acc = []
                test_data,test_label = gen_batch_data(BATCH_SIZE)
                
                test_data = torch.tensor(test_data).to(torch.float32).unsqueeze(1)
                test_label = torch.tensor(test_label)
                
                # 过了一道 softmax 的激励函数后的最大概率才是预测值
                output = cnn(test_data)
                
                # 由于是四个数字，因此需要循环一下
                for test_count in range(4):
                    prediction = torch.max(F.softmax( output[:,test_count*10: (test_count+1)*10 ] ), 1)[1]
                    pred_y = prediction.data.numpy().squeeze()
                    # print("pred:",pred_y)
                    
                    target_y = test_label[:,test_count].data.numpy()
                    # print("tar:",target_y)
                    
                    accuracy = sum(pred_y == target_y)/len(target_y)  # 预测中有多少和真实值一样

                    tem_acc.append(accuracy)
                accuracy_on_data.append( sum(tem_acc)/len(tem_acc) )

    torch.save(cnn,"cnn.pth")

    print("最高正确率:",max(accuracy_on_data))
    plt.plot([i for i in range(len(accuracy_on_data))],accuracy_on_data,'r-')
    plt.show()
    

