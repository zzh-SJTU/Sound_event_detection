
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)
#卷积模块，包括卷积层，RELU激活函数，最大池化，2维批归一化
class conv_block(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_in,
                      channel_out,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.BatchNorm2d(channel_out)
            )
    def forward(self, x):
        return self.block(x)
   
class Crnn(nn.Module):
    def __init__(self, num_freq, class_num, dropout = 0.5, width = 512 , num_of_layers = 1):
        super().__init__()
        self.bn_1d = nn.BatchNorm1d(num_freq)
        self.conv_1 = conv_block(1,16)
        self.conv_2 = conv_block(16,32)
        self.conv_3 = conv_block(32,64)
        self.conv_4 = conv_block(64,128)
        #过三个卷积层，每个卷积层后接dropout层
        self.conv_all = nn.Sequential(self.conv_1,nn.Dropout(dropout),self.conv_2,nn.Dropout(dropout),self.conv_3,nn.Dropout(dropout))
        #循环神经网络（LSTM）
        self.gru = nn.LSTM(512,width,num_layers=num_of_layers,bidirectional=True,batch_first=True)
        #线性层，将循环神经网络的隐层输出映射到类别总数
        self.linear = nn.Linear(2*width, class_num)

    def detection(self, x):
        #[32, 501, 64]
        t = x.shape[1]
        x = x.transpose(1, 2).contiguous() # torch.Size([32, 64, 501])
        x = self.bn_1d(x) # torch.Size([32, 64, 501])
        x = x.unsqueeze(1) # torch.Size([32, 1, 64, 501])
        x = self.conv_all(x) # torch.Size([32, 64, 8, 62])
        x = x.transpose(1, 3).contiguous() #torch.Size([32, 62, 8, 64])
        x = x.flatten(-2) #torch.Size([32, 62, 512])
        x,_ = self.gru(x) #torch.Size([32, 62, 128])
        x = self.linear(x) # torch.Size([32, 62, 10])
        x = x.transpose(1, 2).contiguous() #torch.Size([32, 10, 62])
        #经过循环神经网络之后需要恢复时间维度（通过插值操作完成）
        x = F.interpolate(x,size = (t),mode = 'nearest') #torch.Size([32, 10, 501])
        x = x.transpose(1, 2).contiguous()#torch.Size([32, 501, 10])
        #通过sigmoid函数将输出映射到[0,1]作为概率
        frame_wise_prob = torch.sigmoid(x).clamp(1e-7, 1.)
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size, time_steps, num_freq]
        # Return:
        #     frame_wise_prob: [batch_size, time_steps, class_num]
        ##############################
        return frame_wise_prob
        
    def forward(self, x): 
        frame_wise_prob = self.detection(x)
        clip_prob = linear_softmax_pooling(frame_wise_prob)
        '''(samples_num, feature_maps)'''
        return {
            'clip_probs': clip_prob,
            'time_probs': frame_wise_prob
        }