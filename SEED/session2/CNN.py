from torch import nn
from torch.nn import functional as F
import torch

class ch_atten(nn.Module): # 加上效果反而不好
    def __init__(self, input_channels, infactor=2):
        super(ch_atten, self).__init__()
        #  通道注意力，增强正向特征，抑制负面特征
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_generator = nn.Sequential(
            nn.Linear(input_channels, input_channels//infactor),  
            nn.ReLU(),
            nn.Linear(input_channels//infactor, input_channels),  
            nn.Softmax(dim=1)  # 通过Softmax生成1D权重
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        weight = self.weight_generator(y).view(b, c, 1, 1)
        x = x * weight
        
        return x


class cnnBlcok(nn.Module):
    def __init__(self, in_channels=5, out_channels=45, is_change_scale=True):
        super(cnnBlcok, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=2 , padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 先不要激活函数，减少模型的复杂度-》奥卡姆剃刀原则
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels+in_channels, out_channels, kernel_size=(3,3), stride=2 , padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 先不要激活函数，减少模型的复杂度-》奥卡姆剃刀原则
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

        # self.ch_atten = ch_atten(out_channels)
        self.downCh = nn.Conv2d(out_channels*2, out_channels, kernel_size=(1,1), stride=1, padding=0) #3维卷积融合各通道不同尺度的 特征
        # self.ch_atten = ch_atten(input_channels=out_channels, infactor=2)
        
        self.fc = nn.Linear(out_channels*5*5, 4)

    def forward(self, data):
        # keys = list(data.keys())
        # # print(keys)
        # x = data[keys[5]].cuda()
        # print(x.shape)

        x = data
        # layer0
        resx = F.avg_pool2d(x, kernel_size=2)
        conx =  self.conv0(x)
        x = torch.cat((conx, resx), dim=1)

        # layer1
        resx = F.avg_pool2d(conx, kernel_size=2)
        conx =  self.conv1(x)
        x = torch.cat((conx, resx), dim=1)

        # layer2
        resx = F.avg_pool2d(conx, kernel_size=2)
        conx =  self.conv2(x)
        x = torch.cat((conx, resx), dim=1)
        # print(x.shape)

        # layer3
        resx = F.avg_pool2d(conx, kernel_size=2)
        conx =  self.conv3(x)
        x = torch.cat((conx, resx), dim=1)

        x = self.downCh(x)
        # # print(x.shape)
        # x = self.fc(x.view(x.shape[0],-1))
        return x

# 直接全连接
# class cnnBlcok(nn.Module):
#     def __init__(self, in_channels=5, out_channels=[2,4,8], is_change_scale=True):
#         super(cnnBlcok, self).__init__()

#         self.conv0 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels*out_channels[0], kernel_size=(3,3), stride=1 , padding=1, bias=False),
#             nn.BatchNorm2d(in_channels*out_channels[0]),
#             nn.ReLU(inplace=True), # 先不要激活函数，减少模型的复杂度-》奥卡姆剃刀原则
#             )

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels*out_channels[0], in_channels*out_channels[1], kernel_size=(3,3), stride=1 , padding=1, bias=False),
#             nn.BatchNorm2d(in_channels*out_channels[1]),
#             nn.ReLU(inplace=True), # 先不要激活函数，减少模型的复杂度-》奥卡姆剃刀原则
#             )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels*out_channels[1], in_channels*out_channels[2], kernel_size=(3,3), stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(in_channels*out_channels[2]),
#             nn.ReLU(inplace=True)
#             )

#         # self.ch_atten = ch_atten(out_channels)
#         self.fc = nn.Linear(in_channels*out_channels[2]*5*5, 4)

#     def forward(self, data):
#         keys = list(data.keys())
#         # print(keys)
#         x = data[keys[5]].cuda()

#         # layer0
#         conx =  self.conv0(x)

#         # layer1
#         conx =  self.conv1(x)

#         # layer2
#         conx =  self.conv2(x)
#         # print(x.shape)

#         # print(x.shape)
#         x = self.fc(x.view(x.shape[0],-1))
#         return x


if __name__ == '__main__':

    net = cnnBlcok().cuda()
    from data_input import getloader
    SEEDIV_data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth/"

    data_loader = getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=True, is_drop_last=False, is_shuffle=True)
    
    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        y = net(data)
        print(y.shape)
        break