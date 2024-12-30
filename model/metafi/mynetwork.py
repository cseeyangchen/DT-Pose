import xxsubtype

import torchvision
import torch.nn as nn
import torch
import cv2
from torchvision.transforms import Resize

from model.metafi.ChannelTrans import ChannelTransformer
import time


class metafinet(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self, num_keypoints, num_coor, num_person=1, dataset='mmfi-csi'):
        super(metafinet, self).__init__()
        self.num_keypoints = num_keypoints 
        self.num_coor = num_coor
        self.num_person = num_person
        self.diff = self.num_keypoints*self.num_person - 17
        self.dataset = dataset

        resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
        resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
        resnet_raw_model3 = torchvision.models.resnet34(pretrained=True)
        filters = [64, 64, 128, 256, 512]
        self.encoder_conv1_p1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_bn1_p1 = resnet_raw_model1.bn1
        self.encoder_relu_p1 = resnet_raw_model1.relu
        self.encoder_maxpool_p1 = resnet_raw_model1.maxpool
        self.encoder_layer1_p1 = resnet_raw_model1.layer1
        self.encoder_layer2_p1 = resnet_raw_model1.layer2
        self.encoder_layer3_p1 = resnet_raw_model1.layer3
        self.encoder_layer4_p1 = resnet_raw_model1.layer4

        self.tf = ChannelTransformer(vis=False, img_size=[self.num_keypoints*self.num_person, 12], channel_num=512, num_layers=1, num_heads=3, num_keypoints=self.num_keypoints*self.num_person)


        self.decode = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_coor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.num_coor),
            nn.ReLU(inplace=True)
        )



        self.bn1 = nn.BatchNorm1d(self.num_coor)
        self.bn2 = nn.BatchNorm2d(512)
        self.rl = nn.ReLU(inplace=True)



    def forward(self,x): #16,2,3,114,32
        b,_, _, _ = x.size()

        x = x.unsqueeze(1)
        x1 = x[:, :, 0:1, :, :] #1,114,32
        x2 = x[:, :, 1:2, :, :]
        x3 = x[:, :, 2:3, :, :]
        # print('x1',x1.size())
        # print('x2',x2.size())
        # print('x3',x3.size())

        x1 = torch.transpose(x1, 2, 3) #16,2,114,3,32
        x1 = torch.flatten(x1, 3, 4)# 16,2,114,96
        torch_resize = Resize([136+8*self.diff,32])
        x1 = torch_resize(x1) #136,32

        x2 = torch.transpose(x2, 2, 3)  # 16,2,114,3,32
        x2 = torch.flatten(x2, 3, 4)  # 16,2,114,96
        torch_resize = Resize([136+8*self.diff, 32])
        x2 = torch_resize(x2)

        # print(x3.size())
        x3 = torch.transpose(x3, 2, 3)  # 16,2,114,3,32
        x3 = torch.flatten(x3, 3, 4)  # 16,2,114,96
        torch_resize = Resize([136+8*self.diff, 32])  # metafi: [136, 32]
        x3 = torch_resize(x3)
        # print(x3.size())

        time_start = time.time()

        x1 = self.encoder_conv1_p1(x1)  ##16,2,136,136
        x1 = self.encoder_bn1_p1(x1)  ##size16,64,136,136
        x1 = self.encoder_relu_p1(x1)  ##size(1,64,192,624)

        x2 = self.encoder_conv1_p1(x2)  ##16,2,136,136
        x2 = self.encoder_bn1_p1(x2)  ##size16,64,136,136
        x2 = self.encoder_relu_p1(x2)  ##size(1,64,192,624)

        x3 = self.encoder_conv1_p1(x3)  ##16,2,136,136
        x3 = self.encoder_bn1_p1(x3)  ##size16,64,136,136
        x3 = self.encoder_relu_p1(x3)  ##size(1,64,192,624)

        #x = self.encoder_maxpool(x)
        # print(x1.size())
        x1 = self.encoder_layer1_p1(x1)
        x2 = self.encoder_layer1_p1(x2)
        x3 = self.encoder_layer1_p1(x3)
        # print(x1.size())

        x1 = self.encoder_layer2_p1(x1)
        x2 = self.encoder_layer2_p1(x2)
        x3 = self.encoder_layer2_p1(x3)
        # print(x1.size())

        x1 = self.encoder_layer3_p1(x1)
        x2 = self.encoder_layer3_p1(x2)
        x3 = self.encoder_layer3_p1(x3) # 256*34*8
        # print(x1.size())

        x1 = self.encoder_layer4_p1(x1)
        x2 = self.encoder_layer4_p1(x2)
        x3 = self.encoder_layer4_p1(x3)  # 256*34*8
        # print(x1.size())

        x = torch.cat([x1,x2,x3],dim=3)
        # print(x.size())

        x = self.bn2(x)
        # print(x.size())

        x, weight = self.tf(x)
        fea = x.mean(3).mean(2)

        x = self.decode(x)
        # print(x.size())

        m = torch.nn.AvgPool2d((1, 12), stride=(1, 1))
        x = m(x).squeeze(dim=3)
        x = self.bn1(x)
        # print(x.size())
        #x = self.rl(x)

        time_end = time.time()
        time_sum = time_end - time_start

        x = torch.transpose(x, 1, 2)
        if self.dataset == 'person-in-wifi-3d':
            x = x.view(b,self.num_person, self.num_keypoints, self.num_coor)
        else:
            x = x.view(b,self.num_keypoints, self.num_coor)


        return x,fea



def metafi_weights_init(m):
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.xavier_normal_(m.bias.data)
    # elif classname.find('BatchNorm2d') != -1:
    #     nn.init.normal_(m.weight.data, 1.0)
    #     nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
