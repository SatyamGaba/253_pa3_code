# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:09:01 2020

@author: yadav
"""

import torch.nn as nn
import torch

class convchain(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(convchain,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self,x):
        x = self.conv(x)
        return x
    
class upconv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upconv,self).__init__()
        self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in , ch_out , kernel_size=2, stride=2),
#                nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True),
#                nn.Conv2d(ch_in , ch_out , kernel_size=1, stride=1),
		    nn.BatchNorm2d(ch_out),
#			nn.ReLU(inplace=True)
            
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UNET(nn.Module): 

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv1 = convchain(ch_in = 3, ch_out = 32)
        self.conv2 = convchain(ch_in = 32, ch_out = 64)
        self.conv3 = convchain(ch_in = 64, ch_out = 128)
        self.conv4 = convchain(ch_in = 128, ch_out = 256)
        self.conv5 = convchain(ch_in = 256, ch_out = 512)
        
        self.up5 = upconv(ch_in = 512, ch_out = 256)
        self.upconv5 = convchain(ch_in = 512, ch_out = 256)
        
        self.up4 = upconv(ch_in = 256, ch_out = 128)
        self.upconv4 = convchain(ch_in = 256, ch_out = 128)
        
        self.up3 = upconv(ch_in = 128, ch_out = 64)
        self.upconv3 = convchain(ch_in = 128, ch_out = 64)
        
        self.up2 = upconv(ch_in = 64, ch_out = 32)
        self.upconv2 = convchain(ch_in = 64, ch_out = 34)
        
        self.conv_1 = nn.Conv2d(34, n_class ,kernel_size=1,stride=1,padding=0)
        

    def forward(self,x):
        # encoding path
        x1 = self.conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat path
        d5 = self.up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.upconv5(d5)
                
        d4 = self.up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.upconv4(d4)
        
        

        d3 = self.up3(d4) #original was d4
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.upconv3(d3)
        
        del x2

        d2 = self.up2(d3)
        print (d2.size(), x1.size())
        d2 = torch.cat((x1,d2),dim=1)
        print (x1.size())
        d2 = self.upconv2(d2)
        
        del x1, d3, x3, d4, d5, x5, x4
        d1 = self.conv_1(d2)

        return d1
