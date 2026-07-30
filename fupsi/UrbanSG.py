import os
import torch
import torch.nn as nn
import math
import time

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(channels, channels, 3, 1, 1),
                      nn.BatchNorm2d(channels),
                      nn.PReLU(),
                      nn.Conv2d(channels, channels, 3, 1, 1),
                      nn.BatchNorm2d(channels)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2 # sum pooling
        out = self.upsample(out)
        return torch.div(x, out + self.epsilon)

class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)

class Atten_Block(nn.Module):
    def __init__(self, channel):
        super(Atten_Block, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel // 4, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel, channel // 4, 1, 1, 0)
        self.conv3 = nn.Conv2d(channel, channel // 4, 1, 1, 0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(channel // 4)
        self.conv4 = nn.Conv2d(channel // 4, channel, 1, 1, 0)
    def forward(self, x):
        # theta path
        theta = self.conv1(x)
        theta_ = theta.reshape(theta.shape[0], theta.shape[1], theta.shape[2] * theta.shape[3])#B,C/4,N

        # phi path
        phi = self.conv2(x)
        #phi = self.maxpool1(phi)#B,C/4,H/2,W/2
        phi_ = phi.reshape(phi.shape[0], phi.shape[1], phi.shape[2] * phi.shape[3])

        # Compute the attention-weight matrix.
        attn = torch.matmul(theta_.permute(0,2,1), phi_)
        attn_ = torch.softmax(attn, dim=-1)#B,N,N/4

        # g path
        g = self.conv3(x)#B,C/4,H,W
        #g = self.maxpool2(g)#B,C/4,H/2,W/2
        g_ = g.reshape(g.shape[0],g.shape[1],g.shape[2]*g.shape[3])#B,C/4,N/4

        attn_g = torch.matmul(g_, attn_.permute(0,2,1))
        attn_g = attn_g.reshape(theta.shape[0], theta.shape[1], theta.shape[2], theta.shape[3])
        out = self.conv4(attn_g)
        return x + out

class Generator(nn.Module):
    def __init__(self, scale_factor=4, n_residual_block=4, in_channel=1, base_channel=64, scaler_x=1500, scaler_y=100, ext_flag=True,residual_flag = True):
        super(Generator, self).__init__()
        self.ext_flag = ext_flag
        self.scaler_X = scaler_x
        self.scaler_Y = scaler_y

        if self.ext_flag:
            # Encode categorical day, hour, and weather attributes.
            self.embed_day = nn.Embedding(8, 2)  # ignore=0
            self.embed_hour = nn.Embedding(24, 3)
            self.embed_weather = nn.Embedding(18, 3)  # ignore=0
            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32*32),
                nn.ReLU(inplace=True)
            )
            in_channel = in_channel + 1
            base_channel = base_channel + 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 9, 1, 4),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, 3, 1, 1),
            nn.BatchNorm2d(base_channel)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channel, in_channel, 9, 1, 4),
            nn.PReLU()
        )
        self.residual_flag = residual_flag
        res_blocks = []
        if residual_flag:
            for _ in range(n_residual_block):
                res_blocks.append(ResidualBlock(base_channel))
        else:
            res_blocks.append(nn.Linear(base_channel,base_channel))
        self.res_blocks = nn.Sequential(*res_blocks)

        unsampling = []
        for _ in range(int(math.log(scale_factor, 2))):
            unsampling += [
                nn.Conv2d(base_channel, base_channel * 4, 3, 1, 1),
                nn.BatchNorm2d(base_channel * 4),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.unsampling = nn.Sequential(*unsampling)
        self.relu = nn.ReLU()
        self.den_softmax = N2_Normalization(scale_factor)
        self.recover = Recover_from_density(scale_factor)
        self.attn_block = Atten_Block(base_channel)

    def forward(self, x, ext):
        x_ = x
        if self.ext_flag:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, 32, 32)
            x_ = torch.cat([x, ext_out], dim=1)
        out1 = self.conv1(x_)
        if self.residual_flag:
            out2 = self.res_blocks(out1)
        else:
            out2 = self.res_blocks(out1.permute(0,2,3,1)).permute(0,3,1,2)
        out3 = self.conv2(out2)
        out4 = torch.add(out1, out3)
        out5 = self.unsampling(out4)
        #out5_ = self.attn_block(out5)
        out6 = self.conv3(out5)
        out6_ = self.relu(out6)
        out7 = self.den_softmax(out6_)
        out8 = self.recover(out7, x * self.scaler_X / self.scaler_Y)
        return out8

class Discriminator(nn.Module):
    def __init__(self, in_channel=1, ext_flag=True):
        super(Discriminator, self).__init__()
        self.ext_flag = ext_flag
        if self.ext_flag:
            self.embed_day = nn.Embedding(8, 2)  # ignore=0
            self.embed_hour = nn.Embedding(24, 3)
            self.embed_weather = nn.Embedding(18, 3)  # ignore=0
            self.ext2lr = nn.Sequential(
                nn.Linear(12, 128),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(128, 32 * 32),
                nn.ReLU(inplace=True)
            )
            in_channel = in_channel + 1
            self.ext2hr = nn.Sequential(
                nn.Conv2d(1,4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 4, 3, 1, 1),
                nn.BatchNorm2d(4),
                nn.PixelShuffle(upscale_factor=2),
                nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=3, padding=1),
            nn.PReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
    def forward(self, x, ext):
        batch_size = x.size(0)
        x= x.reshape(batch_size,-1,x.shape[-2],x.shape[-1])
        if self.ext_flag:
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            ext_out2 = self.embed_hour(
                ext[:, 5].long().view(-1, 1)).view(-1, 3)
            ext_out3 = self.embed_weather(
                ext[:, 6].long().view(-1, 1)).view(-1, 3)
            ext_out4 = ext[:, :4]
            ext_out = self.ext2lr(torch.cat(
                [ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, 32, 32)
            ext_out = self.ext2hr(ext_out)
            x = torch.cat([x, ext_out], dim=1)
        out1 = self.net(x)
        out2 = out1.view(batch_size)
        return torch.sigmoid(out2)
