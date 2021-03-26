## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init
from numpy import *

from networks.ConvLSTM import ConvLSTM

def To3D(E1, group):
    [b, c, h, w] = E1.shape
    nf = int(c/group)

    E_list = []
    for i in range(0, group):
        tmp = E1[:, nf*i:nf*(i+1), :, :]
        tmp = tmp.view(b, nf, 1, h, w)
        E_list.append(tmp)
        
    E1_3d = torch.cat(E_list, 2)
    return E1_3d

def To2D(E1_3d):
    [b, c, g, h, w] = E1_3d.shape

    E_list = []
    for i in range(0, g):
        tmp = E1_3d[:, :, i, :, :]
        tmp = tmp.view(b, c, h, w)
        E_list.append(tmp)

    E1 = torch.cat(E_list, 1)
    return E1

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n/2))
            if m.bias is not None:
                m.bias.data.zero_()


class MultiShareTransformNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out, init_tag):
        super(MultiShareTransformNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        self.nf = nf
        use_bias = True
        opts.norm = "None"
        
        self.conv1 = ConvLayer(15, nf*5, kernel_size=3, stride=1, groups=5, bias=use_bias, norm=opts.norm)
        self.res1 = ResidualBlock(nf*5, groups=5, bias=use_bias, norm=opts.norm)

        self.conv2 = nn.Conv3d(nf, nf*2, kernel_size=(2, 3, 3), stride=(1,2,2), padding=(0,1,1), bias=use_bias)
        self.res2 = ResidualBlock(nf * 2*4, groups=4, bias=use_bias, norm=opts.norm)

        self.conv3 = nn.Conv3d(nf*2, nf*2, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.res3 = ResidualBlock(nf * 2*3, groups=3, bias=use_bias, norm=opts.norm)

        self.conv4 = nn.Conv3d(nf*2, nf*4, kernel_size=(2, 3, 3), stride=(1,2,2), padding=(0,1,1), bias=use_bias)
        self.res4 = ResidualBlock(nf * 4*2, groups=2, bias=use_bias, norm=opts.norm)

        self.conv5 = nn.Conv3d(nf*4, nf*4, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.res5 = ResidualBlock(nf * 4*1, groups=1, bias=use_bias, norm=opts.norm)


        self.du_conv1 = ConvLayer(15, nf*5, kernel_size=3, stride=1, groups=5, bias=use_bias, norm=opts.norm)
        self.du_res1 = ResidualBlock(nf*5, groups=5, bias=use_bias, norm=opts.norm)

        self.du_conv2 = nn.Conv3d(nf, nf*2, kernel_size=(2, 3, 3), stride=(1,2,2), padding=(0,1,1), bias=use_bias)
        self.du_res2 = ResidualBlock(nf * 2*4, groups=4, bias=use_bias, norm=opts.norm)

        self.du_conv3 = nn.Conv3d(nf*2, nf*2, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.du_res3 = ResidualBlock(nf * 2*3, groups=3, bias=use_bias, norm=opts.norm)

        self.du_conv4 = nn.Conv3d(nf*2, nf*4, kernel_size=(2, 3, 3), stride=(1,2,2), padding=(0,1,1), bias=use_bias)
        self.du_res4 = ResidualBlock(nf * 4*2, groups=2, bias=use_bias, norm=opts.norm)

        self.du_conv5 = nn.Conv3d(nf*4, nf*4, kernel_size=(2, 3, 3), stride=(1,1,1), padding=(0,1,1), bias=use_bias)
        self.du_res5 = ResidualBlock(nf * 4*1, groups=1, bias=use_bias, norm=opts.norm)

        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        self.convlstm = ConvLSTM(input_size=nf * 4, hidden_size = nf * 4, kernel_size=3)

        self.deconv3 = UpsampleConvLayer(nf * 4, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 2, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.deconv1 = ConvLayer(nf * 1, nf, kernel_size=3, stride=1)

        self.output1 = ConvLayer(nf * 1, nf * 1, kernel_size=3, stride=1)
        self.output2 = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.alpha_ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.alpha_ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        self.alpha_convlstm = ConvLSTM(input_size=nf * 4, hidden_size = nf * 4, kernel_size=3)

        self.alpha_deconv3 = UpsampleConvLayer(nf * 4, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.alpha_dres2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.alpha_deconv2 = UpsampleConvLayer(nf * 2, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.alpha_dres1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.alpha_deconv1 = ConvLayer(nf * 1, nf, kernel_size=3, stride=1)

        self.alpha_output1 = ConvLayer(nf * 1, nf * 1, kernel_size=3, stride=1)
        self.alpha_output2 = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights(self)

        if init_tag==1:
            nn.init.constant_(self.output2.conv2d.bias.data, 0.5) 
            nn.init.constant_(self.output2.conv2d.weight.data, 0)

            nn.init.constant_(self.alpha_output2.conv2d.bias.data, 1)
            nn.init.constant_(self.alpha_output2.conv2d.weight.data, 0)
 
    def forward(self, X, prev_state, prev_state_alpha):
        X1 = torch.cat((X[:, :6, :, :], X[:, 9:, :, :]), 1)
        X2 = torch.cat((X[:, :9, :, :], X[:, 12:, :, :]), 1)

        E1 = self.res1(self.relu(self.conv1(X1)))

        E1_3d = To3D(E1, 5)
        E2_3d = self.conv2(E1_3d)
        E2 = To2D(E2_3d)
        E2 = self.res2(self.relu(E2))

        E2_3d = To3D(E2, 4)
        E3_3d = self.conv3(E2_3d)
        E3 = To2D(E3_3d)
        E3 = self.res3(self.relu(E3))

        E3_3d = To3D(E3, 3)
        E4_3d = self.conv4(E3_3d)
        E4 = To2D(E4_3d)
        E4 = self.res4(self.relu(E4))

        E4_3d = To3D(E4, 2)
        E5_3d = self.conv5(E4_3d)
        E5 = To2D(E5_3d)
        E5 = self.res5(self.relu(E5))


        F1 = self.du_res1(self.relu(self.du_conv1(X2)))

        F1_3d = To3D(F1, 5)
        F2_3d = self.du_conv2(F1_3d)
        F2 = To2D(F2_3d)
        F2 = self.du_res2(self.relu(F2))

        F2_3d = To3D(F2, 4)
        F3_3d = self.du_conv3(F2_3d)
        F3 = To2D(F3_3d)
        F3 = self.du_res3(self.relu(F3))

        F3_3d = To3D(F3, 3)
        F4_3d = self.du_conv4(F3_3d)
        F4 = To2D(F4_3d)
        F4 = self.du_res4(self.relu(F4))

        F4_3d = To3D(F4, 2)
        F5_3d = self.du_conv5(F4_3d)
        F5 = To2D(F5_3d)
        F5 = self.du_res5(self.relu(F5))

        RB = E5 + F5

        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        state = self.convlstm(RB, prev_state)

        D3 = RB + state[0]

        D2 = self.dres2(self.relu(self.deconv3(D3+E5)))
        D1 = self.dres1(self.relu(self.deconv2(D2+E3[:,64:128,:,:])))
        D0 = self.relu(self.deconv1(D1+E1[:,64:96,:,:]))

        haze = self.output2(self.relu(self.output1(D0)))

        RB_alpha = E5 + F5
        for b in range(self.blocks):
            RB_alpha = self.alpha_ResBlocks[b](RB_alpha)

        state_alpha = self.alpha_convlstm(RB_alpha, prev_state_alpha)

        D3_alpha = RB_alpha + state_alpha[0]

        D2_alpha = self.alpha_dres2(self.relu(self.alpha_deconv3(D3_alpha)))
        D1_alpha = self.alpha_dres1(self.relu(self.alpha_deconv2(D2_alpha)))
        D0_alpha = self.relu(self.alpha_deconv1(D1_alpha))

        alpha = self.alpha_output2(self.relu(self.alpha_output1(D0_alpha)))

        return haze, alpha, D0, D0_alpha, state, state_alpha 


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        if last_bias!=0:
            init.constant(self.conv2d.weight, 0)
            init.constant(self.conv2d.bias, last_bias)

    def forward(self, x):
        out = self.conv2d(x)

        return out


class UpsampleConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        return out

class ResidualBlock(nn.Module):
    
    def __init__(self, channels, groups=1, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, groups=groups, bias=bias, norm=norm)

        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = out + input

        return out


