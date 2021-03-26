## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from networks.ConvLSTM import ConvLSTM


class SingleScaleTransformNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out, last_bias=0):
        super(SingleScaleTransformNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        # use_bias = (opts.norm == "IN")
        use_bias = True
        opts.norm = "None"

        ## convolution layers
        self.conv1 = ConvLayer(3, nf * 2, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm)
        self.conv2 = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.conv3 = ConvLayer(nf * 4, nf * 8, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)

        # Residual blocks
        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 8, bias=use_bias, norm=opts.norm))

        self.deconv1 = UpsampleConvLayer(nf * 8, nf * 4, kernel_size=3, stride=1, upsample=2, bias=use_bias,
                                         norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 8, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias,
                                         norm=opts.norm)
        self.deconv3 = ConvLayer(nf * 4, 3, kernel_size=3, stride=1, last_bias = last_bias)

        # self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):

        E1 = self.relu(self.conv1(X))
        E2 = self.relu(self.conv2(E1))
        E3 = self.relu(self.conv3(E2))

        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        D2 = self.relu(self.deconv1(RB))
        C2 = torch.cat((D2, E2), 1)
        D1 = self.relu(self.deconv2(C2))
        C1 = torch.cat((D1, E1), 1)
        D0 = self.deconv3(C1)

        return D0


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=None, bias=True, last_bias=0):
        super(ConvLayer, self).__init__()

        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
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

    def __init__(self, channels, norm=None, bias=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        # self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out


