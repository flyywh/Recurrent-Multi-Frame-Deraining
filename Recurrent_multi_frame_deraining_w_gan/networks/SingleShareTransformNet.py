## torch lib
import torch
import torch.nn as nn
import torch.nn.init as init

from networks.ConvLSTM import ConvLSTM

class SingleShareTransformNet(nn.Module):

    def __init__(self, opts, nc_in, nc_out, last_bias=0):
        super(SingleShareTransformNet, self).__init__()

        self.blocks = opts.blocks
        self.epoch = 0
        nf = opts.nf
        self.nf = nf
        #use_bias = (opts.norm == "IN")
        use_bias = True
        opts.norm = "None"
        
        ## convolution layers
        self.conv1 = ConvLayer(3 , nf, kernel_size=3, stride=1, bias=use_bias, norm=opts.norm) ## input: P_t, O_t-1
        self.res1 = ResidualBlock(nf, bias=use_bias, norm=opts.norm)
        self.conv2 = ConvLayer(nf, nf * 2, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        self.res2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.conv3 = ConvLayer(nf * 2, nf * 4, kernel_size=3, stride=2, bias=use_bias, norm=opts.norm)
        # Residual blocks
        self.ResBlocks = nn.ModuleList()
        for b in range(self.blocks):
            self.ResBlocks.append(ResidualBlock(nf * 4, bias=use_bias, norm=opts.norm))

        self.deconv3 = UpsampleConvLayer(nf * 8, nf * 2, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres2 = ResidualBlock(nf * 2, bias=use_bias, norm=opts.norm)
        self.deconv2 = UpsampleConvLayer(nf * 4, nf * 1, kernel_size=3, stride=1, upsample=2, bias=use_bias, norm=opts.norm)
        self.dres1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.deconv1 = ConvLayer(nf * 2, nf*5, kernel_size=3, stride=1)

        self.up_sample_1_2  = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_sample_1_4  = nn.Upsample(scale_factor=4, mode='bilinear')
        self.up_sample_1_8  = nn.Upsample(scale_factor=8, mode='bilinear')
        self.up_sample_1_16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.up_sample_1_32 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.alpha_rescale_1_2 = ConvLayer(nf, nf, kernel_size=1, stride=2)
        self.alpha_rescale_1_4 = ConvLayer(nf, nf, kernel_size=1, stride=4)
        self.alpha_rescale_1_8 = ConvLayer(nf, nf, kernel_size=1, stride=8)
        self.alpha_rescale_1_16 = ConvLayer(nf, nf, kernel_size=1, stride=16)
        self.alpha_rescale_1_32 = ConvLayer(nf, nf, kernel_size=1, stride=32)

        self.beta_rescale_1_2 = ConvLayer(nf, nf, kernel_size=1, stride=2)
        self.beta_rescale_1_4 = ConvLayer(nf, nf, kernel_size=1, stride=4)
        self.beta_rescale_1_8 = ConvLayer(nf, nf, kernel_size=1, stride=8)
        self.beta_rescale_1_16 = ConvLayer(nf, nf, kernel_size=1, stride=16)
        self.beta_rescale_1_32 = ConvLayer(nf, nf, kernel_size=1, stride=32)

        self.fluc_rescale_1_2 = ConvLayer(nf, nf, kernel_size=1, stride=2)
        self.fluc_rescale_1_4 = ConvLayer(nf, nf, kernel_size=1, stride=4)
        self.fluc_rescale_1_8 = ConvLayer(nf, nf, kernel_size=1, stride=8)
        self.fluc_rescale_1_16 = ConvLayer(nf, nf, kernel_size=1, stride=16)
        self.fluc_rescale_1_32 = ConvLayer(nf, nf, kernel_size=1, stride=32)

        self.alpha_decoder0 = ConvLayer(nf * 5, nf*1 , kernel_size=1, stride=1)
        self.alpha_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.alpha_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.alpha_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1, last_bias=0.85)

        self.beta_decoder0 = ConvLayer(nf * 5, nf * 1, kernel_size=1, stride=1)
        self.beta_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.beta_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.beta_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1, last_bias=0.75)

        self.streak_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.streak_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.streak_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.fluc_decoder0 = ConvLayer(nf * 5, nf * 1, kernel_size=1, stride=1)
        self.fluc_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.fluc_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.fluc_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.residual_decoder1 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.residual_decoder2 = ResidualBlock(nf * 1, bias=use_bias, norm=opts.norm)
        self.residual_output = ConvLayer(nf * 1, 3, kernel_size=1, stride=1)

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        
        E1 = self.res1(self.relu(self.conv1(X)))
        E2 = self.res2(self.relu(self.conv2(E1)))
        E3= self.relu(self.conv3(E2))

        RB = E3
        for b in range(self.blocks):
            RB = self.ResBlocks[b](RB)

        D3 = torch.cat((RB, E3), 1)
        D2 = self.dres2(self.relu(self.deconv3(D3)))
        D2 = torch.cat((D2, E2), 1)
        D1 = self.dres1(self.relu(self.deconv2(D2)))
        D0 = torch.cat((D1, E1), 1)
        D0 = self.relu(self.deconv1(D0))

        D0_alpha    = D0.narrow(1, 0, self.nf)
        D0_beta     = D0.narrow(1, self.nf, self.nf)
        D0_streak   = D0.narrow(1, self.nf*2, self.nf)
        D0_fluc     = D0.narrow(1, self.nf*3, self.nf)
        D0_residual = D0.narrow(1, self.nf*4, self.nf)

        D0_alpha_1_2  = self.up_sample_1_2(self.alpha_rescale_1_2(D0_alpha))
        D0_alpha_1_4  = self.up_sample_1_4(self.alpha_rescale_1_4(D0_alpha))
        D0_alpha_1_8  = self.up_sample_1_8(self.alpha_rescale_1_8(D0_alpha))
        D0_alpha_1_16 = self.up_sample_1_16(self.alpha_rescale_1_16(D0_alpha))
        D0_alpha_1_32 = self.up_sample_1_32(self.alpha_rescale_1_32(D0_alpha))

        D0_alpha = torch.cat((D0_alpha_1_2, D0_alpha_1_4, D0_alpha_1_8, D0_alpha_1_16, D0_alpha_1_32), 1)

        alpha    = self.alpha_output(self.alpha_decoder2(self.alpha_decoder1(self.relu(self.alpha_decoder0(D0_alpha)))))

        D0_beta_1_2  = self.up_sample_1_2(self.beta_rescale_1_2(D0_beta))
        D0_beta_1_4  = self.up_sample_1_4(self.beta_rescale_1_4(D0_beta))
        D0_beta_1_8  = self.up_sample_1_8(self.beta_rescale_1_8(D0_beta))
        D0_beta_1_16 = self.up_sample_1_16(self.beta_rescale_1_16(D0_beta))
        D0_beta_1_32 = self.up_sample_1_32(self.beta_rescale_1_32(D0_beta))

        D0_beta = torch.cat((D0_beta_1_2, D0_beta_1_4, D0_beta_1_8, D0_beta_1_16, D0_beta_1_32), 1)

        beta     = self.beta_output(self.beta_decoder2(self.beta_decoder1(self.relu(self.beta_decoder0(D0_beta)))))

        D0_fluc_1_2  = self.up_sample_1_2(self.fluc_rescale_1_2(D0_fluc))
        D0_fluc_1_4  = self.up_sample_1_4(self.fluc_rescale_1_4(D0_fluc))
        D0_fluc_1_8  = self.up_sample_1_8(self.fluc_rescale_1_8(D0_fluc))
        D0_fluc_1_16 = self.up_sample_1_16(self.fluc_rescale_1_16(D0_fluc))
        D0_fluc_1_32 = self.up_sample_1_32(self.fluc_rescale_1_32(D0_fluc))

        D0_fluc = torch.cat((D0_fluc_1_2, D0_fluc_1_4, D0_fluc_1_8, D0_fluc_1_16, D0_fluc_1_32), 1)
        fluc     = self.fluc_output(self.fluc_decoder2(self.fluc_decoder1(self.relu(self.beta_decoder0(D0_fluc)))))

        streak   = self.streak_output(self.streak_decoder2(self.streak_decoder1(D0_streak)))
        residual = self.residual_output(self.residual_decoder2(self.residual_decoder1(D0_residual)))

        return alpha, beta, fluc, streak, residual

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
        self.conv1  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.conv2  = ConvLayer(channels, channels, kernel_size=3, stride=1, bias=bias, norm=norm)
        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        input = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + input

        return out


