import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoiser(nn.Module):
    def __init__(self,in_nc=1,out_nc=1):
        super(Denoiser, self).__init__()
        self.main = MainNet(in_nc, out_nc)
        self.out = nn.Conv3d(out_nc*2, out_nc, kernel_size=3, padding=1, bias=True)

    def forward(self, x,sigma=None,latent_rep=None):
        input = x
        if sigma is not None:
            input = torch.concat([input,sigma],1)
        if latent_rep is not None:
            input = torch.concat([latent_rep,input],1)
        out1 = self.main(input) + x
        cat1 = torch.cat([x, out1], dim=1)
        return self.out(cat1) + x

class MainNet(nn.Module):
    """B-DenseUNets"""
    def __init__(self, in_nc=12, out_nc=12):
        super(MainNet, self).__init__()
        lay=2
        self.inc = nn.Sequential(
            single_conv(in_nc, 64),
            single_conv(64, 64),
        )
        self.down1 = nn.Conv3d(64, 64,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            RDB(128, lay, 32),
        )
        self.down2 = nn.Conv3d(128, 128,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            RDB(256, lay+1, 32),
        )
        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            RDB(128, lay+1, 32),
        )


        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            RDB(64, lay, 32),
        )

        self.outc = outconv(64, out_nc)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)

        conv1 = self.conv1(down1)


        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)


        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)


        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, (1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DownBlock(nn.Module):
    def __init__(self, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = 0.2

        if nFeat is None:
            nFeat = 64

        if in_channels is None:
            in_channels = 1

        if out_channels is None:
            out_channels = 1

        dual_block = [
            nn.Sequential(
                nn.Conv3d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        dual_block.append(nn.Conv3d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x