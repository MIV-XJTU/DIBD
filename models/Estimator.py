import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            # nn.InstanceNorm3d(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FCN(nn.Module):
    def __init__(self,in_channel = 1,out_channel = 1):
        super(FCN, self).__init__()
        n_channel = 32
        self.Conv1 = conv_block(in_channel, n_channel)
        self.Conv2 = conv_block(n_channel, n_channel)
        self.Conv3 = conv_block(n_channel, n_channel)
        self.Conv4 = conv_block(n_channel, n_channel)
        self.Conv5 = conv_block(n_channel, out_channel)
        self.fcn = nn.Sequential(
            self.Conv1,
            self.Conv2,
            self.Conv3,
            self.Conv4,
            self.Conv5,
        )

    def forward(self, x):
        return self.fcn(x)