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
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv_residual = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.conv(x) + self.conv_residual(x)
        return x
class DM(nn.Module):
    def __init__(self,in_channel = 1,out_channel = 1):
        super(DM, self).__init__()
        n_channel = 32
        self.Conv1 = conv_block(in_channel, n_channel)
        self.Conv2 = conv_block(n_channel, out_channel)


        self.dm = nn.Sequential(
            self.Conv1,
            self.Conv2,
        )

    def forward(self, x,):

        return self.dm(x)