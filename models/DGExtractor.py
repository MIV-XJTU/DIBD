import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.restoration import estimate_sigma

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            # nn.InstanceNorm3d(in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Encoder(nn.Module):
    def __init__(self, in_channel=2,out_channel = 1):
        super(Encoder, self).__init__()


        n1 = 32
        self.Conv1 = conv_block(in_channel, n1)
        self.Conv2 = conv_block(n1, n1)
        self.Conv3 = conv_block(n1, n1)
        self.Conv4 = conv_block(n1, n1)
        self.mu = nn.Conv3d(in_channels=n1,out_channels=out_channel, kernel_size=3,padding=1)
        self.sigma = nn.Conv3d(in_channels=n1,out_channels=out_channel, kernel_size=3,padding=1)

        self.Down1 = nn.Conv3d(n1, n1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0, bias=True)
        self.Down2 = nn.Conv3d(n1, n1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0, bias=True)
        self.encoder = nn.Sequential(
            self.Conv1,
            self.Down1,
            self.Conv2,
            self.Down2,
            self.Conv3

        )


    def forward(self,x):

        self.n1 = self.encoder(x)

        self.n2 = self.mu(self.n1)

        self.n3 = self.sigma(self.n1)

        return self.n2,self.n3


class Decoder(nn.Module):
    def __init__(self,in_channel = 1,out_channel = 1):
        super(Decoder, self).__init__()

        n1 = 32
        self.Conv1 = conv_block(n1, out_channel)
        self.Conv2 = conv_block(n1, n1)
        self.Conv3 = conv_block(n1, n1)
        self.Conv4 = conv_block(in_channel, n1)

        self.Up1 = nn.ConvTranspose3d(n1, n1, kernel_size=[1, 2, 2],stride=[1, 2, 2], padding=0, bias=True)
        self.Up2= nn.ConvTranspose3d(n1, n1, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0, bias=True)

        self.decoder = nn.Sequential(
            self.Conv4,
            self.Up2,
            self.Conv2,
            self.Up1,
            self.Conv1
        )





    def forward(self, x):
        output = self.decoder(x)

        return output

class vae(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(vae, self).__init__()
        self.Encoder = Encoder(in_ch, out_ch)
        self.Decoder = Decoder(in_ch, out_ch)

    def forward(self, noisy, gpu=True):


        mu, var = self.Encoder(noisy)

        eps = torch.randn(size=mu.shape)
        if gpu:
            # est_sigma = est_sigma.cuda()
            mu = mu.cuda()
            var = var.cuda()
            eps = eps.cuda()

        latent_rep = eps*torch.exp(var / 2.) + mu



        rec_noisy = self.Decoder(latent_rep)

        upsampler = torch.nn.Upsample(mode="bilinear", scale_factor=4 , align_corners=True)
        latent_rep = torch.squeeze(latent_rep,dim=1)
        latent_rep = upsampler(latent_rep)
        latent_rep = torch.unsqueeze(latent_rep,dim=1)
        return   mu, var, rec_noisy,latent_rep