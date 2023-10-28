from .Degenerator import DM
from .Estimator import FCN
from .DGExtractor import Encoder,Decoder
from .Denoiser import Denoiser
import torch
import torch.nn as nn
from skimage.restoration import estimate_sigma

class DIBD(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(DIBD, self).__init__()
        sub_in = 1
        self.PrimaryNet = Denoiser(in_ch, out_ch)
        self.DualNet = DM(in_ch,out_ch)
        self.SigmaNet = FCN(sub_in,out_ch)
        self.Encoder = Encoder(sub_in,out_ch)
        self.Decoder = Decoder(sub_in,out_ch)

    def forward(self, noisy):
        device = noisy.device
        sigma = self.SigmaNet(noisy)

        mu, var = self.Encoder(noisy)

        eps = torch.randn(size=mu.shape).to(device)


        latent_rep = eps*torch.exp(var / 2.).to(device) + mu

        rec_noisy = self.Decoder(latent_rep)

        upsampler = torch.nn.Upsample(mode="bilinear", scale_factor=4 , align_corners=True).to(device)
        latent_rep = torch.squeeze(latent_rep,dim=1).to(device)
        latent_rep = upsampler(latent_rep)
        latent_rep = torch.unsqueeze(latent_rep,dim=1).to(device)

        clean = self.PrimaryNet(noisy,sigma,latent_rep)

        clean_sigma = torch.concat([clean,sigma],1).to(device)

        clean_sigma_latent = torch.concat([clean_sigma,latent_rep],1).to(device)

        est_noisy = self.DualNet(clean_sigma_latent)

        return clean, est_noisy, sigma, mu, var, rec_noisy
