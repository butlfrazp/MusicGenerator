import torch
from torch.nn.utils import spectral_norm

from config import (
    latent_dim,
    n_tracks,
    n_measures,
    n_pitches,
    measure_resolution
)
from models.attention.attention import Self_Attn

class GeneraterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = spectral_norm(torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride))
    
    def forward(self, x):
        x = self.transconv(x)
        return torch.nn.functional.relu(x)

class Generator(torch.nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(n_tracks)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(n_tracks)
        ])

        self.attn1 = Self_Attn(64, 'relu')
        self.attn2 = Self_Attn(32, 'relu')
        self.attn3 = Self_Attn(16, 'relu')
        self.attn4 = Self_Attn(5, 'relu')

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x, _ = self.attn1(x)
        x = self.transconv3(x)
        x, _ = self.attn2(x)
        x = [self.attn3(transconv(x))[0] for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x, _ = self.attn4(x)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x

if __name__ == "__main__":
    latent_vector = torch.randn((16, latent_dim))
    generator = Generator()
    x = generator(latent_vector)
    print(x.shape)