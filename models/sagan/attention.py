import torch
import torch.nn as nn
import numpy as np

from config import (
    latent_dim,
    n_tracks,
    n_measures,
    n_pitches,
    measure_resolution
)
from utils.data_loader import CreateDataLoader

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 if in_dim//8 > 0 else 1, kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 if in_dim//8 > 0 else 1 , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height, length = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0, 2, 1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height, length)
        
        out = self.gamma*out + x
        return out,attention

if __name__ == "__main__":
    data_loader = CreateDataLoader().load_data_loader()
    sample = next(iter(data_loader))
    sample = sample[0]
    sample = sample.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
    attention = Self_Attn(5, 'relu')
    x, _ = attention(sample)