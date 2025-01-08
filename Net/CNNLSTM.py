import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import decode.neuralfitter.models.unet_param as unet_param
import Net.Unet as Unet
from torch.autograd import Variable
import Choose_Device as Device

class CNNBiLSTM(nn.Module):
    
    ch_out = 11
    out_channels_heads = (1, 4, 4, 1)  # p head, phot,xyz_mu head, phot,xyz_sig head, bg head

    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]  # channel indices with respective activation function
    tanh_ch_ix = [2, 3, 4]
    relu_ch_ix = [9]

    p_ch_ix = [0]  # channel indices of the respective parameters
    pxyz_mu_ch_ix = slice(1, 5)
    pxyz_sig_ch_ix = slice(5, 9)
    bg_ch_ix = [9]
    img_ch_ix = [10]
    
    
    def __init__(self, in_channels=1, out_channels=11, depth=2, seq_len=3, initial_features=48, gain=2, pad_convs=False, norm=None, norm_groups=None, sigma_eps_default=0.01):
        super(CNNBiLSTM, self).__init__()
        self.sigma_eps_default = sigma_eps_default
        self.initial_features = initial_features
        self.seq_len = seq_len

        self.forward_layer = ConvLSTMCell(2*initial_features,initial_features,norm=norm,norm_groups=norm_groups)
        self.backward_layer = ConvLSTMCell(2*initial_features,initial_features,norm=norm,norm_groups=norm_groups)

        self.unet1 = Unet.Unet(in_channels,initial_features,depth=depth,pad_convs=pad_convs,norm=norm,norm_groups=norm_groups)
        self.unet2 = Unet.Unet(3*initial_features,initial_features,depth=depth,pad_convs=pad_convs,norm=norm,norm_groups=norm_groups)

        self.union_firsthalf = Unet.Unet(seq_len//2*initial_features,initial_features,depth=depth,pad_convs=pad_convs,norm=norm,norm_groups=norm_groups)
        self.union_latterhalf = Unet.Unet(seq_len//2*initial_features,initial_features,depth=depth,pad_convs=pad_convs,norm=norm,norm_groups=norm_groups)

        self.add_conv = mix_conv(3*initial_features, initial_features, norm=norm, norm_groups=norm_groups)
        self.outconvlist = nn.ModuleList([OutLayer(initial_features, i) for i in self.out_channels_heads])

    def forward(self, x, hidden_state=None):
        """
        x: [b, t, c, w, h]
        h, c: 48*w*h
        """
        x = x.unsqueeze(2)
        last_output_forward = []
        last_output_backward = []
        firstlayer = []
        if(hidden_state is None):
            tensor_size = (x.size(3), x.size(4))
            h1, c1 = self.forward_layer.init_hidden(batch_size=x.size(0), tensor_size=tensor_size, device=Device.device)
            h2, c2 = self.backward_layer.init_hidden(batch_size=x.size(0), tensor_size=tensor_size, device=Device.device)

        for t in range(self.seq_len):
            o = self.unet1(x[:,t,:,:,:])
            firstlayer.append(o)

        for t in range(self.seq_len):
            h1, c1 = self.forward_layer(firstlayer[t], [h1, c1])
            last_output_forward.append([h1, c1])
        
        for t in range(self.seq_len-1,-1,-1):
            h2, c2 = self.backward_layer(firstlayer[t], [h2, c2])
            last_output_backward.append([h2, c2])

        tar = self.seq_len // 2
        last_output = []
        for t in range(self.seq_len):
            o = torch.cat([firstlayer[t],last_output_forward[t][0],last_output_backward[self.seq_len-t-1][0]],dim=1)
            o = self.add_conv(o)
            last_output.append(o)

        o1 = torch.cat([last_output[i] for i in range(tar)],dim=1)
        o1 = self.union_firsthalf(o1)
        o2 = torch.cat([last_output[i] for i in range(self.seq_len-1,tar,-1)],dim=1)
        o2 = self.union_latterhalf(o2)
        o = torch.cat([o1,last_output[tar],o2],dim=1)

        o3 = self.unet2(o)

        o_heads = [outconv.forward(o3) for outconv in self.outconvlist]
        o = torch.cat(o_heads, dim=1)

        o[:, [0]] = torch.clamp(o[:, [0]], min=-8., max=8.)

        o[:, self.sigmoid_ch_ix] = torch.sigmoid(o[:, self.sigmoid_ch_ix])
        o[:, self.tanh_ch_ix] = torch.tanh(o[:, self.tanh_ch_ix])

        o[:, self.pxyz_sig_ch_ix] = o[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps_default
        
        return o

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, norm_groups=None):
        super(ConvLSTMCell, self).__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        if self.norm is not None:
            groups_1 = min(in_channels, self.norm_groups)
            self.gn = nn.GroupNorm(groups_1,in_channels)
        else:
            groups_1 = None
            self.gn = None
        self.hidden_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 4*out_channels, kernel_size=3, padding=1)
        

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        h_t = torch.cat([h_cur, x], dim=1)
        if self.gn is not None:
            h_t = self.gn(h_t)
        h_t = self.conv(h_t)
        h_t = nn.ELU().forward(h_t)
        i,f,o,g = torch.split(h_t, self.hidden_channels , dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, tensor_size, device):
        height, width = tensor_size
        return (Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device),
                Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device))
    
# def UnionCNN(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels,2*in_channels,3,padding=1),
#         nn.ELU(),
#         # nn.Conv2d(2*in_channels,4*in_channels,3,padding=1),
#         # nn.ELU(),
#         # nn.Conv2d(4*in_channels,in_channels,3,padding=1),
#         # nn.ELU(),
#         nn.Conv2d(2*in_channels,in_channels,3,padding=1),
#         nn.ELU(),
#         nn.Conv2d(in_channels,out_channels,3,padding=1),
#         nn.ELU(),
#     )

def OutLayer(in_channels, out_channels, norm=None, norm_groups=None):
    if norm is not None:
        groups_1 = min(in_channels, norm_groups)
        groups_2 = min(1, norm_groups)
    else:
        groups_1 = None
        groups_2 = None
    
    if norm == 'GroupNorm':
        return nn.Sequential(
            nn.GroupNorm(groups_1, in_channels),
            nn.Conv2d(in_channels, in_channels,kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
        )
    
def mix_conv(in_channels, out_channels, norm=None, norm_groups=None):
    if norm is not None:
        groups_1 = min(in_channels, norm_groups)
    else:
        groups_1 = None
    
    if norm == 'GroupNorm':
        return nn.Sequential(
            nn.GroupNorm(groups_1, in_channels),
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1),
            nn.ELU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ELU(),
        )