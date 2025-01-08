import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import decode.neuralfitter.models.unet_param as unet_param

class Unet(nn.Module):

    def __init__(self, in_channels, out_channels, depth=4, initial_features=48, gain=2, pad_convs=False, norm=None, norm_groups=None):
        super(Unet, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = 1 if pad_convs else 0
        self.pool_mode = 'StrideConv'

        n_features = [self.in_channels] + [initial_features * gain ** level for level in range(self.depth)]
        self.features_per_level = n_features

        self.encoder = nn.ModuleList([UnetLayer(n_features[level], n_features[level + 1], 3, self.pad_convs, norm=norm, norm_groups=norm_groups)
                                      for level in range(self.depth)])

        self.base = UnetLayer(n_features[-1], gain * n_features[-1], 3, self.pad_convs, norm=norm, norm_groups=norm_groups)

        n_features = [initial_features * gain ** level for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([UnetLayer(n_features[level], n_features[level + 1], 3, self.pad_convs, norm=norm, norm_groups=norm_groups)
                                      for level in range(self.depth)])

        self.poolings = nn.ModuleList([self._pooler(level+1) for level in range(self.depth)])

        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1,
                                                         mode='nearest')
                                         for level in range(self.depth)])

        self.out_conv = nn.Conv2d(n_features[-1], out_channels, 1)
        self.activation = nn.ELU()

    def mergeArray(self, x, target):
        target = target = torch.cat((target, x), 1)
        return target

    def _upsampler(self, in_channels, out_channels, level, mode):
        # use bilinear upsampling + 1x1 convolutions
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2, mode=mode, ndim=2,
                        align_corners=False if mode == 'bilinear' else None)

    # pooling via maxpool2d
    def _pooler(self, level):
        if self.pool_mode == 'MaxPool':
            return nn.MaxPool2d(2)
        elif self.pool_mode == 'StrideConv':
            return nn.Conv2d(self.features_per_level[level], self.features_per_level[level],
                             kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        encoder_out = []

        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolings[level](x)

        x = self.base(x)

        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.mergeArray(encoder_out[level], x)
            x = self.decoder[level](x)

        x = self.out_conv(x)
        x = self.activation(x)
        return x


class UnetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, norm=None, norm_groups=None):
        super(UnetLayer, self).__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        self.p_dropout = None
        self.conv = self._conv_block(in_channels,out_channels,kernel_size,padding,activation=nn.ELU())

    def _conv_block(self, in_channels, out_channels, kernel_size, padding, activation=nn.ReLU()):

        if self.norm is not None:
            num_groups1 = min(in_channels, self.norm_groups)
            num_groups2 = min(out_channels, self.norm_groups)
        else:
            num_groups1 = None
            num_groups2 = None
        if self.norm is None:
            sequence = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=kernel_size, padding=padding),
                                     activation,
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=kernel_size, padding=padding),
                                     activation)
        elif self.norm == 'GroupNorm':
            sequence = nn.Sequential(nn.GroupNorm(num_groups1, in_channels),
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=kernel_size, padding=padding),
                                     activation,
                                     nn.GroupNorm(num_groups2, out_channels),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=kernel_size, padding=padding),
                                     activation)

        if self.p_dropout is not None:
            sequence.add_module('droupout', nn.Dropout2d(p=self.p_dropout))

        return sequence

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """ Upsample the input and change the number of channels
    via 1x1 Convolution if a different number of input/output channels is specified.
    """

    def __init__(self, scale_factor, mode='nearest',
                 in_channels=None, out_channels=None, align_corners=False,
                 ndim=3):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            if ndim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            elif ndim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, 1)
            else:
                raise ValueError("Only 2d and 3d supported")
        else:
            self.conv = None

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        if self.conv is not None:
            return self.conv(x)
        else:
            return x
