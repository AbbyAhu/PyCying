import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv1d, Conv2d, Conv3d
from .spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d

class ChannelRMSNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x):
        mean_hw = x.mean(dim=(2, 3), keepdim=True)
        
        var_c = mean_hw.var(dim=1, keepdim=True, unbiased=False)
        
        x = x * torch.rsqrt(var_c + self.eps)
        
        return x * self.weight[None, :, None, None]

class BaseOperatorLayer(nn.Module):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_width = hidden_width
        self.spe_opt_size = spe_opt_size
        self.spa_opt_size = spa_opt_size
        self.spectral_opts = self._create_spectral_opts()
        self.spatial_opts = self._create_spatial_opts()
        self.nonlinear_opts = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.hidden_width,
                kernel_size=1
            ),
            nn.SiLU(),
            nn.Conv2d(
                self.hidden_width,
                self.out_channels,
                kernel_size=1
            )
        )
        
        self.linear_skip = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1
        )
        self.norm_in = ChannelRMSNorm2d(in_channels)
        self.norm_out = ChannelRMSNorm2d(out_channels)

    def forward(self, input, mode='train'):
        lin_opts = self.norm_in(self.spectral_opts(input, mode) + input)
        outputs = self.norm_out(self.nonlinear_opts(lin_opts) + self.linear_skip(input))
        return outputs

    def _interpolate(self, input):
        raise NotImplementedError

    def _create_spectral_opts(self):
        raise NotImplementedError

    def _create_spatial_opts(self):
        raise NotImplementedError


class OperatorLayer1d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='linear')

    def _create_spectral_opts(self):
        return SpectralConv1d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv1d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )


class OperatorLayer2d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='bilinear')

    def _create_spectral_opts(self):
        return SpectralConv2d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv2d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )


class OperatorLayer3d(BaseOperatorLayer):
    def __init__(self, size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size):
        super().__init__(size, in_channels, out_channels, hidden_width, spe_opt_size, spa_opt_size)

    def _interpolate(self, input):
        return F.interpolate(input, size=self.size, mode='trilinear')

    def _create_spectral_opts(self):
        return SpectralConv3d(
            in_channels=self.in_channels,
            opt_size=self.spe_opt_size
        )
    
    def _create_spatial_opts(self):
        return Conv3d(
            in_channels=self.in_channels,
            kernel_size=self.spa_opt_size
        )