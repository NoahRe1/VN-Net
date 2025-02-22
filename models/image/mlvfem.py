import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        if first_conv:
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding

            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)

            intermed_channels = 45

            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)

            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = (0, padding[1], padding[2])

            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)

            intermed_channels = int(
                math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                           (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)

            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                           stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


class SpatioTemporalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)

        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class MLVFEM(nn.Module):
    def __init__(self, stages):
        super(MLVFEM, self).__init__()

        self.stages = stages
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))

        self.__init_weight()
    
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.stages(x)
        x = self.pool(x)
        x = x.squeeze(2)
        return x

stages = nn.Sequential(
    SpatioTemporalConv(in_channels=6, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), first_conv=True),
    SpatioTemporalResLayer(in_channels=16, out_channels=16, kernel_size=3, layer_size=2),
    SpatioTemporalResLayer(in_channels=16, out_channels=32, kernel_size=3, layer_size=2, downsample=True),
    SpatioTemporalResLayer(in_channels=32, out_channels=32, kernel_size=3, layer_size=2, downsample=True),
)

def get_MLVFEM():
    return MLVFEM(stages)