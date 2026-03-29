import torch
import torch.nn as nn


class BottleneckTransformation(nn.Module):
    """
    Methodology 5.3.2: Lightweight bottleneck module for parameter efficiency.
    Structure: Pointwise (kernel 1) -> Standard (kernel 3) -> Pointwise (kernel 1)
    """

    def __init__(self, c_in, bottleneck_dim):
        super(BottleneckTransformation, self).__init__()
        self.down_proj = nn.Conv1d(c_in, bottleneck_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_dim)

        self.feat_extract = nn.Conv1d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)

        self.up_proj = nn.Conv1d(bottleneck_dim, c_in, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(c_in)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.down_proj(x)))
        out = self.relu(self.bn2(self.feat_extract(out)))
        out = self.relu(self.bn3(self.up_proj(out)))
        return out


class WEANetBlock(nn.Module):
    """
    Methodology 5.3: Fundamental building unit responsible for one level of
    signal decomposition (WFE) and feature extraction (PCC).
    """

    def __init__(self, c_in, n_bases, bottleneck_dim, polyphony_factor=2):
        """
        :param polyphony_factor: key hyperparameter K=2 term the 'polyphony factor'.
        """
        super(WEANetBlock, self).__init__()
        # WFE Layer
        self.wfe = WaveletFrontEnd(c_in, n_bases)

        # 1. Methodology 核心：频段分治 (PCC)。 deliberate decision to set K=2
        # partitions the input channels into two "super-groups": Low-Freq and High-Freq.
        pcc_in_channels = c_in * (2 * n_bases)
        half_channels = pcc_in_channels // 2

        # Methodology implies identity paths F_low and F_high with independent weights.
        self.f_low = BottleneckTransformation(half_channels, bottleneck_dim)
        self.f_high = BottleneckTransformation(half_channels, bottleneck_dim)


        self.residual_downsample = nn.Conv1d(c_in, pcc_in_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):

        residual = self.residual_downsample(x)


        decomposed_feat = self.wfe(x)

        low_freq, high_freq = torch.chunk(decomposed_feat, 2, dim=1)


        out_low = self.f_low(low_freq)
        out_high = self.f_high(high_freq)


        pcc_out = torch.cat([out_low, out_high], dim=1)


        return self.relu(pcc_out + residual)