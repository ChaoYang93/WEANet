import torch
import torch.nn as nn
import numpy as np


class WaveletFrontEnd(nn.Module):
    """
    Methodology 5.3.1: Wavelet-Initialized Front-End (WFE).
    performs an adaptive, multi-resolution decomposition via grouped convolution
    initialized with select wavelet bases.
    """

    def __init__(self, c_in, n_bases, wavelet_coeffs_path=None, learnable=True):
        """
        :param c_in: Number of variables (M) in input time series (groups count).
        :param n_bases: Number of wavelet families (N) selected.
        :param wavelet_coeffs_path: Path to .npy file containing theoretical coefficients.
        :param learnable: Whether to allow filter coefficients to adapt during training.
        """
        super(WaveletFrontEnd, self).__init__()
        self.c_in = c_in

        self.n_filters_per_group = 2 * n_bases

        self.wavelet_conv = nn.Conv1d(in_channels=c_in,
                                      out_channels=c_in * self.n_filters_per_group,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      groups=c_in,
                                      bias=False)


        if wavelet_coeffs_path is not None:

            theoretical_coeffs = np.load(wavelet_coeffs_path)
            self.wavelet_conv.weight.data = torch.from_numpy(theoretical_coeffs).float()
            print(f"-> Successfully initialized WFE with {wavelet_coeffs_path}")
        else:
            print("-> Warning: WFE initialized randomly. Reproducibility might be affected.")


        self.wavelet_conv.weight.requires_grad = learnable

    def forward(self, x):

        return self.wavelet_conv(x)
