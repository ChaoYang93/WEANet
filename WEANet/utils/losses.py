import torch
import torch.nn as nn


class LearnableIDWT(nn.Module):
    """
    Methodology 5.5: parallel, learnable inverse wavelet transform module
    intended to preserve structural inductive bias during training.
    Strictly confined to training phase.
    """

    def __init__(self, c_feat_in, c_orig_out, n_bases):
        """
        :param c_feat_in: hidden channels from first block (M*2N).
        :param c_orig_out: original input channels (M).
        :param n_bases: Number of wavelet families (N).
        """
        super(LearnableIDWT, self).__init__()


        self.learnable_idwt_conv = nn.ConvTranspose1d(in_channels=c_feat_in,
                                                      out_channels=c_orig_out,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1,
                                                      groups=c_orig_out,
                                                      bias=False)



    def forward(self, block1_features):
        return self.learnable_idwt_conv(block1_features)


class DualObjectiveLoss(nn.Module):
    """
    Methodology 5.5: composites loss function that balances task performance
    with structural regularization via L1 Reconstruction loss.
    L_total = L_task + lambda * L_recon
    """

    def __init__(self, configs):
        super(DualObjectiveLoss, self).__init__()
        self.lambda_recon = configs.lambda_recon


        if configs.task == 'classification':
            self.criterion_task = nn.CrossEntropyLoss()
        else:
            self.criterion_task = nn.MSELoss()


        self.criterion_recon = nn.L1Loss()
        self.learnable_idwt = LearnableIDWT(configs.hidden_dim * (2 * configs.n_bases),
                                            configs.enc_in,
                                            configs.n_bases)

    def forward(self, pred, targets, block1_features, original_input):

        original_input_standard = original_input.permute(0, 2, 1)

        loss_task = self.criterion_task(pred, targets)


        if self.training:

            reconstructed_signal = self.learnable_idwt(block1_features)


            loss_recon = self.criterion_recon(reconstructed_signal, original_input_standard)


            loss_total = loss_task + self.lambda_recon * loss_recon
            return loss_total, loss_task, loss_recon


        return loss_task