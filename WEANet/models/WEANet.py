import torch
import torch.nn as nn
from .blocks import WEANetBlock


class WEANet(nn.Module):
    """
    Methodology 5.1/5.4: modular, end-to-end framework constructed by
    stacking a series of self-contained WEANet Blocks (D=3 by default).
    """

    def __init__(self, configs):
        super(WEANet, self).__init__()
        self.d_depth = configs.d_depth  


        self.blocks = nn.ModuleList([
            WEANetBlock(configs.enc_in if i == 0 else configs.hidden_dim,
                        configs.n_bases, config.bottleneck_dim)
            for i in range(self.d_depth)
        ])


        self.final_fusion = nn.Conv1d(configs.hidden_dim * (2 * configs.n_bases),
                                      configs.final_fusion_dim,
                                      kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(configs.final_fusion_dim)
        self.relu = nn.ReLU(inplace=True)


        if configs.task == 'classification':
            self.head = nn.Linear(configs.final_fusion_dim, configs.num_classes)
        else:
            self.head = nn.Linear(configs.final_fusion_dim, configs.pred_len)

    def forward(self, x_enc):

        x = x_enc.permute(0, 2, 1)

        decomposed_features_for_recon = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == 0:
                decomposed_features_for_recon = x

        x = self.relu(self.bn(self.final_fusion(x)))


        x = x.mean(dim=-1)

        pred = self.head(x)


        return pred, decomposed_features_for_recon