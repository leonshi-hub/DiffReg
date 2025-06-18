
import torch
import torch.nn as nn
from utils.util import PointNetEncoder

class PointNetTransformer(nn.Module):
    def __init__(self, d_model=128, npoint=1024):
        super().__init__()
        self.encoder = PointNetEncoder(channel=3, d_model=d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=4, num_decoder_layers=4)
        self.linear = nn.Linear(d_model, 3)  # 直接输出 displacement 向量

    def forward(self, preop, introp):
        # preop, introp: [B, N, 3]
        B, N, _ = preop.shape
        x_input = preop.permute(0, 2, 1)  # [B, 3, N]
        y_input = introp.permute(0, 2, 1)  # [B, 3, N]

        feat_pre = self.encoder(x_input)  # [B, d_model, N]
        feat_int = self.encoder(y_input)  # [B, d_model, N]
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)  # [N, B, d]

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)  # [B, N, d_model]
        disp = self.linear(trans_feat)  # [B, N, 3]

        return disp
