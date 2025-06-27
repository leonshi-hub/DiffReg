
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import PointNetEncoder, get_timestep_embedding


class DDPMDeformer(nn.Module):
    def __init__(self, point_dim=3, cond_dim=128, time_dim=128, hidden_dim=128):
        super().__init__()
        in_dim = point_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x_t, t_embed, cond_feat):
        # x_t: [B, N, 3], t_embed: [B, N, D], cond_feat: [B, N, D]
        h = torch.cat([x_t, cond_feat, t_embed], dim=-1)  # [B, N, 3+cond_dim+time_dim]
        return self.net(h)  # [B, N, 3]


class TransformerDDPMRegNet(nn.Module):
    def __init__(self, d_model=128, npoint=1024):
        super().__init__()
        self.encoder = PointNetEncoder(channel=3, d_model=d_model)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=4, num_decoder_layers=4)
        self.ddpm = DDPMDeformer(cond_dim=d_model, time_dim=d_model)

        self.npoint = npoint
        self.d_model = d_model
        # self.t_embed_proj = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.ReLU(),
        #     nn.Dropout(0.1))


    def forward(self, preop, introp, t, return_noise=False):
        """
        preop:  [B, N, 3]
        introp: [B, N, 3]
        t:      [B]  # timestep
        return_noise: if True, return predicted noise; else, return predicted displacement x0
        """
        B, N, _ = preop.shape
        x_input = preop.permute(0, 2, 1)  # [B, 3, N]
        y_input = introp.permute(0, 2, 1)  # [B, 3, N]

        feat_pre = self.encoder(x_input)  # [B, d_model, N]
        feat_int = self.encoder(y_input)  # [B, d_model, N]
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)  # [N, B, d]

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)  # [B, N, d_model]

        # timestep embedding
        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]
        # t_embed_raw = get_timestep_embedding(t, self.d_model)           # [B, d]
        # t_embed_proj = self.t_embed_proj(t_embed_raw)                   # [B, d]
        # t_embed = t_embed_proj.unsqueeze(1).expand(-1, N, -1)           # [B, N, d]

        # training: return noise; inference: return x0 (displacement)
        if return_noise:
            return lambda x_t: self.ddpm(x_t, t_embed, trans_feat)
        else:
            raise NotImplementedError("Use the training loop to predict x0 or sampling loop separately.")
    def predict_noise_step(self, preop, introp, x_t, t):
        """Predict noise for a given noisy displacement at timestep ``t``.

        This helper recomputes PointNet+Transformer features every call so it can
        be used in iterative feedback sampling where the preoperative point cloud
        is updated at each step.
        """
        B, N, _ = preop.shape
        x_input = preop.permute(0, 2, 1)
        y_input = introp.permute(0, 2, 1)

        feat_pre = self.encoder(x_input)
        feat_int = self.encoder(y_input)
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)

        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)

        return self.ddpm(x_t, t_embed, trans_feat)
