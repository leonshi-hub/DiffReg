import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import PointNetPPEncoder, get_timestep_embedding, get_xyz_positional_encoding


class DDPMDeformer(nn.Module):
    def __init__(self, point_dim=3, cond_dim=128, time_dim=128, hidden_dim=128):
        super().__init__()
        in_dim = point_dim + cond_dim + time_dim
        self.t_embed_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x_t, t_embed, cond_feat):
        t_embed = self.t_embed_proj(t_embed)
        h = torch.cat([x_t, cond_feat, t_embed], dim=-1)  # [B, N, 3+cond_dim+time_dim]
        return self.net(h)  # [B, N, 3]


class TransformerDDPMRegNet(nn.Module):
    def __init__(self, d_model=128, npoint=1024, use_pred_disp=False):
        super().__init__()

        # 新增输入通道为 3 + 18
        self.encoder_pre = PointNetPPEncoder(npoint=1024, d_model=128, use_pred_disp=True)
        self.encoder_int = PointNetPPEncoder(npoint=1024, d_model=128, use_pred_disp=False)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=2, num_decoder_layers=2)
        self.ddpm = DDPMDeformer(cond_dim=d_model, time_dim=d_model)

        self.npoint = npoint
        self.d_model = d_model

    def forward(self, preop, introp, disp, t, pred_disp=None, return_noise=False):
        B, N, _ = preop.shape

        pos_enc_pre = get_xyz_positional_encoding(preop, d_pos=18)  # [B, N, 18]
        pos_enc_int = get_xyz_positional_encoding(introp, d_pos=18)
        x_input = torch.cat([preop, pos_enc_pre], dim=-1).permute(0, 2, 1)  # [B, 21, N]
        y_input = torch.cat([introp, pos_enc_int], dim=-1).permute(0, 2, 1)

        disp_input = disp.permute(0, 2, 1)
        pred_input = pred_disp.permute(0, 2, 1) if pred_disp is not None else None

        #feat_pre = self.encoder_pre(x_input, disp_input, pred_input)  # [B, d_model, N]
        feat_pre = self.encoder_pre(x_input, pred_disp=pred_input)

        feat_int = self.encoder_int(y_input)              # [B, d_model, N]
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)  # [N, B, d]

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)  # [B, N, d_model]

        # timestep embedding
        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)  # [B, N, d_model]

        if return_noise:
            return lambda x_t: self.ddpm(x_t, t_embed, trans_feat)
        else:
            raise NotImplementedError("Use the training loop to predict x0 or sampling loop separately.")

    def predict_noise_step(self, preop, introp, disp, x_t, t, pred_disp=None):
        B, N, _ = preop.shape

        pos_enc_pre = get_xyz_positional_encoding(preop, d_pos=18)
        pos_enc_int = get_xyz_positional_encoding(introp, d_pos=18)
        x_input = torch.cat([preop, pos_enc_pre], dim=-1).permute(0, 2, 1)
        y_input = torch.cat([introp, pos_enc_int], dim=-1).permute(0, 2, 1)

        disp_input = disp.permute(0, 2, 1)
        pred_input = pred_disp.permute(0, 2, 1) if pred_disp is not None else None

        #feat_pre = self.encoder_pre(x_input, disp_input, pred_input)
        feat_pre = self.encoder_pre(x_input, pred_disp=pred_input)

        feat_int = self.encoder_int(y_input)
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)
        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)

        return self.ddpm(x_t, t_embed, trans_feat)
