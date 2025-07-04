import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import PointNetPPEncoder, get_timestep_embedding, get_xyz_positional_encoding, square_distance, index_points


class PointTransformerLayer(nn.Module):
    def __init__(self, d_model, k=128):
        super().__init__()
        self.k = k
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_pos = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, xyz, feat):
        B, N, _ = xyz.shape
        d = feat.shape[-1]

        # 计算 pairwise distance，构建邻接图
        dist = square_distance(xyz, xyz)  # [B, N, N]
        knn_idx = dist.topk(k=self.k, dim=-1, largest=False)[1]  # [B, N, k]

        neighbor_xyz = index_points(xyz, knn_idx)        # [B, N, k, 3]
        neighbor_feat = index_points(feat, knn_idx)      # [B, N, k, d]

        q = self.fc_q(feat).unsqueeze(2)                 # [B, N, 1, d]
        k = self.fc_k(neighbor_feat)                     # [B, N, k, d]
        v = self.fc_v(neighbor_feat)                     # [B, N, k, d]

        rel_pos = xyz.unsqueeze(2) - neighbor_xyz        # [B, N, k, 3]
        pos_enc = self.fc_pos(rel_pos)                   # [B, N, k, d]

        attn = self.softmax((q - k + pos_enc).sum(-1, keepdim=True))  # [B, N, k, 1]
        agg = (attn * (v + pos_enc)).sum(2)              # [B, N, d]
        out = self.proj(agg)
        return feat + self.gamma * out                   # Residual


class PointTransformerBlock(nn.Module):
    def __init__(self, d_model, num_layers=4, k=128):
        super().__init__()
        self.layers = nn.ModuleList([
            PointTransformerLayer(d_model, k=k) for _ in range(num_layers)
        ])

    def forward(self, xyz, feat):
        for layer in self.layers:
            feat = layer(xyz, feat)
        return feat


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
        h = torch.cat([x_t, cond_feat, t_embed], dim=-1)
        return self.net(h)


class TransformerDDPMRegNet(nn.Module):
    def __init__(self, d_model=128, npoint=1024, use_pred_disp=False):
        super().__init__()
        self.encoder_pre = PointNetPPEncoder(npoint=npoint, d_model=d_model, use_pred_disp=use_pred_disp)
        self.encoder_int = PointNetPPEncoder(npoint=npoint, d_model=d_model, use_pred_disp=False)
        self.pt_block = PointTransformerBlock(d_model=d_model, num_layers=4, k=128)
        self.ddpm = DDPMDeformer(cond_dim=d_model, time_dim=d_model)
        self.npoint = npoint
        self.d_model = d_model

    def forward(self, preop, introp, disp, t, pred_disp=None, return_noise=False):
        B, N, _ = preop.shape

        pos_enc_pre = get_xyz_positional_encoding(preop, d_pos=18)
        pos_enc_int = get_xyz_positional_encoding(introp, d_pos=18)
        x_input = torch.cat([preop, pos_enc_pre], dim=-1).permute(0, 2, 1)
        y_input = torch.cat([introp, pos_enc_int], dim=-1).permute(0, 2, 1)

        disp_input = disp.permute(0, 2, 1)
        pred_input = pred_disp.permute(0, 2, 1) if pred_disp is not None else None

        feat_pre = self.encoder_pre(x_input, pred_disp=pred_input)  # [B, C, N]
        feat_int = self.encoder_int(y_input)                        # [B, C, N]

        feat_pre = feat_pre.permute(0, 2, 1)  # [B, N, C]
        feat_int = feat_int.permute(0, 2, 1)

        fused_feat = self.pt_block(preop, feat_pre + feat_int)     # [B, N, C]

        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)

        if return_noise:
            return lambda x_t: self.ddpm(x_t, t_embed, fused_feat)
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

        feat_pre = self.encoder_pre(x_input, pred_disp=pred_input)
        feat_int = self.encoder_int(y_input)

        feat_pre = feat_pre.permute(0, 2, 1)
        feat_int = feat_int.permute(0, 2, 1)

        fused_feat = self.pt_block(preop, feat_pre + feat_int)

        t_embed = get_timestep_embedding(t, self.d_model).unsqueeze(1).expand(-1, N, -1)
        return self.ddpm(x_t, t_embed, fused_feat)
