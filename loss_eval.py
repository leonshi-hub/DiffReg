import os
import torch
import datetime
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np

from models.ddpm_pn import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from LiverDataset import LiverDataset
from utils.util import PC_distance

# === 配置 ===
LOG_NAME = 'liver_ddpm_experiment'
BATCH_SIZE = 1
NUM_EPOCHS = 1
LR = 1e-3
NUM_POINTS = 1024
DIFFUSION_STEPS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
SAVE_VTP = True  # 保存 warped 点云为 .vtp 文件

# === 时间步依赖的 pred_disp 权重函数 ===
def get_pred_disp_weight(t: torch.Tensor, T: int, alpha=5.0):
    step = T - t.float()
    num = 1.0 - torch.exp(-alpha * step / T)
    denom = 1.0 - torch.exp(torch.tensor(-alpha, device=t.device))
    return num / denom

def save_vtp_pointcloud(points: torch.Tensor, save_path: str):
    import pyvista as pv
    pc = points.squeeze(0).cpu().numpy()
    mesh = pv.PolyData(pc)
    mesh.save(save_path)

def chamfer_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x, y: [B, N, 3]
    B, N, _ = x.shape
    dist_mat = torch.cdist(x, y, p=2)  # [B, N, N]
    cd_xy = torch.min(dist_mat, dim=2)[0].mean(dim=1)
    cd_yx = torch.min(dist_mat, dim=1)[0].mean(dim=1)
    return (cd_xy + cd_yx).mean()  # scalar

def main():
    print("\U0001F4E6 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"✅ 数据样本数: {len(dataset)}")

    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    ckpt = torch.load('./log/liver_ddpm_experiment/2025-06-20_16-00-38/checkpoints/best_model.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            preop = batch['preop'].to(DEVICE).float()
            introp = batch['introp'].to(DEVICE).float()
            gt_disp = batch['displacement'].to(DEVICE).float()
            disp_mean = batch['disp_mean'].to(DEVICE).float()
            disp_std = batch['disp_std'].to(DEVICE).float()
            folder = batch['folder'][0]

            disp_cur = torch.zeros_like(gt_disp)
            stepwise_mse = []

            for t_val in reversed(range(DIFFUSION_STEPS)):
                t = torch.full((BATCH_SIZE,), t_val, device=DEVICE, dtype=torch.long)
                x_t, eps = diffusion.add_noise(gt_disp, t)

                eps_theta = model.predict_noise_step(preop, introp, disp_cur, x_t, t, pred_disp=disp_cur)
                loss = F.mse_loss(eps_theta, eps)
                stepwise_mse.append(loss.item())
                print(f"t={t_val}, mse={loss.item():.4f}")

                alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar
                disp_cur = x0_pred

            pred_disp = x0_pred * disp_std + disp_mean
            warped = preop + pred_disp

            # === 数值评估 ===
            gt_disp_unorm = gt_disp * disp_std + disp_mean
            final_rmse = F.mse_loss(pred_disp, gt_disp_unorm).sqrt()
            chamfer = chamfer_naive(warped, introp)
            pc_dist = PC_distance(warped.permute(0, 2, 1), introp.permute(0, 2, 1)).item()

            range_xyz = preop.max(dim=1).values - preop.min(dim=1).values
            point_cloud_range = range_xyz.norm(dim=1).mean()
            relative_rmse = final_rmse / point_cloud_range

            print(f"📂 样本: {folder}")
            print(f"📀 RMSE: {final_rmse.item():.6f}")
            print(f"🕥 Chamfer (naive): {chamfer.item():.6f}")
            print(f"🕥 PC_distance: {pc_dist:.6f}")
            print(f"📀 相对 RMSE（归一化）: {relative_rmse.item():.6f}")

            print("📊 每步 t 的逐步降噪 MSE:")
            print([f"{l:.4f}" for l in stepwise_mse])

            if SAVE_VTP:
                save_dir = f"./evalpn33_vtp/{folder}"
                os.makedirs(save_dir, exist_ok=True)
                save_vtp_pointcloud(preop, f"{save_dir}/preop.vtp")
                save_vtp_pointcloud(introp, f"{save_dir}/introp.vtp")
                save_vtp_pointcloud(warped, f"{save_dir}/warped.vtp")
                print(f"📂 已保存 VTP 点云到 {save_dir}")

if __name__ == '__main__':
    main()
