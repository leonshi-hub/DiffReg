# === eval_ddpm2_pn2_loss3_chain.py — 适配链式训练的评估脚本 ===
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import datetime
from pathlib import Path

from models.ddpm2_pn2_posi import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from utils.util import PC_distance
from utils.diffusion_utils import ddim_sample_feedback
from LiverDataset import LiverDataset

# === 配置 ===
LOG_NAME = 'eval_ddpm2_pn2_loss3_chain'
BATCH_SIZE = 1
NUM_POINTS = 1024
DIFFUSION_STEPS = 800
DDIM_STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
CKPT_PATH = './log/liver_ddpm2_pn2_loss3_chain/2025-07-05_*/checkpoints/best_model.pth'
SAVE_VTP = True

# === 保存 VTP 点云 & 箭头 ===
def save_vtp_pointcloud(points: torch.Tensor, save_path: str):
    import pyvista as pv
    pc = points.squeeze(0).cpu().numpy()
    mesh = pv.PolyData(pc)
    mesh.save(save_path)

def save_vtp_arrows(start_points: torch.Tensor, end_points: torch.Tensor, save_path: str):
    import pyvista as pv
    start = start_points.squeeze(0).cpu().numpy()
    end = end_points.squeeze(0).cpu().numpy()
    all_points = np.vstack([start, end])
    N = start.shape[0]
    lines = []
    for i in range(N):
        lines.extend([2, i, i + N])
    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = np.array(lines, dtype=np.int32)
    poly.save(save_path)

# === 简单 Chamfer 实现 ===
def chamfer_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    dist_mat = torch.cdist(x, y, p=2)
    cd_xy = torch.min(dist_mat, dim=2)[0].mean(dim=1)
    cd_yx = torch.min(dist_mat, dim=1)[0].mean(dim=1)
    return (cd_xy + cd_yx).mean()

def main():
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    ckpt_file = sorted(Path(CKPT_PATH).parent.glob('*/checkpoints/best_model.pth'))[-1]
    ckpt = torch.load('log/liver_ddpm2_pn2_loss3_chain/2025-07-05_*/checkpoints/best_model.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)

    for i, batch in enumerate(dataloader):
        preop = batch['preop'].to(DEVICE).float()
        introp = batch['introp'].to(DEVICE).float()
        gt_disp = batch['displacement'].to(DEVICE).float()
        disp_mean = batch['disp_mean'].to(DEVICE).float()
        disp_std = batch['disp_std'].to(DEVICE).float()
        folder = batch['folder'][0]

        # === DDPM 采样：链式 ===
        disp_cur = torch.zeros_like(gt_disp)
        for t_val in reversed(range(DIFFUSION_STEPS)):
                t = torch.full((BATCH_SIZE,), t_val, device=DEVICE, dtype=torch.long)
                x_t, eps = diffusion.add_noise(gt_disp, t)
                eps_theta = model.predict_noise_step(preop, introp, disp_cur, x_t, t, pred_disp=disp_cur)
                alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar
                x0_pred = torch.clamp(x0_pred, -1., 1.)
                disp_cur = x0_pred.detach().detach()
        pred_disp_ddpm = disp_cur * disp_std + disp_mean
        warped_ddpm = preop + pred_disp_ddpm

        # === DDIM 采样：使用 feedback 版本 ===
        pred_disp_ddim = ddim_sample_feedback(
            model, preop, introp, disp_mean, disp_std, diffusion,
            ddim_steps=DDIM_STEPS, eta=0.0
        )
        warped_ddim = preop + pred_disp_ddim

        # === 评估指标 ===
        def eval_metrics(warped):
            rmse = F.mse_loss(warped, introp).sqrt()
            max_dist = torch.norm(warped - introp, dim=2).max()
            chamfer = chamfer_naive(warped, introp)
            pc_dist = PC_distance(warped.permute(0, 2, 1), introp.permute(0, 2, 1)).item()
            return rmse, max_dist, chamfer, pc_dist

        rmse_d, max_d, ch_d, pc_d = eval_metrics(warped_ddpm)
        rmse_i, max_i, ch_i, pc_i = eval_metrics(warped_ddim)

        print(f"📂 样本: {folder}")
        print(f"— DDPM (T={DIFFUSION_STEPS})")
        print(f"   📏 RMSE: {rmse_d.item():.6f}, MAX: {max_d.item():.6f}, Chamfer: {ch_d.item():.6f}, PC_dist: {pc_d:.6f}")
        print(f"— DDIM (steps={DDIM_STEPS})")
        print(f"   📏 RMSE: {rmse_i.item():.6f}, MAX: {max_i.item():.6f}, Chamfer: {ch_i.item():.6f}, PC_dist: {pc_i:.6f}")

        if SAVE_VTP:
            save_dir = f"./eval_vtp_chain/{folder}"
            os.makedirs(save_dir, exist_ok=True)
            save_vtp_pointcloud(warped_ddpm, f"{save_dir}/warped_ddpm.vtp")
            save_vtp_pointcloud(warped_ddim, f"{save_dir}/warped_ddim.vtp")
            save_vtp_pointcloud(preop, f"{save_dir}/preop.vtp")
            save_vtp_pointcloud(introp, f"{save_dir}/introp.vtp")
            save_vtp_arrows(preop, warped_ddpm, f"{save_dir}/ddpm_arrows.vtp")
            save_vtp_arrows(preop, warped_ddim, f"{save_dir}/ddim_arrows.vtp")
            print(f"📁 已保存 VTP 到 {save_dir}")

if __name__ == '__main__':
    main()
