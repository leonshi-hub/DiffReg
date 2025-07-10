import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

from models.ddpm2_pn2_posi import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from utils.diffusion_utils import ddim_sample_feedback
from LiverDataset import LiverDataset
from utils.util import PC_distance

# === 配置 ===
LOG_NAME = 'liver_ddpm2_pn2_experiment'
BATCH_SIZE = 1
NUM_POINTS = 1024
DIFFUSION_STEPS = 2000
DDIM_STEPS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh_2'
SAVE_VTP = True


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


def chamfer_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    dist_mat = torch.cdist(x, y, p=2)
    cd_xy = torch.min(dist_mat, dim=2)[0].mean(dim=1)
    cd_yx = torch.min(dist_mat, dim=1)[0].mean(dim=1)
    return (cd_xy + cd_yx).mean()


def main():
    print("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(10))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"✅ 数据样本数: {len(dataset)}")

    # === 模型加载 ===
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    ckpt = torch.load('log/liver_ddpm2_pn2_loss3_experiment/2025-07-04_21-19-31/checkpoints/best_model.pth', map_location=DEVICE)
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

            # === DDPM 推理（用 eta=1.0）===
            pred_disp_ddpm = ddim_sample_feedback(
                model, preop, introp, disp_mean, disp_std,
                diffusion, ddim_steps=DIFFUSION_STEPS, eta=1.0
            )
            warped_ddpm = preop + pred_disp_ddpm

            # === DDIM 推理（用 eta=0.0）===
            pred_disp_ddim = ddim_sample_feedback(
                model, preop, introp, disp_mean, disp_std,
                diffusion, ddim_steps=DDIM_STEPS, eta=0.0
            )
            warped_ddim = preop + pred_disp_ddim

            # === 评估 ===
            def eval_metrics(warped):
                rmse = F.mse_loss(warped, introp).sqrt()
                max_dist = torch.norm(warped - introp, dim=2).max()
                chamfer = chamfer_naive(warped, introp)
                pc_dist = PC_distance(warped.permute(0, 2, 1), introp.permute(0, 2, 1)).item()
                return rmse, max_dist, chamfer, pc_dist

            rmse_ddpm, max_ddpm, chamfer_ddpm, pc_ddpm = eval_metrics(warped_ddpm)
            rmse_ddim, max_ddim, chamfer_ddim, pc_ddim = eval_metrics(warped_ddim)
            rmse_preop, max_preop, chamfer_preop, pc_preop = eval_metrics(preop)

            print(f"📂 样本: {folder}")
            print(f"— 原始 preop（未配准）")
            print(f"   📏 RMSE: {rmse_preop.item():.6f}, MAX: {max_preop.item():.6f}, Chamfer: {chamfer_preop.item():.6f}, PC_dist: {pc_preop:.6f}")
            print(f"— DDPM (T={DIFFUSION_STEPS})")
            print(f"   📏 RMSE: {rmse_ddpm.item():.6f}, MAX: {max_ddpm.item():.6f}, Chamfer: {chamfer_ddpm.item():.6f}, PC_dist: {pc_ddpm:.6f}")
            print(f"— DDIM (steps={DDIM_STEPS})")
            print(f"   📏 RMSE: {rmse_ddim.item():.6f}, MAX: {max_ddim.item():.6f}, Chamfer: {chamfer_ddim.item():.6f}, PC_dist: {pc_ddim:.6f}")

            if SAVE_VTP:
                save_dir = f"./eval_vtp_ddpm2pn2loss3/{folder}"
                os.makedirs(save_dir, exist_ok=True)
                save_vtp_pointcloud(warped_ddpm, f"{save_dir}/warped_ddpm.vtp")
                save_vtp_pointcloud(warped_ddim, f"{save_dir}/warped_ddim.vtp")
                save_vtp_pointcloud(introp, f"{save_dir}/introp.vtp")
                save_vtp_pointcloud(preop, f"{save_dir}/preop.vtp")
                save_vtp_arrows(preop, warped_ddpm, f"{save_dir}/ddpm_arrows.vtp")
                save_vtp_arrows(preop, warped_ddim, f"{save_dir}/ddim_arrows.vtp")
                print(f"📁 已保存 VTP 到 {save_dir}（含箭头）")


if __name__ == '__main__':
    main()
