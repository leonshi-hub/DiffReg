import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.ddpm2_pn_posi import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from utils.diffusion_utils import ddim_sample_feedback
from LiverDataset import LiverDataset
from utils.util import PC_distance

# === 配置 ===
BATCH_SIZE = 1
NUM_POINTS = 1024
DIFFUSION_STEPS = 200
DDIM_STEPS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
SAVE_VTP = True

def save_vtp_pointcloud(points: torch.Tensor, save_path: str):
    import pyvista as pv
    pc = points.squeeze(0).cpu().numpy()
    mesh = pv.PolyData(pc)
    mesh.save(save_path)

def chamfer_naive(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    dist_mat = torch.cdist(x, y, p=2)
    cd_xy = torch.min(dist_mat, dim=2)[0].mean(dim=1)
    cd_yx = torch.min(dist_mat, dim=1)[0].mean(dim=1)
    return (cd_xy + cd_yx).mean()

def main():
    print("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5))  # 可根据需要修改数量
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"✅ 数据样本数: {len(dataset)}")

    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    ckpt_path = 'log/liver_ddpm2_experiment/2025-06-25_18-27-31/checkpoints/best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
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

            # === DDPM 推理 ===
            disp_cur = torch.zeros_like(gt_disp)
            for t_val in reversed(range(DIFFUSION_STEPS)):
                t = torch.full((BATCH_SIZE,), t_val, device=DEVICE, dtype=torch.long)
                x_t, eps = diffusion.add_noise(gt_disp, t)
                eps_theta = model.predict_noise_step(preop, introp, disp_cur, x_t, t, pred_disp=disp_cur)

                alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1)
                sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
                x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar
                disp_cur = x0_pred
            pred_disp_ddpm = x0_pred * disp_std + disp_mean
            warped_ddpm = preop + pred_disp_ddpm

            # === DDIM 推理 ===
            pred_disp_ddim = ddim_sample_feedback(model, preop, introp, disp_mean, disp_std, diffusion, ddim_steps=DDIM_STEPS, eta=0.0)
            warped_ddim = preop + pred_disp_ddim

            def eval_metrics(warped):
                rmse = F.mse_loss(warped, introp).sqrt()
                max_dist = torch.norm(warped - introp, dim=2).max()
                chamfer = chamfer_naive(warped, introp)
                pc_dist = PC_distance(warped.permute(0, 2, 1), introp.permute(0, 2, 1)).item()
                return rmse, max_dist, chamfer, pc_dist

            rmse_ddpm, max_ddpm, chamfer_ddpm, pc_ddpm = eval_metrics(warped_ddpm)
            rmse_ddim, max_ddim, chamfer_ddim, pc_ddim = eval_metrics(warped_ddim)

            print(f"📂 样本: {folder}")
            print(f"— DDPM (T={DIFFUSION_STEPS})")
            print(f"   📏 RMSE: {rmse_ddpm.item():.6f}, MAX: {max_ddpm.item():.6f}, Chamfer: {chamfer_ddpm.item():.6f}, PC_dist: {pc_ddpm:.6f}")
            print(f"— DDIM (steps={DDIM_STEPS})")
            print(f"   📏 RMSE: {rmse_ddim.item():.6f}, MAX: {max_ddim.item():.6f}, Chamfer: {chamfer_ddim.item():.6f}, PC_dist: {pc_ddim:.6f}")

            if SAVE_VTP:
                save_dir = f"./eval_vtp_ddpm2/{folder}"
                os.makedirs(save_dir, exist_ok=True)
                save_vtp_pointcloud(warped_ddpm, f"{save_dir}/warped_ddpm.vtp")
                save_vtp_pointcloud(warped_ddim, f"{save_dir}/warped_ddim.vtp")
                save_vtp_pointcloud(introp, f"{save_dir}/introp.vtp")
                save_vtp_pointcloud(preop, f"{save_dir}/preop.vtp")
                print(f"📁 已保存对比 VTP 到 {save_dir}")

if __name__ == '__main__':
    main()
