import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.ddpm_displacement import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from utils.diffusion_utils import ddim_sample
from LiverDataset import LiverDataset

# === 设置 ===
CHECKPOINT_PATH = '/home/shiliyuan/Projects/DiffReg/diffreg_pointnet_trans/log/liver_ddpm_experiment/2025-05-19_13-57-20/checkpoints/best_model.pth'
SAVE_ROOT = './log/liver_ddpm_experiment/eval_vis'
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
BATCH_SIZE = 1
NUM_POINTS = 1024
DDIM_STEPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_txt_pointcloud(points, save_path):
    """保存为 .txt 格式"""
    np.savetxt(save_path, points, fmt='%.6f')


def visualize_batch(preop, introp, warped, folder_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_txt_pointcloud(preop[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_preop.txt"))
    save_txt_pointcloud(introp[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_introp.txt"))
    save_txt_pointcloud(warped[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_warped.txt"))


@torch.no_grad()
def main():
    # === 加载模型 ===
    print("🚀 加载模型")
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # === 加载数据 ===
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    diffusion = DiffusionSchedule(T=1000, device=DEVICE)

    # === 可视化保存路径 ===
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    # === 推理并保存 ===
    print("🎨 开始采样与可视化")
    for batch in tqdm(dataloader, desc="Evaluating"):
        preop = batch['preop'].to(DEVICE).float()     # [1, N, 3]
        introp = batch['introp'].to(DEVICE).float()
        folder_name = batch['folder'][0]

        pred_disp = ddim_sample(model, preop, introp, diffusion, ddim_steps=DDIM_STEPS)
        warped = preop + pred_disp

        save_path = os.path.join(SAVE_ROOT, folder_name)
        visualize_batch(preop, introp, warped, folder_name, save_path)

    print("✅ 所有点云保存完成，可在 ParaView 或 Open3D 中查看")


if __name__ == '__main__':
    main()
