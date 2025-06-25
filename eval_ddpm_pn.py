# === 修改后的 eval_ddpm_pn.py ===
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.ddpm_pn import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from utils.diffusion_utils import ddim_sample_feedback
from LiverDataset import LiverDataset

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from torch.utils.data import Subset

# === 配置 ===
CHECKPOINT_PATH = 'log/liver_ddpm_experiment/2025-06-20_16-00-38/checkpoints/best_model.pth'
SAVE_ROOT = './log/liver_ddpm_experiment/eval2_vis'
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
BATCH_SIZE = 1
NUM_POINTS = 1024
DDIM_STEPS = 50
MAX_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_vtp_pointcloud(points: np.ndarray, save_path: str):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))
    polydata.SetPoints(vtk_points)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.Write()


def visualize_batch(preop, introp, warped, folder_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_vtp_pointcloud(preop[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_preop.vtp"))
    save_vtp_pointcloud(introp[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_introp.vtp"))
    save_vtp_pointcloud(warped[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_warped.vtp"))


@torch.no_grad()
def main():
    print("🚀 加载模型")
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset_all = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset_all, range(min(len(dataset_all), MAX_SAMPLES)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    diffusion = DiffusionSchedule(T=300, device=DEVICE)
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    print("🎨 开始采样与保存为 .vtp")
    for batch in tqdm(dataloader, desc="Evaluating"):
        preop = batch['preop'].to(DEVICE).float()
        introp = batch['introp'].to(DEVICE).float()
        disp_mean = batch['disp_mean'].to(DEVICE).float()
        disp_std = batch['disp_std'].to(DEVICE).float()
        folder_name = batch['folder'][0]

        disp_mean = disp_mean.view(1, NUM_POINTS, 3)
        disp_std = disp_std.view(1, NUM_POINTS, 3)

        # === DDIM采样 ===
        disp_pred = ddim_sample_feedback(
            model, preop, introp, disp_mean, disp_std, diffusion, ddim_steps=DDIM_STEPS
        )
        warped = preop + disp_pred

        sample_dir = os.path.join(SAVE_ROOT, folder_name)
        visualize_batch(preop, introp, warped, folder_name, sample_dir)

    print("✅ 所有 .vtp 点云已保存，可使用 ParaView 查看结果。")


if __name__ == '__main__':
    main()
