import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.ddpm_displacement import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from LiverDataset import LiverDataset

import vtk
from vtk.util.numpy_support import numpy_to_vtk
from torch.utils.data import Subset

# === 配置 ===
CHECKPOINT_PATH = './log/liver_ddpm_experiment/2025-05-20_14-37-08/checkpoints/best_model.pth'
SAVE_ROOT = './log/liver_ddpm_experiment/eval_vis_onestep'
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
BATCH_SIZE = 1
NUM_POINTS = 1024
MAX_SAMPLES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_vtp_pointcloud(points: np.ndarray, save_path: str):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))
    polydata.SetPoints(vtk_points)

    # 添加 vertex cells 确保 ParaView 可视化
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    polydata.SetVerts(vertices)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.Write()


def visualize_batch(preop, introp, warped, folder_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_vtp_pointcloud(preop[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_preop.vtp"))
    save_vtp_pointcloud(introp[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_introp.vtp"))
    save_vtp_pointcloud(warped[0].cpu().numpy(), os.path.join(save_dir, f"{folder_name}_onestep_denoised.vtp"))


@torch.no_grad()
def main():
    print("🚀 加载模型")
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset_all = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset_all, range(min(len(dataset_all), MAX_SAMPLES)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    diffusion = DiffusionSchedule(T=1000, device=DEVICE)
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    print("🔬 开始一步加噪 + 一步去噪验证")
    for batch in tqdm(dataloader, desc="1-step Denoising Test"):
        preop = batch['preop'].to(DEVICE).float()
        introp = batch['introp'].to(DEVICE).float()
        gt_disp = batch['displacement'].to(DEVICE).float()
        disp_mean = batch['disp_mean'].to(DEVICE).float()
        disp_std = batch['disp_std'].to(DEVICE).float()
        folder_name = batch['folder'][0]

        t = torch.randint(0, diffusion.T, (BATCH_SIZE,), device=DEVICE).long()
        x_t, eps = diffusion.add_noise(gt_disp, t)

        predict_eps_fn = model(preop, introp, t, return_noise=True)
        pred_eps = predict_eps_fn(x_t)

        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = diffusion.get_params(t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar * pred_eps) / sqrt_alpha_bar
        x0_pred = x0_pred * disp_std + disp_mean
        warped = preop + x0_pred

        sample_dir = os.path.join(SAVE_ROOT, folder_name)
        visualize_batch(preop, introp, warped, folder_name, sample_dir)

    print("✅ 一步加噪 + 去噪输出完成，可在 ParaView 中查看 onestep_denoised.vtp")


if __name__ == '__main__':
    main()
# 该脚本用于在 LiverDataset 上进行一步加噪 + 去噪验证，并将结果保存为 .vtp 文件以供 ParaView 可视化
# 该脚本假设你已经有了一个训练好的模型，并且数据集路径和模型路径已正确设置
# 你可以根据需要调整 MAX_SAMPLES 来控制评估的样本数量
# 该脚本使用了 PyTorch 和 VTK 库来处理数据和可视化
# 请确保你已经安装了这些库