import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from LiverDataset import LiverDataset
from models.transformer_3d_tps import get_model
import numpy as np
import open3d as o3d

# === ✅ 配置 ===
CHECKPOINT_PATH = '/home/shiliyuan/Projects/DiffReg/diffreg_pointnet_trans/log/liver_flatten_safe/2025-05-17_17-56-48/checkpoints/best_model.pth'
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
BATCH_SIZE = 1
NPOINT = 1024
MAX_SAMPLES = 10
OUTPUT_DIR = './eval_output'

def write_ply(pc, filename, color):
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim == 3 and pc.shape[0] == 1:
        pc = pc.squeeze(0)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"点云格式错误：需要 [N, 3]，但得到 {pc.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(str(filename), pcd, format='ply')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = LiverDataset(DATA_ROOT, num_points=NPOINT, preload=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(d_model=128, channel=3, npoint=NPOINT).cuda()
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("✅ 开始保存前 100 个样本为 .ply 文件...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= MAX_SAMPLES:
                break

            preop = batch['preop'].cuda().float()
            introp = batch['introp'].cuda().float()
            warped = model(introp, preop)

            preop_np = preop[0].cpu().numpy().astype(np.float32)
            introp_np = introp[0].cpu().numpy().astype(np.float32)
            warped_np = warped[0].permute(1, 0).cpu().numpy().astype(np.float32)

            sample_dir = Path(OUTPUT_DIR) / f"{i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            write_ply(preop_np, sample_dir / 'preop.ply', [0, 0, 1])      # 蓝
            write_ply(introp_np, sample_dir / 'introp.ply', [1, 0, 0])    # 红
            write_ply(warped_np, sample_dir / 'warped.ply', [0, 1, 0])    # 绿

            print(f"✅ 已保存样本 {i+1}/{MAX_SAMPLES} 到 {sample_dir}")

if __name__ == '__main__':
    main()
