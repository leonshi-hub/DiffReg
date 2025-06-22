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
MAX_SAMPLES = 100

# === 工具函数 ===
def to_o3d(pc, color):
    pc = np.asarray(pc)
    if pc.dtype != np.float32:
        pc = pc.astype(np.float32)
    if pc.ndim == 3 and pc.shape[0] == 1:
        pc = pc.squeeze(0)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"点云格式错误：需要 [N, 3]，但得到 {pc.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color(color)
    return pcd

def show_three(preop, introp, warped, title="point clouds"):
    pcd_preop = to_o3d(preop, [0, 0, 1])     # 蓝色：术前
    pcd_introp = to_o3d(introp, [1, 0, 0])   # 红色：目标术中
    pcd_warped = to_o3d(warped, [0, 1, 0])   # 绿色：形变后
    o3d.visualization.draw_geometries([pcd_preop, pcd_introp, pcd_warped],
                                      window_name=title,
                                      point_show_normal=False)

# === 主程序 ===
def main():
    print("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NPOINT, preload=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🧠 加载模型并恢复权重...")
    model = get_model(d_model=128, channel=3, npoint=NPOINT).cuda()
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("✅ 模型加载完毕，开始前 100 个样本的预测与可视化...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= MAX_SAMPLES:
                break

            preop = batch['preop'].cuda().float()
            introp = batch['introp'].cuda().float()
            warped = model(introp, preop)  # preop → warped ≈ introp

            preop_np = preop[0].cpu().numpy().astype(np.float32)           # [N, 3]
            introp_np = introp[0].cpu().numpy().astype(np.float32)         # [N, 3]
            warped_np = warped[0].permute(1, 0).cpu().numpy().astype(np.float32)  # [3, N] → [N, 3]

            print(f"\n🔍 第 {i+1}/{MAX_SAMPLES} 个样本")
            print(f"[Shape check] preop: {preop_np.shape}, introp: {introp_np.shape}, warped: {warped_np.shape}")
            show_three(preop_np, introp_np, warped_np, title=f"Sample {i+1}")

if __name__ == '__main__':
    main()
