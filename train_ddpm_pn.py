import os
import torch
import datetime
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import math

from models.ddpm_pn import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from LiverDataset import LiverDataset

# === 配置 ===
LOG_NAME = 'liver_ddpm_experiment'
BATCH_SIZE = 3
NUM_EPOCHS = 300
LR = 1e-3
NUM_POINTS = 1024
DIFFUSION_STEPS = 300
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 时间步依赖的 pred_disp 权重函数 ===
def get_pred_disp_weight(t: torch.Tensor, T: int, alpha=5.0):
    """
    给定时间步 t，输出 pred_disp 权重，越小越依赖 gt_disp，越大越信任预测
    返回值范围：0 ~ 1
    """
    step = T - t.float()  # 越靠近 0，step 越大
    num = 1.0 - torch.exp(-alpha * step / T)
    denom = 1.0 - torch.exp(torch.tensor(-alpha, device=t.device))  # ← 修复点
    return num / denom  # shape: [B]

def main():
    # === 创建日志路径 ===
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = Path('./log') / LOG_NAME / time_str
    checkpoints_dir = exp_dir / 'checkpoints'
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    log_file = exp_dir / 'train_log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)

    def log(msg):
        print(msg)
        logging.info(msg)

    # === 加载数据集 ===
    log("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5000))  # 只取前n个样本
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    log(f"✅ 数据样本数: {len(dataset)}")

    # === 模型与优化器 ===
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === Warmup + Cosine Scheduler ===
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10  # 前10轮线性 warmup
        return 0.5 * (1 + math.cos((epoch - 10) / (NUM_EPOCHS - 10) * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)
    best_loss = float('inf')

    # === 开始训练 ===
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        log(f"\n🔁 Epoch {epoch}/{NUM_EPOCHS}")

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()           # [B, N, 3]
            introp = batch['introp'].to(DEVICE).float()         # [B, N, 3]
            gt_disp = batch['displacement'].to(DEVICE).float()  # [B, N, 3]
            disp_mean = batch['disp_mean'].to(DEVICE).float()   # [B, N, 3]
            disp_std = batch['disp_std'].to(DEVICE).float()     # [B, N, 3]

            t = torch.randint(0, diffusion.T, (BATCH_SIZE,), device=DEVICE).long()  # [B]
            x_t, eps = diffusion.add_noise(gt_disp, t)  # [B, N, 3], [B, N, 3]

            with torch.no_grad():
                # === 1. 预测 displacement
                zero_pred = torch.zeros_like(gt_disp)
                x0_pred = model.predict_noise_step(preop, introp, gt_disp*0, x_t, t, pred_disp=zero_pred)
                pred_disp_raw = x0_pred * disp_std + disp_mean  # 反标准化

                # === 2. 加权混合：pred_disp vs gt_disp
                w = get_pred_disp_weight(t, diffusion.T, alpha=5.0).view(BATCH_SIZE, 1, 1)  # [B,1,1]
                gt_disp_unnorm = gt_disp * disp_std + disp_mean
                pred_disp = w * pred_disp_raw + (1 - w) * gt_disp_unnorm  # 融合输入

            # === 3. 前向传播 ===
            predict_eps_fn = model(preop, introp, gt_disp, t, pred_disp=pred_disp, return_noise=True)
            pred_eps = predict_eps_fn(x_t)

            loss = F.mse_loss(pred_eps, eps)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        log(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")
        scheduler.step()
        log(f"[Epoch {epoch}] 当前学习率: {optimizer.param_groups[0]['lr']:.6e}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoints_dir / 'best_model.pth'
            log(f"✅ 保存最佳模型至: {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)

    # === 保存最终模型 ===
    torch.save(model.state_dict(), checkpoints_dir / 'last_model.pth')
    log("🏁 训练完成，已保存最终模型。")


if __name__ == '__main__':
    main()
