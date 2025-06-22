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
    step = T - t.float()
    num = 1.0 - torch.exp(-alpha * step / T)
    denom = 1.0 - torch.exp(torch.tensor(-alpha, device=t.device))
    return num / denom

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
    dataset = Subset(dataset, range(5000))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    log(f"✅ 数据样本数: {len(dataset)}")

    # === 模型与优化器 ===
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === Cosine Annealing Warm Restarts Scheduler ===
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=6e-4
    )

    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)
    best_loss = float('inf')

    # === 开始训练 ===
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        log(f"\n🔁 Epoch {epoch}/{NUM_EPOCHS}")

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()
            introp = batch['introp'].to(DEVICE).float()
            gt_disp = batch['displacement'].to(DEVICE).float()
            disp_mean = batch['disp_mean'].to(DEVICE).float()
            disp_std = batch['disp_std'].to(DEVICE).float()

            t = torch.randint(0, diffusion.T, (BATCH_SIZE,), device=DEVICE).long()
            x_t, eps = diffusion.add_noise(gt_disp, t)

            with torch.no_grad():
                zero_pred = torch.zeros_like(gt_disp)
                x0_pred = model.predict_noise_step(preop, introp, gt_disp*0, x_t, t, pred_disp=zero_pred)
                pred_disp_raw = x0_pred * disp_std + disp_mean

                w = get_pred_disp_weight(t, diffusion.T, alpha=5.0).view(BATCH_SIZE, 1, 1)
                gt_disp_unnorm = gt_disp * disp_std + disp_mean
                pred_disp = w * pred_disp_raw + (1 - w) * gt_disp_unnorm

            predict_eps_fn = model(preop, introp, gt_disp, t, pred_disp=pred_disp, return_noise=True)
            pred_eps = predict_eps_fn(x_t)

            loss = F.mse_loss(pred_eps, eps)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        log(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")
        scheduler.step(epoch)
        log(f"[Epoch {epoch}] 当前学习率: {optimizer.param_groups[0]['lr']:.6e}")

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

    torch.save(model.state_dict(), checkpoints_dir / 'last_model.pth')
    log("🏁 训练完成，已保存最终模型。")

if __name__ == '__main__':
    main()
