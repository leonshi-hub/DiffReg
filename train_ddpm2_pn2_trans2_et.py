# === train_ddpm2_pn2.py（加入监督损失）===
import os
import torch
import datetime
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
import math

from models.ddpm2_pn2_posi import TransformerDDPMRegNet
from utils.ddpm_schedule import DiffusionSchedule
from LiverDataset import LiverDataset

# === 配置 ===
LOG_NAME = 'liver_ddpm2_pn2_loss3_experiment'
BATCH_SIZE = 5
NUM_EPOCHS = 600
LR = 1e-4
NUM_POINTS = 1024
DIFFUSION_STEPS = 2000
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pred_disp_weight(t: torch.Tensor, T: int, alpha=5.0):
    step = T - t.float()
    num = 1.0 - torch.exp(-alpha * step / T)
    denom = 1.0 - torch.exp(torch.tensor(-alpha, device=t.device))
    return num / denom

def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def main():
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

    log("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(50000))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    log(f"✅ 数据样本数: {len(dataset)}")

    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        lr = get_lr(epoch, warmup_epochs=30, total_epochs=NUM_EPOCHS, base_lr=LR, min_lr=5e-6)
        for g in optimizer.param_groups:
            g['lr'] = lr
        log(f"[Epoch {epoch}] 当前学习率: {lr:.6e}")

        total_loss = 0
        log(f"\n🔁 Epoch {epoch}/{NUM_EPOCHS}")

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()          # [B, N, 3]
            introp = batch['introp'].to(DEVICE).float()
            gt_disp = batch['displacement'].to(DEVICE).float()
            disp_mean = batch['disp_mean'].to(DEVICE).float()
            disp_std = batch['disp_std'].to(DEVICE).float()

            t = torch.randint(0, diffusion.T, (BATCH_SIZE,), device=DEVICE).long()
            x_t, eps = diffusion.add_noise(gt_disp, t)

            # === 推理出 displacement 用于条件 ===
            with torch.no_grad():
                zero_pred = torch.zeros_like(gt_disp)
                x0_pred = model.predict_noise_step(preop, introp, gt_disp * 0, x_t, t, pred_disp=zero_pred)
                pred_disp_raw = x0_pred * disp_std + disp_mean

                w = get_pred_disp_weight(t, diffusion.T, alpha=5.0).view(BATCH_SIZE, 1, 1)
                gt_disp_unnorm = gt_disp * disp_std + disp_mean
                pred_disp = w * pred_disp_raw + (1 - w) * gt_disp_unnorm

            # === DDPM 主任务：预测 ε ===
            predict_eps_fn = model(preop, introp, gt_disp, t, pred_disp=pred_disp, return_noise=True)
            pred_eps = predict_eps_fn(x_t)

            # === 三重 loss ===
            loss_eps = F.mse_loss(pred_eps, eps)
            loss_disp = F.mse_loss(pred_disp_raw, gt_disp_unnorm)
            loss_warped = F.mse_loss(preop + pred_disp_raw, introp)

            loss = loss_eps + 0.5 * loss_disp + 0.5 * loss_warped
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        log(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")

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
