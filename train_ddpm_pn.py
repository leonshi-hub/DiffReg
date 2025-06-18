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
BATCH_SIZE = 4
NUM_EPOCHS = 300
LR = 1e-3
NUM_POINTS = 1024
DIFFUSION_STEPS = 300
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    log("\U0001F4E6 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5000))  # 只取前n个样本
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    log(f"数据样本数: {len(dataset)}")

    # === 模型与优化器 ===
    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS).to(DEVICE)
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
        log(f"\n\U0001F501 Epoch {epoch}/{NUM_EPOCHS}")

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()
            introp = batch['introp'].to(DEVICE).float()
            gt_disp = batch['displacement'].to(DEVICE).float()
            disp_mean = batch['disp_mean'].to(DEVICE).float()
            disp_std = batch['disp_std'].to(DEVICE).float()

            t = torch.randint(0, diffusion.T, (BATCH_SIZE,), device=DEVICE).long()
            x_t, eps = diffusion.add_noise(gt_disp, t)

            predict_eps_fn = model(preop, introp, gt_disp, t, return_noise=True)
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
