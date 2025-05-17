import os
import datetime
import importlib
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from LiverDataset import LiverDataset
from tqdm import tqdm

# === 固定参数（保留 transformer_3d_tps 的 flatten 接口）===
MODEL_NAME = 'transformer_3d_tps'
BATCH_SIZE = 2            # ⛳ 显存控制核心：小 batch 避免 flatten 炸
NPOINT = 1024              # ⛳ 尽量避免用 1024
EPOCHS = 30
LR = 0.001
GPU = '0'
LOG_DIR = 'liver_flatten_safe'
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

def main():
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = Path('./log') / LOG_DIR / time_str
    checkpoints_dir = exp_dir / 'checkpoints'
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    log_file = exp_dir / f'{MODEL_NAME}.txt'
    logger = logging.getLogger(MODEL_NAME)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file))

    def log(msg): print(msg); logger.info(msg)

    log("✅ Step 1: 加载 liver 数据")
    dataset = LiverDataset(DATA_ROOT, num_points=NPOINT, preload=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    log(f"总样本数：{len(dataset)}")

    log("🧠 Step 2: 加载模型")
    MODEL = importlib.import_module(f'models.{MODEL_NAME}')
    model = MODEL.get_model(d_model=128, channel=3, npoint=NPOINT).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float('inf')

    log("🎯 Step 3: 开始训练")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        print(f"🌊 Epoch {epoch}/{EPOCHS} 开始...")

        for batch in tqdm(loader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].cuda().float()
            introp = batch['introp'].cuda().float()

            optimizer.zero_grad()
            warped, loss = model(introp, preop)
            loss.mean().backward()
            optimizer.step()

            total_loss += loss.mean().item()

        avg_loss = total_loss / len(loader)
        log(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoints_dir / 'best_model.pth'
            log(f"→ 保存最佳模型至 {save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)

    log("💾 训练完成，保存最终模型")
    torch.save(model.state_dict(), checkpoints_dir / 'last_model.pth')


if __name__ == '__main__':
    main()
