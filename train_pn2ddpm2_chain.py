# === train_ddpm2_pn2_loss3_chain.py（链式disp_input训练，三重loss）===
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

LOG_NAME = 'liver_ddpm2_pn2_loss3_chain_pre'
BATCH_SIZE = 3
NUM_EPOCHS = 600
LR = 1e-4
NUM_POINTS = 1024
DIFFUSION_STEPS = 2000
CHAIN_STEPS = 5            # 控制链式步数
PER_STEP_BACKWARD = False     # 控制是否每一步都做 backward
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_pretrained_modules(model, pretrain_path, device='cuda'):
    print(f"📂 加载预训练权重文件：")
    pretrain_state = torch.load(pretrain_path, map_location=device)

    for submodule_name in ['encoder_pre', 'encoder_int', 'transformer']:
        if hasattr(model, submodule_name):
            submodule = getattr(model, submodule_name)
            prefix = f"{submodule_name}."
            filtered_state = {
                k.replace(prefix, ""): v
                for k, v in pretrain_state.items()
                if k.startswith(prefix)
            }
            missing_keys, unexpected_keys = submodule.load_state_dict(filtered_state, strict=False)
            print(f"✅ {submodule_name} 加载成功（{len(filtered_state)}参数）")
            if missing_keys:
                print(f"⚠️ 缺失 key: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 多余 key: {unexpected_keys}")

def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def get_pred_disp_weight(t: torch.Tensor, T: int, alpha=5.0):
    step = T - t.float()
    num = 1.0 - torch.exp(-alpha * step / T)
    denom = 1.0 - torch.exp(torch.tensor(-alpha, device=t.device))
    return num / denom

def main():
    print("🔧 配置参数:")
    print(f"LOG_NAME = '{LOG_NAME}'")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"NUM_EPOCHS = {NUM_EPOCHS}")  # 固定值
    print(f"LR = {LR}")          # 固定值
    print(f"NUM_POINTS = {NUM_POINTS}")
    print(f"DIFFUSION_STEPS = {DIFFUSION_STEPS}")
    print(f"🗂 当前实验日志目录: ./log/{LOG_NAME}/<时间戳>/")
    print(f"CHAIN_STEPS = {CHAIN_STEPS}")

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = Path('./log') / LOG_NAME / time_str
    checkpoints_dir = exp_dir / 'checkpoints'
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(10000))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TransformerDDPMRegNet(d_model=128, npoint=NUM_POINTS, use_pred_disp=True).to(DEVICE)
    load_pretrained_modules(model, 'pretrain_ckpt/pretrain_pointnetpp_transformer_2025-07-09_13-58-33/best_model.pth', device=DEVICE)

    print(f"✅ 已加载预训练 encoder 权重")
   
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    diffusion = DiffusionSchedule(T=DIFFUSION_STEPS, device=DEVICE)
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        lr = get_lr(epoch, warmup_epochs=30, total_epochs=NUM_EPOCHS, base_lr=LR, min_lr=5e-6)
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f"[Epoch {epoch}] 当前学习率: {lr:.6e}")

        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()
            introp = batch['introp'].to(DEVICE).float()
            gt_disp = batch['displacement'].to(DEVICE).float()
            disp_mean = batch['disp_mean'].to(DEVICE).float()
            disp_std = batch['disp_std'].to(DEVICE).float()

            B, N, _ = gt_disp.shape
            t0 = torch.randint(CHAIN_STEPS, DIFFUSION_STEPS, (1,), device=DEVICE).item()
            t_seq = list(reversed(range(t0 - CHAIN_STEPS + 1, t0 + 1)))

            x_t = diffusion.add_noise(gt_disp, torch.full((B,), t0, device=DEVICE, dtype=torch.long))[0]
            disp_cur = torch.zeros_like(gt_disp)

            optimizer.zero_grad()
            total_step_loss = 0.0

            for t_val in t_seq:
                t = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)
                pred_eps = model.predict_noise_step(preop, introp, disp_cur, x_t, t, pred_disp=disp_cur)

                sqrt_alpha_bar, sqrt_one_minus_alpha_bar = diffusion.get_params(t)
                eps_gt = (x_t - sqrt_alpha_bar * gt_disp) / sqrt_one_minus_alpha_bar
                loss_eps = F.mse_loss(pred_eps, eps_gt)

                x0_pred = (x_t - sqrt_one_minus_alpha_bar * pred_eps) / sqrt_alpha_bar
                x0_pred = torch.clamp(x0_pred, -1., 1.)
                disp_cur = x0_pred.detach()
                x_t = sqrt_alpha_bar * x0_pred + sqrt_one_minus_alpha_bar * pred_eps

                w = get_pred_disp_weight(t, DIFFUSION_STEPS, alpha=5.0).view(B, 1, 1)
                pred_disp_raw = x0_pred * disp_std + disp_mean
                gt_disp_unnorm = gt_disp * disp_std + disp_mean
                pred_disp = w * pred_disp_raw + (1 - w) * gt_disp_unnorm

                loss_disp = F.mse_loss(pred_disp_raw, gt_disp_unnorm)
                loss_warped = F.mse_loss(preop + pred_disp_raw, introp)
                step_loss = loss_eps + 0.5 * loss_disp + 0.5 * loss_warped

                if PER_STEP_BACKWARD:
                    step_loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    total_step_loss += step_loss

            if not PER_STEP_BACKWARD:
                total_step_loss.backward()
                optimizer.step()

            total_loss += total_step_loss.item() if not PER_STEP_BACKWARD else 0.0

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoints_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)

    torch.save(model.state_dict(), checkpoints_dir / 'last_model.pth')
    print("✅ 训练完成")

if __name__ == '__main__':
    main()
