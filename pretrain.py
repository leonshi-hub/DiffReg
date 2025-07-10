# === pretrain_pointnetpp_transformer.py — 预训练 encoder + transformer 用于配准 ===
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from LiverDataset import LiverDataset
from utils.util import get_xyz_positional_encoding, PC_distance
from models.ddpm2_pn2_posi import PointNetPPEncoder

# === 配置 ===
LOG_NAME = 'pretrain_pointnetpp_transformer'
BATCH_SIZE = 5
NUM_EPOCHS = 300
LR = 1e-4
NUM_POINTS = 1024
DATA_ROOT = '/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = './pretrain_ckpt'


class PretrainPointNetPPTransformer(nn.Module):
    def __init__(self, d_model=128, npoint=1024):
        super().__init__()
        self.encoder_pre = PointNetPPEncoder(npoint=npoint, d_model=d_model, use_pred_disp=True)
        self.encoder_int = PointNetPPEncoder(npoint=npoint, d_model=d_model, use_pred_disp=False)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=4, num_decoder_layers=4)
        self.linear = nn.Linear(d_model, 3)

    def forward(self, preop, introp):
        # preop, introp: [B, N, 3]
        B, N, _ = preop.shape
        pe_pre = get_xyz_positional_encoding(preop, d_pos=18)
        pe_int = get_xyz_positional_encoding(introp, d_pos=18)

        pred_disp = torch.zeros_like(preop)  # 预训练时使用零位移作为 dummy 条件
        x_input = torch.cat([preop, pe_pre], dim=-1).permute(0, 2, 1)  # [B, 27, N]
        y_input = torch.cat([introp, pe_int], dim=-1).permute(0, 2, 1)  # [B, 21, N]

        feat_pre = self.encoder_pre(x_input, pred_disp=pred_disp.permute(0, 2, 1))  # [B, d_model, N]
        feat_int = self.encoder_int(y_input)  # [B, d_model, N]
        feat_pre, feat_int = feat_pre.permute(2, 0, 1), feat_int.permute(2, 0, 1)  # [N, B, d]

        trans_feat = self.transformer(feat_pre, feat_int).permute(1, 0, 2)  # [B, N, d_model]
        disp = self.linear(trans_feat)  # [B, N, 3]
        return disp


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_dir = Path(SAVE_DIR) / f'{LOG_NAME}_{time_str}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("📦 加载 liver 数据...")
    dataset = LiverDataset(DATA_ROOT, num_points=NUM_POINTS, preload=False)
    dataset = Subset(dataset, range(5000))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"✅ 样本数: {len(dataset)}")

    model = PretrainPointNetPPTransformer(d_model=128, npoint=NUM_POINTS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}]"):
            preop = batch['preop'].to(DEVICE).float()
            introp = batch['introp'].to(DEVICE).float()

            pred_disp = model(preop, introp)  # [B, N, 3]
            warped = preop + pred_disp
            target_disp = introp - preop

            loss_disp = F.mse_loss(pred_disp, target_disp)
            loss_warped = F.mse_loss(warped, introp)
            pc_dist = PC_distance(warped.permute(0, 2, 1), introp.permute(0, 2, 1)).mean()

            loss = loss_disp + 0.5 * loss_warped + 0.1 * pc_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch}] 平均 Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')
            torch.save(model.encoder_pre.state_dict(), ckpt_dir / 'encoder_pre.pth')
            torch.save(model.encoder_int.state_dict(), ckpt_dir / 'encoder_int.pth')
            torch.save(model.transformer.state_dict(), ckpt_dir / 'transformer.pth')
            print(f"💾 保存最佳模型 -> {ckpt_dir / 'best_model.pth'}")

    torch.save(model.state_dict(), ckpt_dir / 'last_model.pth')
    print("🏁 预训练完成。")

if __name__ == '__main__':
    main()
