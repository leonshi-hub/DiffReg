import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def ddim_sample(model, preop, introp, diffusion, ddim_steps=50, eta=0.0):
    """
    DDIM sampling: 从 x_T ~ N(0,1) 开始，在 transformer 条件下一步步去噪
    Args:
        model: 你的 Transformer + DDPM 模型
        preop, introp: [B, N, 3]
        diffusion: DiffusionSchedule 实例
        ddim_steps: 采样步数（通常为 25~100）
        eta: 噪声调节参数（0 表示 deterministic）
    Returns:
        warped: [B, N, 3] 最终生成的 displacement field
    """
    B, N, _ = preop.shape
    device = preop.device
    T = diffusion.T

    # 1. 准备时间步序列（等距）
    times = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)

    # 2. 初始噪声
    x_t = torch.randn(B, N, 3, device=device)

    # 3. 获取模型的条件函数
    model.eval()
    predict_eps_fn = model(preop, introp, times[0].expand(B), return_noise=True)

    for i in tqdm(range(ddim_steps), desc="DDIM Sampling"):
        t = times[i].expand(B)  # 当前步
        t_next = times[i + 1].expand(B) if i < ddim_steps - 1 else torch.zeros_like(t)

        alpha_bar_t = diffusion.alphas_cumprod[t].view(B, 1, 1)
        alpha_bar_next = diffusion.alphas_cumprod[t_next].view(B, 1, 1)

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t)

        eps_theta = predict_eps_fn(x_t)

        # 4. 预测 x0（干净 displacement）
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        x0_pred = torch.clamp(x0_pred, -1., 1.)

        # 5. 计算下一个 x_t（DDIM）
        sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_next)
        noise = torch.randn_like(x_t) if i < ddim_steps - 1 else torch.zeros_like(x_t)

        x_t = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps_theta + sigma * noise

    return x0_pred  # 最终输出的 displacement field
