import torch


def make_beta_schedule(timesteps, mode='linear', beta_start=1e-4, beta_end=0.02):
    if mode == 'linear':
        betas = torch.linspace(beta_start, beta_end, timesteps)
    elif mode == 'cosine':
        # Optional: implement cosine schedule (more stable for large T)
        raise NotImplementedError("Cosine schedule not implemented yet.")
    else:
        raise ValueError(f"Unsupported schedule mode: {mode}")
    return betas


class DiffusionSchedule:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.device = device

        self.betas = make_beta_schedule(T).to(device)                  # βₜ
        self.alphas = 1.0 - self.betas                                 # αₜ
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)        # ᾱₜ

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def get_params(self, t: torch.Tensor):
        """
        Args:
            t: [B] integer tensor (timestep)
        Returns:
            (sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t): both shape [B, 1, 1]
        """
        t = t.long()
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar, sqrt_one_minus_alpha_bar

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Args:
            x0: clean signal (B, N, 3)
            t: timestep (B,)
            noise: optional Gaussian noise (B, N, 3)
        Returns:
            x_t: noised signal (B, N, 3)
            eps: used noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_params(t)
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
