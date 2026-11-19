import torch


class DiffusionSchedule:
    def __init__(self, steps, device):
        self.steps = steps
        self.device = device
        beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        previous = torch.cat((torch.ones(1, device=device), self.alpha_bar[:-1]))
        self.posterior_variance = beta * (1.0 - previous) / (1.0 - self.alpha_bar)

    def add_noise(self, images, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(images)
        alpha_bar = self.alpha_bar[timesteps].view(-1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * images + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def sample(self, model, count, return_steps=False):
        model.eval()
        x = torch.randn(count, 1, 28, 28, device=self.device)
        wanted = {self.steps - 1, self.steps // 2, self.steps // 4, 0}
        trajectory = {}
        for index in reversed(range(self.steps)):
            t = torch.full((count,), index, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)
            alpha = self.alpha[index]
            alpha_bar = self.alpha_bar[index]
            beta = self.beta[index]
            x = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(
                alpha
            )
            if index > 0:
                x = x + self.posterior_variance[index].sqrt() * torch.randn_like(x)
            if return_steps and index in wanted:
                trajectory[index] = x.detach().cpu().clamp(-1, 1)
        if return_steps:
            return x.clamp(-1, 1), trajectory
        return x.clamp(-1, 1)
