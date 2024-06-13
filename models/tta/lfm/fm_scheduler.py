from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class FlowMatchingTrainer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        init_type="gaussian",
        noise_scale=1.0,
        reflow_t_schedule="uniform",
        use_ode_sampler="euler",
        sigma_var=0.0,
        ode_tol=1e-5,
        sample_N=25,
    ):
        super().__init__()
        self.model = model
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.reflow_t_schedule = reflow_t_schedule
        self.use_ode_sampler = use_ode_sampler
        self.sigma_var = sigma_var
        self.ode_tol = ode_tol
        self.sample_N = sample_N
        self.T = 1
        self.eps = 1e-3
        self.sigma_t = lambda t: (1.0 - t) * sigma_var
        print("Init. Distribution Variance:", self.noise_scale)
        print("SDE Sampler Variance:", sigma_var)
        print("ODE Tolerence:", self.ode_tol)

    def forward(self, x_0, c):
        # get a random training step $t \sim Uniform({1, ..., T})$
        # t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - self.eps) + self.eps
        t_expand = t.view(-1, 1, 1, 1).repeat(
            1, x_0.shape[1], x_0.shape[2], x_0.shape[3]
        )
        c = c.to(x_0.device)

        noise = torch.randn_like(x_0)
        target = x_0 - noise
        perturbed_data = t_expand * x_0 + (1 - t_expand) * noise

        model_out = self.model(perturbed_data, t * 999, c)

        # get the gradient
        loss = F.mse_loss(model_out, target, reduction="none").mean([1, 2, 3]).mean()
        return loss

    @torch.no_grad()
    def euler_sample(self, cond, shape, guidance_scale):
        device = self.model.device
        batch = torch.randn(shape, device=device)
        x = torch.randn_like(batch)
        # uniform
        dt = 1.0 / self.sample_N
        eps = 1e-3
        for i in range(self.sample_N):
            num_t = i / self.sample_N * (self.T - eps) + eps
            t = torch.ones(batch.shape[0], device=device) * num_t

            model_out = self.model(torch.cat([x] * 2), torch.cat([t * 999] * 2), cond)
            # perform guidance
            noise_pred_uncond, noise_pred_text = model_out.chunk(2)
            pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # pred = self.apply_model(x, t*999, cond)

            sigma_t = self.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2) / (
                2 * (self.noise_scale**2) * ((1.0 - num_t) ** 2)
            ) * (
                0.5 * num_t * (1.0 - num_t) * pred
                - 0.5 * (2.0 - num_t) * x.detach().clone()
            )

            x = (
                x.detach().clone()
                + pred_sigma * dt
                + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
            )

        nfe = self.sample_N
        return x, nfe
