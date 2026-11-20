import time
from functools import lru_cache

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.nn import functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


@torch.no_grad()
def reconstruction_metrics(model, loader, device, model_type="ae"):
    model.eval()
    mse_values = []
    psnr_values = []
    ssim_values = []
    for images, _ in loader:
        images = images.to(device)
        if model_type == "vae":
            mu, _ = model.encode(images)
            reconstructions = model.decode(mu)
        else:
            reconstructions = model(images)
        mse = (images - reconstructions).square().flatten(1).mean(1)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))

        kernel = 7
        mu_x = F.avg_pool2d(images, kernel, stride=1, padding=kernel // 2)
        mu_y = F.avg_pool2d(reconstructions, kernel, stride=1, padding=kernel // 2)
        var_x = F.avg_pool2d(images.square(), kernel, 1, kernel // 2) - mu_x.square()
        var_y = (
            F.avg_pool2d(reconstructions.square(), kernel, 1, kernel // 2)
            - mu_y.square()
        )
        cov_xy = (
            F.avg_pool2d(images * reconstructions, kernel, 1, kernel // 2)
            - mu_x * mu_y
        )
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        )
        mse_values.append(mse.cpu())
        psnr_values.append(psnr.cpu())
        ssim_values.append(ssim.flatten(1).mean(1).cpu())

    return {
        "test_mse": torch.cat(mse_values).mean().item(),
        "test_psnr_db": torch.cat(psnr_values).mean().item(),
        "test_ssim": torch.cat(ssim_values).mean().item(),
    }


@lru_cache(maxsize=2)
def _inception_model(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.AuxLogits = None
    model.aux_logits = False
    return model.eval().to(device)


@torch.no_grad()
def _inception_features(model, batches, device):
    features = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for images in batches:
        images = images.to(device).clamp(0, 1)
        images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images - mean) / std
        features.append(model(images).cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(real_features, generated_features):
    real_mean = real_features.mean(axis=0)
    generated_mean = generated_features.mean(axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    generated_cov = np.cov(generated_features, rowvar=False)
    diagonal = np.eye(real_cov.shape[0]) * 1e-6
    real_cov += diagonal
    generated_cov += diagonal
    cov_mean = sqrtm(real_cov @ generated_cov)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    difference = real_mean - generated_mean
    score = difference @ difference + np.trace(real_cov + generated_cov - 2 * cov_mean)
    return float(max(score, 0.0))


@torch.no_grad()
def fid_score(generator, loader, device, sample_count=1024, batch_size=32, kind="vae",
              diffusion=None, noise_dim=100):
    model = _inception_model(device)
    real_batches = []
    remaining = sample_count
    for images, _ in loader:
        take = min(remaining, images.shape[0])
        real_batches.append(images[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"Test set has fewer than {sample_count} samples.")
    real_features = _inception_features(model, real_batches, device)

    generated_features = []
    remaining = sample_count
    generator.eval()
    while remaining:
        count = min(batch_size, remaining)
        if kind == "ddpm":
            generated = diffusion.sample(generator, count)
            generated = (generated + 1.0) / 2.0
        elif kind == "vae":
            generated = generator.decode(
                torch.randn(count, noise_dim, device=device)
            )
        else:
            noise = torch.randn(count, noise_dim, device=device)
            if kind == "dcgan":
                generated = generator(noise.reshape(count, noise_dim, 1, 1))
            else:
                generated = generator(noise)
            generated = (generated + 1.0) / 2.0 if kind in ("gan", "dcgan") else generated
        generated_batches = [generated]
        generated_features.append(_inception_features(model, generated_batches, device))
        remaining -= count
    return frechet_distance(real_features, np.concatenate(generated_features, axis=0))


@torch.no_grad()
def inference_latency_ms(generate, device, warmup=10, repeats=50):
    for _ in range(warmup):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        generate()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeats
