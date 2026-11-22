"""Membuat ulang Figur 1, 2, dan 4 dari checkpoint tanpa melatih ulang.

Figur 1 diberi label baris, Figur 2 disusun sebagai empat kisi 4x4 (Panel A-D),
dan Figur 4 diberi label baris. Figur 3 dan 5 tidak diubah.

Contoh:
    python src/regenerate_figures.py --checkpoints outputs/checkpoints \
        --output outputs/figures --ddpm-steps 1000
"""
import argparse
from pathlib import Path

import torch

from data import make_loaders
from diffusion import DiffusionSchedule
from figures import save_interpolation_grid, save_reconstruction_grid, save_sample_grids
from models import (
    Autoencoder,
    DCGenerator,
    MLPGenerator,
    SmallUNet,
    VariationalAutoencoder,
)
from run_experiment import set_seed


def load_models(checkpoint_dir, args, device):
    models = {
        "vanilla_ae": Autoencoder(latent_dim=args.latent_dim),
        "sparse_ae": Autoencoder(sparse=True, latent_dim=args.latent_dim),
        "vae": VariationalAutoencoder(latent_dim=args.latent_dim),
        "gan": MLPGenerator(args.noise_dim),
        "dcgan": DCGenerator(args.noise_dim),
        "ddpm": SmallUNet(),
    }
    for key, model in models.items():
        path = Path(checkpoint_dir) / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
    return models


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Buat ulang Figur 1, 2, dan 4.")
    parser.add_argument("--checkpoints", default="outputs/checkpoints")
    parser.add_argument("--output", default="outputs/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--ddpm-steps", type=int, default=1000,
                        help="Harus sama dengan nilai saat training (laporan memakai 1000).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi torch.cuda.is_available() bernilai False.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    set_seed(args.seed)
    figures = Path(args.output)
    figures.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = make_loaders(
        batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers, device=device
    )
    models = load_models(args.checkpoints, args, device)
    diffusion = DiffusionSchedule(args.ddpm_steps, device)

    # Figur 1: delapan citra test pertama, deterministik
    batch = next(iter(test_loader))[0][:8].to(device)
    reconstructions = {
        "Asli": batch,
        "Vanilla AE": models["vanilla_ae"](batch),
        "Sparse AE": models["sparse_ae"](batch),
        "VAE": models["vae"](batch)[0],
    }
    save_reconstruction_grid(
        [(label, images.detach().cpu().squeeze(1).numpy()) for label, images in reconstructions.items()],
        figures / "figure_1_reconstructions.png",
        "Figur 1. Rekonstruksi pada delapan citra test",
    )

    # Figur 2: 16 sampel per model dalam kisi 4x4 (Panel A-D)
    samples = {
        "VAE": models["vae"].decode(torch.randn(16, args.latent_dim, device=device)),
        "Vanilla GAN": models["gan"](torch.randn(16, args.noise_dim, device=device)),
        "DCGAN": models["dcgan"](torch.randn(16, args.noise_dim, device=device)),
        "DDPM": diffusion.sample(models["ddpm"], 16),
    }
    panels = []
    for label, images in samples.items():
        if label != "VAE":
            images = (images + 1) / 2
        panels.append((label, images.detach().cpu().clamp(0, 1).squeeze(1).numpy()))
    save_sample_grids(
        panels,
        figures / "figure_2_generated_samples.png",
        "Figur 2. Sampling acak, kisi 4x4 per model",
    )

    # Figur 4: interpolasi linier antara dua citra test berkelas berbeda
    images, labels = next(iter(test_loader))
    first_index = 0
    second_index = next((i for i in range(len(labels)) if labels[i] != labels[first_index]), 1)
    endpoints = images[[first_index, second_index]].to(device)
    alphas = torch.linspace(0, 1, 6, device=device)
    rows = []
    for key in ("vanilla_ae", "sparse_ae", "vae"):
        model = models[key]
        if key == "vae":
            z_a = model.encode(endpoints[:1])[0]
            z_b = model.encode(endpoints[1:])[0]
        else:
            z_a, z_b = model.encode(endpoints[:1]), model.encode(endpoints[1:])
        rows.append(({"vanilla_ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE"}[key], model.decode(
            torch.cat([(1 - alpha) * z_a + alpha * z_b for alpha in alphas])
        )))
    for key, title in (("gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        z_a = torch.randn(1, args.noise_dim, device=device)
        z_b = torch.randn(1, args.noise_dim, device=device)
        z = torch.cat([(1 - alpha) * z_a + alpha * z_b for alpha in alphas])
        rows.append((title, (models[key](z) + 1) / 2))
    save_interpolation_grid(
        [(label, images.detach().cpu().clamp(0, 1).squeeze(1).numpy()) for label, images in rows],
        [f"{alpha:.1f}" for alpha in alphas.tolist()],
        figures / "figure_4_interpolation.png",
        "Figur 4. Interpolasi linier pada ruang laten",
    )
    print(f"Figur 1, 2, dan 4 tersimpan di {figures.resolve()}")


if __name__ == "__main__":
    main()
