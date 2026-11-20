import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.utils import save_image

from data import make_loaders
from diffusion import DiffusionSchedule
from metrics import (
    fid_score,
    inference_latency_ms,
    reconstruction_metrics,
)
from models import (
    Autoencoder,
    DCDiscriminator,
    DCGenerator,
    FrozenLatentDecoder,
    MLPClassifier,
    MLPDiscriminator,
    MLPGenerator,
    SmallUNet,
    VariationalAutoencoder,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_autoencoder(model, train_loader, val_loader, device, epochs, lr, sparse_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "sparsity": 0.0}
        count = 0
        for images, _ in train_loader:
            images = images.to(device)
            hidden, latent = model.encode_with_hidden(images)
            reconstructed = model.decode(latent)
            reconstruction = F.mse_loss(reconstructed, images)
            sparsity = hidden.abs().mean() if model.sparse else reconstruction.new_zeros(())
            loss = reconstruction + sparse_weight * sparsity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch = images.shape[0]
            totals["loss"] += loss.item() * batch
            totals["reconstruction"] += reconstruction.item() * batch
            totals["sparsity"] += sparsity.item() * batch
            count += batch
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                val_loss = F.mse_loss(model(images), images)
                val_total += val_loss.item() * images.shape[0]
                val_count += images.shape[0]
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": totals["loss"] / count,
                "reconstruction": totals["reconstruction"] / count,
                "sparsity": totals["sparsity"] / count,
                "val_loss": val_total / val_count,
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"  epoch {epoch + 1}/{epochs}: loss={history[-1]['loss']:.5f}")
    return history


def train_vae(model, train_loader, val_loader, device, epochs, lr, beta):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        total_loss = total_reconstruction = total_kl = 0.0
        count = 0
        for images, _ in train_loader:
            images = images.to(device)
            reconstructed, mu, logvar = model(images)
            reconstruction = F.binary_cross_entropy(
                reconstructed, images, reduction="sum"
            ) / images.shape[0]
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            loss = reconstruction + beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch = images.shape[0]
            total_loss += loss.item() * batch
            total_reconstruction += reconstruction.item() * batch
            total_kl += kl.item() * batch
            count += batch
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                reconstructed, mu, logvar = model(images)
                reconstruction = F.binary_cross_entropy(
                    reconstructed, images, reduction="sum"
                ) / images.shape[0]
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
                val_total += (reconstruction + beta * kl).item() * images.shape[0]
                val_count += images.shape[0]
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / count,
                "reconstruction": total_reconstruction / count,
                "kl": total_kl / count,
                "val_loss": val_total / val_count,
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"  epoch {epoch + 1}/{epochs}: loss={history[-1]['loss']:.3f}")
    return history


def train_gan(generator, discriminator, train_loader, device, epochs, lr, noise_dim):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        generator.train()
        discriminator.train()
        total_g = total_d = 0.0
        batches = 0
        for images, _ in train_loader:
            real = images.to(device) * 2 - 1
            batch = real.shape[0]
            real_targets = torch.ones(batch, device=device)
            fake_targets = torch.zeros(batch, device=device)

            noise = torch.randn(batch, noise_dim, device=device)
            fake = generator(noise).detach()
            loss_d = criterion(discriminator(real), real_targets) + criterion(
                discriminator(fake), fake_targets
            )
            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            noise = torch.randn(batch, noise_dim, device=device)
            loss_g = criterion(discriminator(generator(noise)), real_targets)
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_g.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            total_g += loss_g.item()
            total_d += loss_d.item()
            batches += 1
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "generator": total_g / batches,
                "discriminator": total_d / batches,
                "loss": total_g / batches,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"  epoch {epoch + 1}/{epochs}: G={history[-1]['generator']:.3f}, "
            f"D={history[-1]['discriminator']:.3f}"
        )
    return history


def train_ddpm(model, schedule, train_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        for images, _ in train_loader:
            images = images.to(device) * 2 - 1
            timesteps = torch.randint(
                0, schedule.steps, (images.shape[0],), device=device
            )
            noisy, noise = schedule.add_noise(images, timesteps)
            predicted_noise = model(noisy, timesteps)
            loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / batches,
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"  epoch {epoch + 1}/{epochs}: noise_loss={history[-1]['loss']:.5f}")
    return history


def train_classifier(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        total_loss = correct = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.shape[0]
        model.eval()
        val_correct = val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()
                val_count += labels.shape[0]
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / count,
                "train_accuracy": correct / count,
                "val_accuracy": val_correct / val_count,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"  epoch {epoch + 1}/{epochs}: loss={history[-1]['loss']:.4f}, "
            f"val_acc={history[-1]['val_accuracy']:.3f}"
        )
    return history


def train_frozen_decoder(classifier, decoder, train_loader, device, epochs, lr):
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        decoder.train()
        total_loss = 0.0
        count = 0
        for images, _ in train_loader:
            images = images.to(device)
            with torch.no_grad():
                latent = classifier.encode(images)
            reconstruction = decoder(latent)
            loss = F.mse_loss(reconstruction, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            count += images.shape[0]
        sync(device)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / count,
                "seconds": time.perf_counter() - started,
            }
        )
    return history


@torch.no_grad()
def classifier_accuracy(model, loader, device):
    model.eval()
    correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        count += labels.shape[0]
    return correct / count


@torch.no_grad()
def classifier_statistics(model, loader, device):
    model.eval()
    latent_batches = []
    for images, _ in loader:
        latent_batches.append(model.encode(images.to(device)).cpu())
    latents = torch.cat(latent_batches)
    weights = {
        "input_to_hidden": model.fc1.weight,
        "hidden_to_latent": model.fc2.weight,
        "latent_to_class": model.fc3.weight,
    }
    diagnostics = {}
    for name, weight in weights.items():
        weight = weight.detach().double()
        gram = weight.T @ weight
        identity = torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        diagnostics[name] = {
            "shape": list(weight.shape),
            "rank": int(torch.linalg.matrix_rank(weight).item()),
            "normalized_gram_error": (
                torch.linalg.matrix_norm(gram - identity) / math.sqrt(gram.shape[0])
            ).item(),
        }
    mean = latents.mean(0)
    std = latents.std(0, unbiased=False).clamp_min(1e-6)
    return {
        "latent_mean": mean.tolist(),
        "latent_std": std.tolist(),
        "weight_diagnostics": diagnostics,
    }


def save_checkpoint(path, model, **extra):
    torch.save({"model": model.state_dict(), **extra}, path)


def main():
    parser = argparse.ArgumentParser(
        description="Eksperimen komparatif generative models pada Fashion-MNIST."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--classifier-epochs", type=int, default=25)
    parser.add_argument("--decoder-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--ddpm-steps", type=int, default=100)
    parser.add_argument("--sparse-weight", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1.0)
    parser.add_argument("--fid-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", default="outputs")
    parser.add_argument(
        "--load-checkpoints",
        help="Load trained model checkpoints from this directory and skip training.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        help="Path to a state_dict checkpoint created by the matching MLPClassifier.",
    )
    parser.add_argument("--skip-fid", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi torch.cuda.is_available() bernilai False.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    set_seed(args.seed)
    output = Path(args.output)
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    checkpoints.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Perangkat: {device}; sampel: 56.000 train, 7.000 val, 7.000 test")

    models = {}
    histories = {}
    ae = Autoencoder(latent_dim=args.latent_dim).to(device)
    sparse_ae = Autoencoder(sparse=True, latent_dim=args.latent_dim).to(device)
    vae = VariationalAutoencoder(latent_dim=args.latent_dim).to(device)
    gan = MLPGenerator(args.noise_dim).to(device)
    gan_discriminator = MLPDiscriminator().to(device)
    dcgan = DCGenerator(args.noise_dim).to(device)
    dc_discriminator = DCDiscriminator().to(device)
    ddpm = SmallUNet().to(device)
    diffusion = DiffusionSchedule(args.ddpm_steps, device)
    classifier = MLPClassifier(latent_dim=args.latent_dim).to(device)
    frozen_decoder = FrozenLatentDecoder(latent_dim=args.latent_dim).to(device)

    jobs = [
        ("Vanilla AE", "vanilla_ae", ae),
        ("Sparse AE", "sparse_ae", sparse_ae),
        ("VAE", "vae", vae),
        ("Vanilla GAN", "gan", gan),
        ("DCGAN", "dcgan", dcgan),
        ("DDPM", "ddpm", ddpm),
    ]
    checkpoint_source = Path(args.load_checkpoints) if args.load_checkpoints else None
    if checkpoint_source is not None:
        print(f"\nMemuat checkpoint model dari {checkpoint_source}")
        for _, key, model in jobs:
            checkpoint_path = checkpoint_source / f"{key}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint model tidak ditemukan: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            if not isinstance(checkpoint, dict) or "model" not in checkpoint:
                raise ValueError(f"Format checkpoint model tidak valid: {checkpoint_path}")
            model.load_state_dict(checkpoint["model"], strict=True)
            models[key] = model
            histories[key] = []
    else:
        for index, (label, key, model) in enumerate(jobs):
            print(f"\nMelatih {label}")
            set_seed(args.seed + index)
            train_loader.generator.manual_seed(args.seed)
            if key in ("vanilla_ae", "sparse_ae"):
                history = train_autoencoder(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    args.epochs,
                    1e-3,
                    args.sparse_weight if key == "sparse_ae" else 0.0,
                )
            elif key == "vae":
                history = train_vae(
                    model, train_loader, val_loader, device, args.epochs, 1e-3, args.vae_beta
                )
            elif key == "gan":
                history = train_gan(
                    model,
                    gan_discriminator,
                    train_loader,
                    device,
                    args.epochs,
                    2e-4,
                    args.noise_dim,
                )
            elif key == "dcgan":
                history = train_gan(
                    model,
                    dc_discriminator,
                    train_loader,
                    device,
                    args.epochs,
                    2e-4,
                    args.noise_dim,
                )
            else:
                history = train_ddpm(
                    model, diffusion, train_loader, device, args.epochs, 2e-4
                )
            models[key] = model
            histories[key] = history
            save_checkpoint(checkpoints / f"{key}.pt", model, seed=args.seed + index)
            if key == "gan":
                save_checkpoint(checkpoints / "gan_discriminator.pt", gan_discriminator)
            if key == "dcgan":
                save_checkpoint(checkpoints / "dcgan_discriminator.pt", dc_discriminator)

    if checkpoint_source is not None:
        classifier_path = checkpoint_source / "classifier.pt"
        if not classifier_path.is_file():
            raise FileNotFoundError(f"Checkpoint classifier tidak ditemukan: {classifier_path}")
        classifier_checkpoint = torch.load(
            classifier_path, map_location=device, weights_only=True
        )
        if not isinstance(classifier_checkpoint, dict) or "model" not in classifier_checkpoint:
            raise ValueError(f"Format checkpoint classifier tidak valid: {classifier_path}")
        classifier.load_state_dict(classifier_checkpoint["model"], strict=True)
        histories["classifier"] = []
    elif args.classifier_checkpoint:
        checkpoint_path = Path(args.classifier_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint classifier tidak ditemukan: {checkpoint_path}")
        print(f"\nMemuat MLP classifier dari {checkpoint_path}")
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if not isinstance(state_dict, dict) or not all(
            isinstance(value, torch.Tensor) for value in state_dict.values()
        ):
            raise ValueError(
                "Checkpoint harus berupa state_dict langsung, seperti keluaran "
                "torch.save(model.state_dict(), 'mlp_classifier.pt')."
            )
        classifier.load_state_dict(state_dict, strict=True)
        histories["classifier"] = []
    else:
        print("\nMelatih MLP classifier untuk eksperimen batasan laten")
        set_seed(args.seed + 20)
        train_loader.generator.manual_seed(args.seed)
        histories["classifier"] = train_classifier(
            classifier, train_loader, val_loader, device, args.classifier_epochs, 1e-3
        )
    save_checkpoint(checkpoints / "classifier.pt", classifier)
    classifier_test_accuracy = classifier_accuracy(classifier, test_loader, device)
    if classifier_test_accuracy < 0.88:
        print(
            f"PERINGATAN: akurasi test MLP {classifier_test_accuracy:.2%}, "
            "di bawah target 88% pada instruksi tugas."
        )
    if checkpoint_source is not None:
        decoder_path = checkpoint_source / "frozen_decoder.pt"
        if not decoder_path.is_file():
            raise FileNotFoundError(f"Checkpoint frozen decoder tidak ditemukan: {decoder_path}")
        decoder_checkpoint = torch.load(
            decoder_path, map_location=device, weights_only=True
        )
        if not isinstance(decoder_checkpoint, dict) or "model" not in decoder_checkpoint:
            raise ValueError(f"Format checkpoint decoder tidak valid: {decoder_path}")
        frozen_decoder.load_state_dict(decoder_checkpoint["model"], strict=True)
        histories["frozen_decoder"] = []
    else:
        print("Melatih decoder pada fitur classifier yang dibekukan")
        histories["frozen_decoder"] = train_frozen_decoder(
            classifier, frozen_decoder, train_loader, device, args.decoder_epochs, 1e-3
        )
    save_checkpoint(checkpoints / "frozen_decoder.pt", frozen_decoder)

    print("\nMenghitung metrik pada test set")
    probe_stats = classifier_statistics(classifier, test_loader, device)
    results = {
        "dataset": "Fashion-MNIST",
        "seed": args.seed,
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "split_counts": {"train": 56000, "validation": 7000, "test": 7000},
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "ddpm_steps": args.ddpm_steps,
        "sparse_weight": args.sparse_weight,
        "vae_beta": args.vae_beta,
        "models": {},
        "fid": {
            "samples": args.fid_samples,
            "skipped": args.skip_fid,
            "feature_extractor": "ImageNet-pretrained Inception v3, grayscale repeated to RGB",
        },
        "configuration": {
            "latent_dim": args.latent_dim,
            "noise_dim": args.noise_dim,
            "hidden_dim": 256,
            "optimizer": "Adam",
            "learning_rates": {
                "autoencoder_vae_classifier_decoder": 1e-3,
                "gan_dcgan_ddpm": 2e-4,
            },
            "classifier_epochs": args.classifier_epochs,
            "decoder_epochs": args.decoder_epochs,
            "sparse_weight": args.sparse_weight,
            "vae_beta": args.vae_beta,
            "ddpm_steps": args.ddpm_steps,
            "batch_size": args.batch_size,
            "fid_samples": args.fid_samples,
            "classifier_checkpoint": args.classifier_checkpoint,
            "loaded_checkpoints": str(checkpoint_source) if checkpoint_source else None,
        },
    }
    for key in ("vanilla_ae", "sparse_ae", "vae"):
        model_type = "vae" if key == "vae" else "ae"
        results["models"][key] = reconstruction_metrics(
            models[key], test_loader, device, model_type=model_type
        )
    results["classifier"] = {
        "test_accuracy": classifier_test_accuracy,
        **probe_stats,
    }
    results["models"]["frozen_mlp_decoder"] = reconstruction_metrics(
        _FeatureDecoder(classifier, frozen_decoder), test_loader, device
    )

    generator_specs = [
        ("vae", vae, "vae"),
        ("gan", gan, "gan"),
        ("dcgan", dcgan, "dcgan"),
        ("ddpm", ddpm, "ddpm"),
    ]
    for key in ("vanilla_ae", "sparse_ae"):
        model = models[key]
        model.eval()
        fixed_batch = torch.zeros(64, 1, 28, 28, device=device)
        results["models"][key]["latency_ms_batch64"] = inference_latency_ms(
            lambda model=model, fixed_batch=fixed_batch: model(fixed_batch), device
        )
    for key, model, kind in generator_specs:
        model.eval()

        def generate(model=model, kind=kind):
            noise_dim = args.latent_dim if kind == "vae" else args.noise_dim
            noise = torch.randn(64, noise_dim, device=device)
            if kind == "vae":
                return model.decode(noise)
            if kind == "ddpm":
                return diffusion.sample(model, 64)
            return model(noise)

        results["models"].setdefault(key, {})
        results["models"][key]["latency_ms_batch64"] = inference_latency_ms(
            generate,
            device,
            warmup=1 if kind == "ddpm" else 10,
            repeats=3 if kind == "ddpm" else 50,
        )
        results["models"][key]["latency_repeats"] = 3 if kind == "ddpm" else 50
        if not args.skip_fid:
            print(f"  FID: {key}")
            results["models"][key]["fid"] = fid_score(
                model,
                test_loader,
                device,
                sample_count=args.fid_samples,
                kind=kind,
                diffusion=diffusion,
                noise_dim=args.latent_dim if kind == "vae" else args.noise_dim,
            )

    results["training_seconds_per_epoch"] = {
        key: [row["seconds"] for row in history] for key, history in histories.items()
    }
    (output / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    history_fields = sorted(
        {field for history in histories.values() for row in history for field in row}
    )
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", *history_fields])
        writer.writeheader()
        for key, history in histories.items():
            for row in history:
                writer.writerow({"model": key, **row})
    create_figures(
        figures,
        models,
        histories,
        classifier,
        frozen_decoder,
        test_loader,
        diffusion,
        device,
        args,
        probe_stats,
    )
    print(f"\nSelesai. Hasil tersimpan di {output.resolve()}")


class _FeatureDecoder(nn.Module):
    def __init__(self, classifier, decoder):
        super().__init__()
        self.classifier = classifier
        self.decoder = decoder

    def forward(self, images):
        return self.decoder(self.classifier.encode(images))


@torch.no_grad()
def create_figures(
    figures, models, histories, classifier, frozen_decoder, test_loader,
    diffusion, device, args, probe_stats
):
    batch = next(iter(test_loader))[0][:8].to(device)
    reconstructions = {
        "Asli": batch,
        "Vanilla AE": models["vanilla_ae"](batch),
        "Sparse AE": models["sparse_ae"](batch),
        "VAE": models["vae"](batch)[0],
    }
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for row, (label, images) in enumerate(reconstructions.items()):
        for column, image in enumerate(images):
            axes[row, column].imshow(image.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=55, va="center")
    fig.suptitle("Figur 1. Rekonstruksi pada delapan citra test")
    fig.tight_layout()
    fig.savefig(figures / "figure_1_reconstructions.png", dpi=200)
    plt.close(fig)

    samples = {
        "VAE": models["vae"].decode(torch.randn(16, args.latent_dim, device=device)),
        "Vanilla GAN": models["gan"](torch.randn(16, args.noise_dim, device=device)),
        "DCGAN": models["dcgan"](torch.randn(16, args.noise_dim, device=device)),
        "DDPM": diffusion.sample(models["ddpm"], 16),
    }
    fig, axes = plt.subplots(4, 16, figsize=(16, 5))
    for row, (label, images) in enumerate(samples.items()):
        if label != "VAE":
            images = (images + 1) / 2
        for column, image in enumerate(images):
            axes[row, column].imshow(image.squeeze().cpu().clamp(0, 1), cmap="gray")
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=65, va="center")
    fig.suptitle("Figur 2. Sampling acak (16 sampel per model)")
    fig.tight_layout()
    fig.savefig(figures / "figure_2_generated_samples.png", dpi=200)
    plt.close(fig)

    original = batch[:1]
    latent = classifier.encode(original)
    class_vector = F.one_hot(classifier(original).argmax(1), num_classes=10).float()
    class_latent = F.relu(F.linear(class_vector, classifier.fc3.weight.t()))
    class_hidden = F.relu(F.linear(class_latent, classifier.fc2.weight.t()))
    class_projection = torch.sigmoid(F.linear(class_hidden, classifier.fc1.weight.t())).reshape(
        -1, 1, 28, 28
    )
    latent_hidden = F.relu(F.linear(latent, classifier.fc2.weight.t()))
    latent_projection = torch.sigmoid(
        F.linear(latent_hidden, classifier.fc1.weight.t())
    ).reshape(
        -1, 1, 28, 28
    )
    frozen_reconstruction = frozen_decoder(latent)
    ae_reconstruction = models["vanilla_ae"](original)
    mu, _ = models["vae"].encode(original)
    mlp_mean = torch.tensor(probe_stats["latent_mean"], device=device).unsqueeze(0)
    mlp_std = torch.tensor(probe_stats["latent_std"], device=device).unsqueeze(0)
    prior_latent = mlp_mean + mlp_std * torch.randn_like(latent)
    prior_reconstruction = frozen_decoder(prior_latent)
    panels = [
        ("Asli", original),
        ("Bobot balik: kelas", class_projection),
        ("Bobot balik: latent", latent_projection),
        ("Decoder latent MLP", frozen_reconstruction),
        ("Rekonstruksi AE", ae_reconstruction),
        ("Prior latent MLP", prior_reconstruction),
        ("Rekonstruksi VAE", models["vae"].decode(mu)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 2.7))
    for axis, (label, image) in zip(axes, panels):
        axis.imshow(image[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(label, fontsize=8)
        axis.axis("off")
    fig.suptitle("Figur 3. Proyeksi bobot dan decoding latent MLP")
    fig.tight_layout()
    fig.savefig(figures / "figure_3_classifier_limits.png", dpi=200)
    plt.close(fig)

    images, labels = next(iter(test_loader))
    first_index = 0
    second_index = next(
        (i for i in range(len(labels)) if labels[i] != labels[first_index]), 1
    )
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
        rows.append((key.replace("_", " ").title(), model.decode(
            torch.cat([(1 - alpha) * z_a + alpha * z_b for alpha in alphas])
        )))
    for key, title in (("gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        z_a = torch.randn(1, args.noise_dim, device=device)
        z_b = torch.randn(1, args.noise_dim, device=device)
        z = torch.cat([(1 - alpha) * z_a + alpha * z_b for alpha in alphas])
        generated = models[key](z)
        rows.append((title, (generated + 1) / 2))
    fig, axes = plt.subplots(len(rows), 6, figsize=(10, 7))
    for row, (label, images) in enumerate(rows):
        for column, image in enumerate(images):
            axes[row, column].imshow(image.squeeze().cpu().clamp(0, 1), cmap="gray")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(f"α={alphas[column].item():.1f}")
        axes[row, 0].set_ylabel(label, rotation=0, labelpad=65, va="center")
    fig.suptitle("Figur 4. Interpolasi linier pada ruang laten")
    fig.tight_layout()
    fig.savefig(figures / "figure_4_interpolation.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, history in histories.items():
        if key == "classifier" or key == "frozen_decoder":
            continue
        if key in ("gan", "dcgan"):
            axes[0].plot(
                [row["epoch"] for row in history],
                [row["generator"] for row in history],
                label=f"{key} G",
            )
            axes[0].plot(
                [row["epoch"] for row in history],
                [row["discriminator"] for row in history],
                linestyle="--",
                label=f"{key} D",
            )
        else:
            axes[0].plot(
                [row["epoch"] for row in history],
                [row["loss"] for row in history],
                label=key,
            )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    _, trajectory = diffusion.sample(models["ddpm"], 1, return_steps=True)
    ordered = sorted(trajectory.items(), reverse=True)
    trajectory_images = torch.cat([image for _, image in ordered], dim=0)
    axes[1].imshow(
        torch.cat([image[0] for image in trajectory_images], dim=1),
        cmap="gray",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_title("DDPM reverse trajectory (T -> 0)")
    axes[1].set_xticks([14 + 28 * index for index in range(len(ordered))])
    axes[1].set_xticklabels([f"t={timestep}" for timestep, _ in ordered], fontsize=8)
    axes[1].set_yticks([])
    axes[1].tick_params(axis="x", length=0, pad=2)
    fig.suptitle("Figur 5. Kurva training dan proses denoising")
    fig.tight_layout()
    fig.savefig(figures / "figure_5_training_and_diffusion.png", dpi=200)
    plt.close(fig)

    for name, image in (
        ("fixed_test_samples.png", batch),
        ("fixed_interpolation_endpoints.png", endpoints),
    ):
        save_image(image.cpu(), figures / name, nrow=8)


if __name__ == "__main__":
    main()
