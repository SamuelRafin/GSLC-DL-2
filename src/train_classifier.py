"""Tahap 1: melatih MLP classifier 784 -> 256 -> 64 -> 10 pada Fashion-MNIST.

Memakai split 80:10:10 seed 42 yang sama dengan pipeline (data.make_loaders),
jadi test split tidak pernah dilihat saat training. Bobot yang disimpan adalah
bobot epoch terakhir; test set tidak dipakai untuk memilih checkpoint.

Contoh:
    python src/train_classifier.py --epochs 20 --checkpoint mlp_classifier.pt
    python src/train_classifier.py --evaluate mlp_classifier.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch

from data import make_loaders
from models import MLPClassifier
from run_experiment import classifier_accuracy, set_seed, train_classifier


def pick_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi torch.cuda.is_available() bernilai False.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description="Latih atau evaluasi MLP classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--checkpoint", default="mlp_classifier.pt",
                        help="Lokasi menyimpan state_dict hasil training.")
    parser.add_argument("--log", default="classifier_training.json",
                        help="Lokasi menyimpan log training (JSON).")
    parser.add_argument("--evaluate", metavar="PATH",
                        help="Hanya evaluasi checkpoint ini pada split val/test, tanpa training.")
    args = parser.parse_args()

    device = pick_device(args.device)
    set_seed(args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
    )
    print(f"Perangkat: {device}; sampel: 56.000 train, 7.000 val, 7.000 test")
    model = MLPClassifier().to(device)

    if args.evaluate:
        state_dict = torch.load(args.evaluate, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        val_accuracy = classifier_accuracy(model, val_loader, device)
        test_accuracy = classifier_accuracy(model, test_loader, device)
        print(f"Val accuracy : {val_accuracy:.2%}")
        print(f"Test accuracy: {test_accuracy:.2%} (target minimal 88%)")
        return

    train_loader.generator.manual_seed(args.seed)
    started = time.perf_counter()
    history = train_classifier(
        model, train_loader, val_loader, device, args.epochs, args.lr
    )
    total_seconds = time.perf_counter() - started

    val_accuracy = classifier_accuracy(model, val_loader, device)
    test_accuracy = classifier_accuracy(model, test_loader, device)
    print(f"\nVal accuracy : {val_accuracy:.2%}")
    print(f"Test accuracy: {test_accuracy:.2%} (target minimal 88%)")
    if test_accuracy < 0.88:
        print("PERINGATAN: akurasi test di bawah target 88%; tambah epoch lalu jalankan ulang.")

    # state_dict langsung, formatnya sama dengan --classifier-checkpoint di run_experiment.py
    torch.save(model.state_dict(), args.checkpoint)
    shapes = {name: list(w.shape) for name, w in model.state_dict().items()}
    log = {
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "split_counts": {"train": 56000, "validation": 7000, "test": 7000},
        "total_training_seconds": total_seconds,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "weight_shapes": shapes,
        "torch": torch.__version__,
        "history": history,
    }
    Path(args.log).write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Checkpoint: {args.checkpoint}; log: {args.log}")


if __name__ == "__main__":
    main()
