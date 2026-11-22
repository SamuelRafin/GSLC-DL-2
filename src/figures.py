"""Penggambar figur yang hanya memakai numpy dan matplotlib.

Dipakai oleh run_experiment.py (create_figures) dan regenerate_figures.py.
Label baris/panel ditulis sebagai teks pada axes, sehingga tetap tampil
walaupun axis dimatikan.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _row_label(axis, text):
    axis.text(-0.10, 0.5, text, transform=axis.transAxes, ha="right", va="center",
              fontsize=11, weight="bold")


def save_reconstruction_grid(rows, path, title):
    """rows: daftar (label, array N x 28 x 28 pada rentang [0, 1]); satu baris per label."""
    n_rows, n_cols = len(rows), len(rows[0][1])
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    for r, (label, images) in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        _row_label(axes[r, 0], label)
    figure.suptitle(title)
    figure.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.01, wspace=0.03, hspace=0.03)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def tile_grid(images, rows=4, cols=4, pad=2):
    """Susun array (rows*cols) x H x W menjadi satu citra kisi rows x cols dengan jarak putih."""
    count, height, width = images.shape
    if count < rows * cols:
        raise ValueError(f"Butuh {rows * cols} citra, tersedia {count}.")
    canvas = np.ones((rows * (height + pad) + pad, cols * (width + pad) + pad), dtype=np.float32)
    for index in range(rows * cols):
        r, c = divmod(index, cols)
        top = pad + r * (height + pad)
        left = pad + c * (width + pad)
        canvas[top:top + height, left:left + width] = images[index]
    return canvas


def save_sample_grids(panels, path, title):
    """panels: daftar (nama model, array 16 x 28 x 28 pada [0, 1]); satu kisi 4x4 per panel."""
    figure, axes = plt.subplots(1, len(panels), figsize=(16, 4.9))
    for index, (axis, (name, images)) in enumerate(zip(axes, panels)):
        axis.imshow(tile_grid(images[:16]), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Panel {chr(ord('A') + index)}: {name}", fontsize=13, weight="bold")
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.05)
    figure.savefig(path, dpi=200, metadata={"Description": "layout=grid4x4"})
    plt.close(figure)


def save_interpolation_grid(rows, alpha_labels, path, title):
    """rows: daftar (label, array 6 x 28 x 28 pada [0, 1]); alpha_labels: teks alpha per kolom."""
    n_rows, n_cols = len(rows), len(rows[0][1])
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7))
    for r, (label, images) in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"alpha = {alpha_labels[c]}", fontsize=10)
        _row_label(axes[r, 0], label)
    figure.suptitle(title)
    figure.subplots_adjust(left=0.17, right=0.99, top=0.91, bottom=0.01, wspace=0.03, hspace=0.05)
    figure.savefig(path, dpi=200)
    plt.close(figure)
