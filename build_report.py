"""Membangun laporan PDF (Bagian I-VI sesuai panduan tugas D6421 Sesi 07).

Semua angka pada tabel dan narasi dibaca dari metrics.json dan
training_history.json di dalam arsip hasil, jadi laporan ikut berubah bila
eksperimen dijalankan ulang.
"""
import argparse
import json
import math
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT = 0.075, 0.075
CONTENT_WIDTH = 1 - LEFT - RIGHT
BOTTOM_LIMIT = 0.05
INK = "#243b53"
NAVY = "#102a43"
TEAL = "#0b7285"
GREY = "#486581"

MODEL_NAMES = [
    ("Vanilla AE", "vanilla_ae"),
    ("Sparse AE", "sparse_ae"),
    ("VAE", "vae"),
    ("Vanilla GAN", "gan"),
    ("DCGAN", "dcgan"),
    ("DDPM", "ddpm"),
]


# ---------------------------------------------------------------- utilities
def idn(value, decimals=2):
    """Format angka gaya Indonesia: 1.234,56."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def pct(value, decimals=2):
    return idn(value * 100, decimals) + "%"


def chars_per_line(font_size, width_frac=CONTENT_WIDTH):
    return max(10, int(width_frac * PAGE_W * 72 / (0.54 * font_size)))


def wrap_lines(text, font_size, width_frac=CONTENT_WIDTH):
    width = chars_per_line(font_size, width_frac)
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(
                    raw, width=width, break_long_words=False, break_on_hyphens=False
                )
                or [""]
            )
    return lines


def check(y, label):
    if y < BOTTOM_LIMIT:
        print(f"PERINGATAN: konten melewati batas bawah di {label} (y={y:.3f})")


def new_page(title, number, subtitle=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    fig.text(LEFT, 0.975, "DEEP LEARNING (D6421)  |  STUDI GENERATIVE MODELS",
             fontsize=7.5, color=GREY, va="top")
    fig.text(1 - RIGHT, 0.975, "Fashion-MNIST", fontsize=7.5, color=GREY,
             ha="right", va="top")
    fig.text(LEFT, 0.945, title, fontsize=14.5, weight="bold", color=NAVY, va="top")
    if subtitle:
        fig.text(LEFT, 0.912, subtitle, fontsize=8.6, color=GREY, va="top")
    fig.add_artist(plt.Line2D([LEFT, 1 - RIGHT], [0.895, 0.895],
                              transform=fig.transFigure, color="#bcccdc", lw=0.8))
    fig.text(0.5, 0.025, f"Samuel Rafin Djunaidi  |  LB01  |  {number}",
             fontsize=7.5, color="#627d98", ha="center", va="bottom")
    return fig


def paragraph(fig, text, y, fs=8.5, leading=1.28, x=LEFT, width_frac=CONTENT_WIDTH, gap=0.010):
    lines = wrap_lines(text, fs, width_frac)
    fig.text(x, y, "\n".join(lines), fontsize=fs, color=INK, va="top", linespacing=leading)
    return y - max(1, len(lines)) * fs / (PAGE_H * 72) * leading - gap


def heading(fig, text, y, fs=10.2, color=TEAL):
    fig.text(LEFT, y, text, fontsize=fs, weight="bold", color=color, va="top")
    return y - (fs * 1.9) / (PAGE_H * 72)


def part_heading(fig, text, y):
    fig.text(LEFT, y, text, fontsize=12.5, weight="bold", color=NAVY, va="top")
    fig.add_artist(plt.Line2D([LEFT, 1 - RIGHT], [y - 0.024, y - 0.024],
                              transform=fig.transFigure, color="#bcccdc", lw=0.6))
    return y - 0.038


def eq(fig, label, tex, y, fs=8.4, label_w=0.13, gap=0.030):
    if label:
        fig.text(LEFT, y, label, fontsize=fs - 0.4, weight="bold", color=NAVY, va="top")
    fig.text(LEFT + label_w, y, tex, fontsize=fs, color=INK, va="top")
    return y - gap


def draw_table(fig, y_top, col_fracs, header, rows, fs=7.0, pad_x=0.004, pad_y=0.0035,
               bold_first=True):
    xs = [LEFT]
    for frac in col_fracs:
        xs.append(xs[-1] + frac * CONTENT_WIDTH)
    line_h = fs * 1.22 / (PAGE_H * 72)

    def wrap_cell(text, i):
        usable = col_fracs[i] * CONTENT_WIDTH - 2 * pad_x
        width = max(4, int(usable * PAGE_W * 72 / (0.56 * fs)))
        return textwrap.wrap(str(text), width, break_long_words=False,
                             break_on_hyphens=False) or [""]

    table_rows = [header] + rows
    y = y_top
    for r_index, row in enumerate(table_rows):
        wrapped = [wrap_cell(cell, i) for i, cell in enumerate(row)]
        height = max(len(c) for c in wrapped) * line_h + 2 * pad_y
        background = "#e6f4f1" if r_index == 0 else ("#f8fafc" if r_index % 2 == 0 else "white")
        fig.add_artist(Rectangle((LEFT, y - height), CONTENT_WIDTH, height,
                                 transform=fig.transFigure, facecolor=background,
                                 edgecolor="#d9e2ec", lw=0.5))
        for c_index, lines in enumerate(wrapped):
            bold = r_index == 0 or (c_index == 0 and bold_first)
            fig.text(xs[c_index] + pad_x, y - pad_y, "\n".join(lines), fontsize=fs,
                     va="top", color=NAVY if bold else INK,
                     weight="bold" if bold else "normal", linespacing=1.22)
        y -= height
    for x in xs[1:-1]:
        fig.add_artist(plt.Line2D([x, x], [y_top, y], transform=fig.transFigure,
                                  color="#d9e2ec", lw=0.4))
    return y - 0.008


def put_image(fig, image, y_top, width_frac, caption=None, x=None, cap_fs=7.4):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    if x is None:
        x = LEFT + (CONTENT_WIDTH - width_frac) / 2
    height_frac = width_frac * PAGE_W * image.height / image.width / PAGE_H
    axis = fig.add_axes([x, y_top - height_frac, width_frac, height_frac])
    axis.imshow(image)
    axis.axis("off")
    y = y_top - height_frac
    if caption:
        lines = wrap_lines(caption, cap_fs)
        fig.text(LEFT, y - 0.006, "\n".join(lines), fontsize=cap_fs, color=GREY, va="top")
        y -= 0.006 + len(lines) * cap_fs * 1.3 / (PAGE_H * 72)
    return y - 0.010


def trim_white(image, pad=8, threshold=245):
    array = np.asarray(image.convert("L"))
    mask = array < threshold
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    left = max(cols[0] - pad, 0)
    right = min(cols[-1] + pad, image.width)
    top = max(rows[0] - pad, 0)
    bottom = min(rows[-1] + pad, image.height)
    return image.crop((left, top, right, bottom))


def split_classifier_figure(path):
    """Figur 3 asli berisi 7 panel dalam satu baris; dipecah jadi dua baris agar terbaca."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    top = int(0.135 * height)
    cut = int(0.566 * width)
    row_a = trim_white(image.crop((0, top, cut, height)), pad=4)
    row_b = trim_white(image.crop((cut, top, width, height)), pad=4)
    return row_a, row_b


# ------------------------------------------------------------ data helpers
def dig(data, *keys, default=None):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def trend_word(first, last):
    return "naik" if last > first else "turun"


def collapse_gaps(image, min_run=24, keep=8, threshold=245):
    """Pangkas baris putih panjang di tengah citra agar crop tidak boros ruang."""
    array = np.asarray(image.convert("RGB"))
    blank = (array.min(axis=2) >= threshold).all(axis=1)
    kept, index, total = [], 0, len(blank)
    while index < total:
        if blank[index]:
            end = index
            while end < total and blank[end]:
                end += 1
            kept.extend(range(index, index + keep if end - index > min_run else end))
            index = end
        else:
            kept.append(index)
            index += 1
    return Image.fromarray(array[kept])


def extract_trajectory(original_figure):
    source = Image.open(original_figure).convert("RGB")
    crop = source.crop((int(source.width * 0.52), int(source.height * 0.12),
                        source.width, int(source.height * 0.95)))
    return collapse_gaps(trim_white(crop, pad=12))


def make_training_figure(history, output_path):
    figure = plt.figure(figsize=(12, 6.4))
    grid = figure.add_gridspec(2, 6, hspace=0.55, wspace=1.0)
    axes = {
        "ae": figure.add_subplot(grid[0, 0:2]),
        "vae": figure.add_subplot(grid[0, 2:4]),
        "ddpm": figure.add_subplot(grid[0, 4:6]),
        "gan": figure.add_subplot(grid[1, 0:3]),
        "dcgan": figure.add_subplot(grid[1, 3:6]),
    }
    for key, label in (("vanilla_ae", "Vanilla AE"), ("sparse_ae", "Sparse AE")):
        values = [row["reconstruction"] for row in history[key]]
        axes["ae"].plot(range(1, len(values) + 1), values, label=label)
    axes["ae"].set_title("Loss rekonstruksi (MSE)", fontsize=11)
    axes["ae"].set_ylabel("MSE")
    axes["ae"].legend(fontsize=9)
    values = [row["loss"] for row in history["vae"]]
    axes["vae"].plot(range(1, len(values) + 1), values, color="#2a9d8f")
    axes["vae"].set_title("VAE: BCE + KL per sampel", fontsize=11)
    axes["vae"].set_ylabel("Loss")
    values = [row["loss"] for row in history["ddpm"]]
    axes["ddpm"].plot(range(1, len(values) + 1), values, color="#6a4c93")
    axes["ddpm"].set_title("DDPM: prediksi derau", fontsize=11)
    axes["ddpm"].set_ylabel("MSE(eps, eps_theta)")
    for key, title in (("gan", "Vanilla GAN"), ("dcgan", "DCGAN")):
        gen = [row["generator"] for row in history[key]]
        dis = [row["discriminator"] for row in history[key]]
        x = range(1, len(gen) + 1)
        axes[key].plot(x, gen, label="Generator")
        axes[key].plot(x, dis, label="Discriminator")
        axes[key].axhline(math.log(2), color="#9fb3c8", ls=":", lw=1)
        axes[key].axhline(2 * math.log(2), color="#9fb3c8", ls=":", lw=1)
        axes[key].text(len(gen), math.log(2), " G ideal = ln2", fontsize=7.5,
                       color=GREY, ha="right", va="bottom")
        axes[key].text(len(gen), 2 * math.log(2), " D ideal = 2 ln2", fontsize=7.5,
                       color=GREY, ha="right", va="bottom")
        axes[key].set_title(title, fontsize=11)
        axes[key].set_ylabel("BCE loss")
        axes[key].legend(fontsize=9, loc="center left")
    for axis in axes.values():
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.09)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def draw_architecture(fig, y_top, height):
    """Diagram skematik: setiap baris satu jaringan, kotak = lapisan, panah = aliran data."""
    colors = {"io": "#dbeafe", "fc": "#e6f4f1", "lat": "#fff3bf", "sp": "#ffe3e3", "cv": "#ffe8cc"}
    rows = [
        ("Vanilla AE", [("x\n784", "io"), ("Linear 256\nReLU", "fc"), ("z (64)\nReLU", "lat"),
                        ("Linear 256\nReLU", "fc"), ("x-hat 784\nSigmoid", "io")]),
        ("Sparse AE", [("x\n784", "io"), ("h (256)\nReLU + L1", "sp"), ("z (64)\nReLU", "lat"),
                       ("Linear 256\nReLU", "fc"), ("x-hat 784\nSigmoid", "io")]),
        ("VAE", [("x\n784", "io"), ("Linear 256\nReLU", "fc"), ("mu, log var\n64 + 64", "lat"),
                 ("z = mu + s*e\n64", "lat"), ("Linear 256\nReLU", "fc"), ("x-hat 784\nSigmoid", "io")]),
        ("Vanilla GAN\nGenerator", [("z ~ N(0,I)\n100", "io"), ("Linear 256\nLeakyReLU", "fc"),
                                    ("Linear 512\nBN, LeakyReLU", "fc"), ("Linear 784\nTanh", "io")]),
        ("Vanilla GAN\nDiscriminator", [("x atau G(z)\n784", "io"), ("Linear 512\nLeakyReLU, Drop", "fc"),
                                        ("Linear 256\nLeakyReLU, Drop", "fc"), ("Linear 1\nlogit", "io")]),
        ("DCGAN\nGenerator", [("z\n100x1x1", "io"), ("ConvT 7x7\n128x7x7, BN, ReLU", "cv"),
                              ("ConvT 4x4 s2\n64x14x14, BN, ReLU", "cv"), ("ConvT 4x4 s2\n1x28x28, Tanh", "io")]),
        ("DCGAN\nDiscriminator", [("x atau G(z)\n1x28x28", "io"), ("Conv 4x4 s2\n64x14x14, LReLU", "cv"),
                                  ("Conv 4x4 s2\n128x7x7, BN, LReLU", "cv"), ("Conv 7x7\n1 logit", "io")]),
        ("DDPM\nU-Net", [("x_t dan t\n1x28x28", "io"), ("Conv + Res 64\n28x28 + emb. t", "cv"),
                         ("Pool, Res 128\n14x14", "cv"), ("Pool, Res 128\n7x7 (tengah)", "cv"),
                         ("Up + skip\nRes 64 (x2)", "cv"), ("eps-pred\n1x28x28", "io")]),
        ("MLP\nClassifier", [("x\n784", "io"), ("Linear 256\nReLU", "fc"), ("z_MLP (64)\nReLU", "lat"),
                             ("Linear 10\nlogits", "io")]),
    ]
    axis = fig.add_axes([LEFT, y_top - height, CONTENT_WIDTH, height])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, len(rows))
    axis.axis("off")
    for index, (label, boxes) in enumerate(rows):
        yc = len(rows) - index - 0.5
        axis.text(0, yc, label, fontsize=6.6, weight="bold", color=NAVY, va="center")
        x_start, x_end = 15.5, 100
        count = len(boxes)
        gap = 2.6
        box_w = (x_end - x_start - gap * (count - 1)) / count
        for b_index, (text, kind) in enumerate(boxes):
            x0 = x_start + b_index * (box_w + gap)
            axis.add_patch(FancyBboxPatch((x0, yc - 0.34), box_w, 0.68,
                                          boxstyle="round,pad=0.0,rounding_size=0.08",
                                          facecolor=colors[kind], edgecolor="#9fb3c8", lw=0.5))
            axis.text(x0 + box_w / 2, yc, text, fontsize=5.5, ha="center", va="center", color=INK)
            if b_index < count - 1:
                axis.annotate("", xy=(x0 + box_w + gap, yc), xytext=(x0 + box_w, yc),
                              arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#627d98",
                                              mutation_scale=5))


# -------------------------------------------------------------------- main
def build_report(archive_path, output_path, name, nim, class_name, repository, figures_dir=None):
    output_path = Path(output_path)
    assets = output_path.parent / "_report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    figure_names = [
        "figure_1_reconstructions.png",
        "figure_2_generated_samples.png",
        "figure_3_classifier_limits.png",
        "figure_4_interpolation.png",
        "figure_5_training_and_diffusion.png",
    ]
    with zipfile.ZipFile(archive_path) as bundle:
        metrics = json.loads(bundle.read("DL2-final/metrics.json"))
        history = json.loads(bundle.read("DL2-final/training_history.json"))
        for figure_name in figure_names:
            (assets / figure_name).write_bytes(bundle.read(f"DL2-final/figures/{figure_name}"))

    if figures_dir:
        for figure_name in figure_names:
            candidate = Path(figures_dir) / figure_name
            if candidate.is_file():
                shutil.copyfile(candidate, assets / figure_name)
                print(f"Memakai {figure_name} dari {figures_dir}")

    training_figure = assets / "figure_5_readable.png"
    make_training_figure(history, training_figure)
    trajectory_image = extract_trajectory(assets / "figure_5_training_and_diffusion.png")
    figure2_probe = Image.open(assets / "figure_2_generated_samples.png")
    figure2_is_grid = "grid4x4" in str(figure2_probe.info.get("Description", ""))
    if not figure2_is_grid:
        print("PERINGATAN: Figur 2 masih berupa baris 1x16, bukan kisi 4x4 per model. "
              "Jalankan src/regenerate_figures.py lalu bangun ulang arsip hasil.")
    fig3_row_a, fig3_row_b = split_classifier_figure(assets / "figure_3_classifier_limits.png")

    models = metrics["models"]
    ae, sae, vae = models["vanilla_ae"], models["sparse_ae"], models["vae"]
    gan, dcgan, ddpm = models["gan"], models["dcgan"], models["ddpm"]
    frozen = models["frozen_mlp_decoder"]
    classifier = metrics["classifier"]
    weights = classifier["weight_diagnostics"]
    accuracy = classifier["test_accuracy"]
    env = metrics.get("environment", {})
    config = metrics.get("configuration", {})
    seed = metrics.get("seed", 42)
    batch_size = metrics.get("batch_size", 128)
    epochs = metrics.get("epochs", 30)
    ddpm_steps = metrics.get("ddpm_steps", 1000)
    fid_samples = dig(metrics, "fid", "samples", default=1024)
    sparse_weight = metrics.get("sparse_weight", 0.001)
    decoder_epochs = config.get("decoder_epochs", 10)
    gpu = env.get("gpu") or "GPU"
    ddpm_seconds = ddpm["latency_ms_batch64"] / 1000
    epoch_seconds = {
        key: sum(row["seconds"] for row in history[key]) / len(history[key])
        for _, key in MODEL_NAMES if history.get(key)
    }
    fid_order = sorted(
        [(label, models[key]["fid"]) for label, key in MODEL_NAMES if "fid" in models[key]],
        key=lambda item: item[1],
    )
    fid_text = ", ".join(f"{label} ({idn(value)})" for label, value in fid_order)
    mse_drop = (ae["test_mse"] - sae["test_mse"]) / ae["test_mse"] * 100
    frozen_ratio = frozen["test_mse"] / ae["test_mse"]
    ddpm_vs_dcgan = ddpm["latency_ms_batch64"] / dcgan["latency_ms_batch64"]
    ddpm_per_image = ddpm["latency_ms_batch64"] / 64

    def first_last(key, field):
        rows = history[key]
        return rows[0][field], rows[-1][field]

    ae_first, ae_last = first_last("vanilla_ae", "reconstruction")
    sae_first, sae_last = first_last("sparse_ae", "reconstruction")
    vae_first, vae_last = first_last("vae", "loss")
    dd_first, dd_last = first_last("ddpm", "loss")
    gg_first, gg_last = first_last("gan", "generator")
    gd_first, gd_last = first_last("gan", "discriminator")
    dg_first, dg_last = first_last("dcgan", "generator")
    dd2_first, dd2_last = first_last("dcgan", "discriminator")
    g_ideal, d_ideal = math.log(2), 2 * math.log(2)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "mathtext.fontset": "dejavusans"})

    with PdfPages(output_path) as pdf:
        # ===================================================== Halaman 1
        fig = new_page("Studi Komparatif Model Generatif", 1,
                       "Autoencoder, Sparse AE, VAE, GAN, DCGAN, dan DDPM pada Fashion-MNIST")
        y = part_heading(fig, "Bagian I. Identitas Penulis dan Tautan Repositori", 0.865)
        y = draw_table(fig, y, [0.22, 0.78], ["Item", "Keterangan"], [
            ["Nama lengkap", name],
            ["NIM", str(nim)],
            ["Kelas", class_name],
            ["Mata kuliah", "Deep Learning (D6421), Sesi 07: Generative Models (GANs & Diffusion) "
                            "terintegrasi Sesi 06 (Autoencoders)"],
            ["Dosen pengampu", "Muhammad Fikri Hasani, S.Kom., M.T. & Tim Dosen Pengampu D6421"],
            ["Repositori GitHub", repository],
            ["Isi repositori", "Skrip modular PyTorch (src/data.py, models.py, diffusion.py, metrics.py, "
                               "train_classifier.py, run_experiment.py), requirements.txt, README berisi "
                               "petunjuk eksekusi, dan git commit history"],
        ], fs=8.0, pad_y=0.005)
        y = heading(fig, "Abstrak", y - 0.01, 11)
        abstract = (
            f"Studi ini membandingkan enam model generatif (Vanilla AE, Sparse AE, VAE, Vanilla GAN, "
            f"DCGAN, DDPM) pada Fashion-MNIST dengan split 80:10:10 (seed {seed}), batch {batch_size}, "
            f"dan {epochs} epoch, serta menguji secara empiris batasan bobot dan fitur laten MLP "
            f"classifier untuk image generation. Sparse AE memperoleh MSE test {idn(sae['test_mse'], 5)} "
            f"(Vanilla AE {idn(ae['test_mse'], 5)}; VAE {idn(vae['test_mse'], 5)}). Pada FID "
            f"({idn(fid_samples, 0)} sampel) urutan terbaik ke terburuk adalah {fid_text}; kualitas DDPM "
            f"dibayar dengan latensi {idn(ddpm_seconds)} detik per batch 64 pada {gpu}. Decoder yang "
            f"dilatih pada fitur MLP beku mencapai MSE {idn(frozen['test_mse'], 5)}, sekitar "
            f"{idn(frozen_ratio, 1)} kali galat AE berdimensi laten sama ({idn(ae['test_mse'], 5)}), "
            f"sedangkan proyeksi balik transpos bobot dan prior sampling pada ruang laten classifier "
            f"gagal membentuk citra utuh. Hasil dibatasi satu seed, resolusi 28x28, dan FID berbasis "
            f"Inception v3 pra-latih ImageNet."
        )
        y = paragraph(fig, abstract, y, fs=9.0)
        y = heading(fig, "Kata kunci", y, 9.5)
        y = paragraph(fig, "representasi laten; autoencoder; VAE; GAN; DCGAN; diffusion; "
                           "information bottleneck; Fashion-MNIST", y, fs=8.6)
        y = heading(fig, "Peta isi laporan", y, 9.5)
        y = paragraph(fig, (
            "Bagian II: landasan teori, desain arsitektur, dan formulasi matematika. "
            "Bagian III: pembuktian batasan MLP classifier. Bagian IV: tabel komparasi metrik. "
            "Bagian V: lima visualisasi komparatif utama. Bagian VI: pembahasan kritis, jawaban "
            "pertanyaan penelitian Q1-Q6, verifikasi Generative Trilemma, dan rekomendasi industri."),
            y, fs=8.6)
        check(y, "halaman 1")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 2
        fig = new_page("Bagian II. Landasan Teori dan Desain", 2,
                       "Landasan teori, protokol standar, dan tabel konfigurasi arsitektur")
        y = 0.872
        y = heading(fig, "II.1 Landasan teori", y)
        y = paragraph(fig, (
            "Autoencoder (AE) mempelajari pasangan encoder-decoder deterministik yang meminimalkan galat "
            "rekonstruksi melalui bottleneck; Sparse AE menambah penalti L1 pada aktivasi tersembunyi agar "
            "kode lebih jarang. VAE memodelkan z sebagai distribusi q(z|x) yang diregularisasi KL terhadap "
            "prior N(0,I), sehingga dapat disampling [2]. GAN melatih generator dan discriminator dalam "
            "permainan minimax yang pada discriminator optimal terkait dengan divergensi Jensen-Shannon [1]; "
            "DCGAN menerapkan prinsip arsitektur konvolusional agar permainan itu lebih stabil [3]. DDPM "
            "membalik proses difusi derau dengan jaringan yang memprediksi derau pada tiap langkah waktu "
            "melalui regresi MSE biasa [4]. MLP classifier adalah model diskriminatif yang memodelkan "
            "P(y|x), bukan P(x), sehingga menjadi pembanding untuk menunjukkan batas fitur diskriminatif."),
            y, fs=8.4)
        y = heading(fig, "II.2 Protokol eksperimen terstandar", y)
        y = paragraph(fig, (
            f"Dataset: Fashion-MNIST (28x28 grayscale, 10 kelas, 70.000 citra). Partisi resmi train dan test "
            f"digabung lalu dibagi menjadi 56.000 train, 7.000 validation, dan 7.000 test dengan "
            f"torch.Generator seed {seed}; torch.manual_seed({seed}) dan np.random.seed({seed}) ditetapkan "
            f"pada semua model. Preprocessing ToTensor tanpa augmentasi (AE/VAE memakai piksel [0,1]; GAN dan "
            f"DDPM [-1,1]). DataLoader memakai batch {batch_size}, shuffle=True hanya pada train. Semua "
            f"model generatif dilatih {epochs} epoch; decoder beku dilatih {decoder_epochs} epoch; DDPM "
            f"memakai {ddpm_steps} langkah. Perangkat: {gpu}, Python {env.get('python', '-')}, PyTorch "
            f"{env.get('torch', '-')}. Metrik: MSE rata-rata per piksel; PSNR = 10 log10(1/MSE); SSIM "
            f"jendela 7x7; FID dari {idn(fid_samples, 0)} sampel real dan sintetis memakai Inception v3 "
            f"ImageNet (grayscale diulang ke RGB, resize 299x299) [7]; latensi = waktu satu batch 64 "
            f"dengan warm-up dan sinkronisasi CUDA."),
            y, fs=8.4)
        y = heading(fig, "II.3 Tabel konfigurasi arsitektur dan hiperparameter", y)
        rows = [
            ["Vanilla AE", "Enc 784-256-64; Dec 64-256-784", "64", "ReLU; Sigmoid (output)",
             "Adam, 1e-3", "435.536"],
            ["Sparse AE", "Sama dengan Vanilla AE; penalti L1 pada h (256 unit)", "64 (h: 256)",
             "ReLU; Sigmoid (output)", f"Adam, 1e-3; lambda={sparse_weight:g}", "435.536"],
            ["VAE", "Enc 784-256, lalu mu dan log var (64); Dec 64-256-784", "64",
             "ReLU; Sigmoid (output)", "Adam, 1e-3; beta=1", "451.984"],
            ["Vanilla GAN", "G: 100-256-512(BN)-784; D: 784-512-256-1 (Dropout 0,3)", "100 (noise)",
             "G: LeakyReLU 0,2, Tanh; D: LeakyReLU 0,2", "Adam, 2e-4, beta1=0,5",
             "G 560.656; D 533.505"],
            ["DCGAN", "G: ConvT 100-128 (7x7)-64 (14x14)-1 (28x28), BN; D: Conv 1-64-128 (BN)-1",
             "100 (noise)", "G: ReLU, Tanh; D: LeakyReLU 0,2", "Adam, 2e-4, beta1=0,5",
             "G 759.680; D 138.689"],
            ["DDPM", f"U-Net residual, GroupNorm(8), embedding waktu sinusoidal; T={ddpm_steps}, "
                     f"beta 1e-4 ke 0,02", "28x28 (citra)", "SiLU", "Adam, 2e-4", "1.045.377"],
            ["MLP Classifier", "784-256-64-10", "64", "ReLU", "Adam, 1e-3, 20 epoch", "218.058"],
        ]
        y = draw_table(fig, y, [0.115, 0.31, 0.10, 0.19, 0.16, 0.125],
                       ["Model", "Lapisan arsitektur", "Dim laten", "Aktivasi", "Optimizer, LR", "Parameter"],
                       rows, fs=6.8)
        y = paragraph(fig, (
            "Catatan: jumlah parameter dihitung dari definisi lapisan pada src/models.py dan "
            "src/diffusion.py. Penalti Sparse AE memakai mean(|h|), setara lambda * ||h||_1 / 256 "
            "yang dirata-ratakan per batch."), y, fs=7.6)
        check(y, "halaman 2")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 3
        fig = new_page("Bagian II (lanjutan). Diagram dan Formulasi Loss", 3,
                       "Diagram skematik tujuh jaringan dan formulasi loss lengkap")
        y = 0.872
        y = heading(fig, "II.4 Diagram skematik arsitektur", y)
        draw_architecture(fig, y, 0.405)
        y -= 0.412
        y = paragraph(fig, "Warna: biru = input/output, hijau = lapisan linear, kuning = representasi laten, "
                           "merah muda = lapisan dengan penalti sparsity, oranye = lapisan konvolusional.",
                      y, fs=7.4)
        y = heading(fig, "II.5 Formulasi matematika loss", y)
        y = eq(fig, "Vanilla AE",
               r"$\mathcal{L}_{AE}=\frac{1}{ND}\sum_{i=1}^{N}\Vert x_i-\hat{x}_i\Vert_2^2,\;\; "
               r"\hat{x}=g_\phi(f_\theta(x)),\;\; z=f_\theta(x)\in\mathbb{R}^{64}$", y, gap=0.027)
        y = eq(fig, "Sparse AE",
               r"$\mathcal{L}_{SAE}=\mathcal{L}_{AE}+\lambda\,\mathrm{mean}(|h|),\;\; "
               r"h=\mathrm{ReLU}(W_1x+b_1)\in\mathbb{R}^{256},\;\; \lambda=10^{-3}$", y, gap=0.027)
        y = eq(fig, "VAE",
               r"$\mathcal{L}_{VAE}=-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]+\beta\,"
               r"D_{KL}(q_\phi(z|x)\,\Vert\,\mathcal{N}(0,I))$", y, gap=0.026)
        y = eq(fig, "",
               r"$z=\mu+\sigma\odot\epsilon,\;\epsilon\sim\mathcal{N}(0,I);\;\; "
               r"D_{KL}=-\frac{1}{2}\sum_{j=1}^{64}(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$", y, gap=0.034)
        y = eq(fig, "Vanilla GAN",
               r"$\min_G\max_D\;\mathbb{E}_{x\sim p_{data}}[\log D(x)]+"
               r"\mathbb{E}_{z\sim\mathcal{N}(0,I)}[\log(1-D(G(z)))]$", y, gap=0.027)
        y = eq(fig, "DCGAN / GAN",
               r"$\mathcal{L}_D=-\mathbb{E}[\log D(x)]-\mathbb{E}[\log(1-D(G(z)))],\;\; "
               r"\mathcal{L}_G=-\mathbb{E}[\log D(G(z))]$  (non-saturating)", y, gap=0.034)
        y = eq(fig, "DDPM",
               r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\;\; "
               r"\bar\alpha_t=\prod_{s\leq t}(1-\beta_s),\;\; "
               r"\mathcal{L}_{DDPM}=\mathbb{E}_{t,x_0,\epsilon}\Vert\epsilon-\epsilon_\theta(x_t,t)\Vert_2^2$",
               y, gap=0.037)
        y = eq(fig, "",
               r"$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}"
               r"\epsilon_\theta(x_t,t)\right)+\sigma_t z,\;\; z\sim\mathcal{N}(0,I)$", y, gap=0.044)
        y = eq(fig, "MLP Classifier",
               r"$\mathcal{L}_{CE}=-\frac{1}{N}\sum_{i}\log\mathrm{softmax}(W_3z_i+b_3)_{y_i},\;\; "
               r"z=\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2)$", y, gap=0.036)
        y = paragraph(fig, (
            "Kelima keluarga generatif memakai data, split, dan seed yang sama; perbedaan hasil karenanya "
            "mencerminkan inductive bias dan objective masing-masing, bukan perbedaan protokol."),
            y, fs=8.0)
        check(y, "halaman 3")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 4
        fig = new_page("Bagian III. Batasan MLP Classifier untuk Image Generation", 4,
                       "Pembuktian teoretis dan hasil numerik Tahap 1-5")
        y = 0.872
        y = heading(fig, "III.1 Pemodelan P(y|x) berbanding P(x)", y)
        y = paragraph(fig, (
            "Classifier mengoptimalkan likelihood kondisional, sedangkan model generatif perlu mempelajari "
            "distribusi data itu sendiri:"), y, fs=8.3, gap=0.004)
        y = eq(fig, "", r"$\theta^{*}_{clf}=\arg\max_\theta\,\mathbb{E}_{(x,y)}[\log p_\theta(y|x)]"
                        r"\qquad\mathrm{vs}\qquad \theta^{*}_{gen}=\arg\max_\theta\,\mathbb{E}_{x}[\log p_\theta(x)]$",
               y, label_w=0.0, gap=0.030)
        y = paragraph(fig, (
            "Pada kerangka information bottleneck [6], representasi z = f(x) yang optimal bagi klasifikasi "
            "memaksimalkan I(z;y) - beta I(z;x): ia hanya perlu mempertahankan informasi yang membedakan kelas. "
            "Tidak ada suku dalam cross-entropy yang menghargai pelestarian variasi intra-kelas (bentuk "
            "siluet, tekstur, posisi), sehingga informasi itu boleh hilang tanpa menaikkan loss."),
            y, fs=8.3)
        y = heading(fig, "III.2 Bobot maju tidak dapat dibalik: W^T tidak sama dengan W^-1", y)
        y = paragraph(fig, "Proyeksi balik top-down memakai transpos bobot (Tahap 2) dan latent (Tahap 3):",
                      y, fs=8.3, gap=0.004)
        y = eq(fig, "", r"$\hat{z}=\mathrm{ReLU}(W_3^{T}y_c),\;\; \hat{h}_1=\mathrm{ReLU}(W_2^{T}\hat{z}),\;\; "
                        r"\hat{x}=\sigma(W_1^{T}\hat{h}_1)$", y, label_w=0.0, gap=0.030)
        y = paragraph(fig, (
            f"Tiga alasan transpos bukan invers. (a) W1 berukuran 256x784 sehingga rank(W1) <= 256 dan "
            f"W1^T W1 (784x784) berpangkat paling banyak 256, jadi tidak mungkin sama dengan matriks "
            f"identitas; setidaknya {784 - weights['input_to_hidden']['rank']} arah masukan berada di null "
            f"space W1 dan tidak berpengaruh pada keluaran, sehingga tak ada pemetaan yang dapat "
            f"memulihkannya. (b) Bahkan untuk matriks persegi, W^T W = I hanya berlaku bila W ortogonal, "
            f"dan pelatihan cross-entropy tidak memaksakan ortogonalitas; diagnostik ||W^T W - I||_F / "
            f"sqrt(n) pada tabel di bawah jauh dari nol. (c) ReLU(u) = max(0,u) tidak injektif: semua u <= 0 "
            f"dipetakan ke 0, sehingga informasi nilai negatif hilang secara ireversibel. Proyeksi dari y_c "
            f"ke baris W3 juga hanya "
            "menghasilkan superposisi template bobot kasar."), y, fs=8.3)
        y = heading(fig, "III.3 Ruang laten tanpa regularisasi kerapatan P(z)", y)
        y = paragraph(fig, "Tahap 5 menyampling z ~ N(mu_z, diag(sigma_z^2)) dari statistik latent test set:",
                      y, fs=8.3, gap=0.004)
        y = eq(fig, "", r"$z_{rand}\sim\mathcal{N}(\mu_z,\mathrm{diag}(\sigma_z^{2})),\qquad "
                        r"z_{MLP}=\mathrm{ReLU}(\cdot)\geq 0$", y, label_w=0.0, gap=0.030)
        y = paragraph(fig, (
            "Asumsi Gaussian independen ini tidak dijamin sesuai: z_MLP non-negatif, kemungkinan berkorelasi, "
            "dan tidak ada suku KL atau diskriminator yang menautkan sebaran z ke prior yang diketahui, sehingga "
            "sampel acak tidak harus jatuh pada manifold yang didekode menjadi objek utuh."), y, fs=8.3)
        y = heading(fig, "III.4 Prosedur Tahap 1-5 dan hasil numerik", y)
        y = draw_table(fig, y, [0.07, 0.50, 0.43], ["Tahap", "Operasi", "Hasil"], [
            ["1", "Latih MLP 784-256-64-10 dengan cross-entropy; simpan W1, W2, W3 dan z_MLP (64)",
             f"Akurasi test {pct(accuracy)} (target minimal 88%)"],
            ["2", "Proyeksi balik one-hot kelas y_c (kelas prediksi citra uji pertama) lewat transpos bobot tanpa training",
             "Figur 3 panel 2: tekstur abu-abu berbercak tanpa kontur pakaian"],
            ["3", "Proyeksi balik z_MLP citra uji lewat W2^T lalu W1^T",
             "Figur 3 panel 3: pola blok hampir biner, tidak mirip citra asli"],
            ["4", f"Bekukan classifier; latih decoder 64-256-784 (MSE, {decoder_epochs} epoch) pada z_MLP",
             f"MSE {idn(frozen['test_mse'], 5)}, PSNR {idn(frozen['test_psnr_db'])} dB, "
             f"SSIM {idn(frozen['test_ssim'], 4)}"],
            ["5", "Sampling z_rand dari N(mu_z, diag sigma_z^2) lalu didekode",
             "Figur 3 panel 6: bentuk tidak teratur, bukan objek utuh"],
        ], fs=7.0)
        names = {"input_to_hidden": "W1 (input ke hidden)", "hidden_to_latent": "W2 (hidden ke laten)",
                 "latent_to_class": "W3 (laten ke kelas)"}
        weight_rows = []
        for key, label in names.items():
            info = weights[key]
            out_dim, in_dim = info["shape"]
            weight_rows.append([label, f"{out_dim}x{in_dim}", str(info["rank"]), str(in_dim - info["rank"]),
                                idn(info["normalized_gram_error"], 3)])
        y = draw_table(fig, y, [0.30, 0.16, 0.14, 0.17, 0.23],
                       ["Matriks", "Ukuran", "Rank", "Dimensi null space (min)", "||W^T W - I||_F / sqrt(n)"],
                       weight_rows, fs=7.0)
        y = draw_table(fig, y, [0.40, 0.20, 0.20, 0.20], ["Perbandingan laten 64-dim", "MSE", "PSNR (dB)", "SSIM"], [
            ["Decoder pada z_MLP beku", idn(frozen["test_mse"], 5), idn(frozen["test_psnr_db"]),
             idn(frozen["test_ssim"], 4)],
            ["Vanilla AE (z_AE)", idn(ae["test_mse"], 5), idn(ae["test_psnr_db"]), idn(ae["test_ssim"], 4)],
        ], fs=7.0)
        y = paragraph(fig, (
            f"Decoder pada z_MLP masih menangkap siluet kasar, tetapi galatnya {idn(frozen_ratio, 1)} kali "
            f"galat AE dengan laten berdimensi sama. Ini konsisten dengan hilangnya detail spasial intra-kelas "
            f"pada fitur diskriminatif. Catatan keterbatasan: decoder beku dilatih {decoder_epochs} epoch "
            f"sedangkan AE dilatih {epochs} epoch dan encoder AE ikut dioptimalkan untuk rekonstruksi, sehingga "
            f"selisih ini tidak sepenuhnya murni dari fitur; Tahap 2, 3, dan 5 hanya divisualisasikan pada satu "
            f"citra/sampel."), y, fs=7.8)
        check(y, "halaman 4")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 5
        fig = new_page("Bagian IV. Tabel Komparasi Metrik Numerik", 5,
                       f"Evaluasi pada Test Set (7.000 citra); FID dari {idn(fid_samples, 0)} sampel; latensi per batch 64")
        y = 0.872
        latent_label = {"vanilla_ae": "64", "sparse_ae": "64 (hidden 256)", "vae": "64",
                        "gan": "100 (noise)", "dcgan": "100 (noise)", "ddpm": "28x28 (citra)"}
        stability = {
            "vanilla_ae": f"Konvergen; MSE train {idn(ae_first, 4)} ke {idn(ae_last, 4)}",
            "sparse_ae": f"Konvergen; MSE train {idn(sae_first, 4)} ke {idn(sae_last, 4)}",
            "vae": f"Konvergen; loss {idn(vae_first, 1)} ke {idn(vae_last, 1)}",
            "gan": (f"Tidak mencapai titik setimbang; rata-rata per epoch: G {idn(gg_first)} ke "
                    f"{idn(gg_last)}, D {idn(gd_first)} ke {idn(gd_last)}; mode collapse tidak diukur"),
            "dcgan": (f"Tidak mencapai titik setimbang; rata-rata per epoch: G {idn(dg_first)} ke "
                      f"{idn(dg_last)}, D {idn(dd2_first)} ke {idn(dd2_last)}; mode collapse tidak diukur"),
            "ddpm": f"Konvergen; loss derau {idn(dd_first, 3)} ke {idn(dd_last, 3)}",
        }
        rows = []
        for label, key in MODEL_NAMES:
            result = models[key]
            rows.append([
                label, latent_label[key], idn(epoch_seconds[key]), idn(result["latency_ms_batch64"]),
                idn(result["test_mse"], 5) if "test_mse" in result else "N/A",
                idn(result["test_psnr_db"]) if "test_psnr_db" in result else "N/A",
                idn(result["test_ssim"], 3) if "test_ssim" in result else "N/A",
                idn(result["fid"]) if "fid" in result else "N/A",
                stability[key],
            ])
        y = draw_table(fig, y, [0.105, 0.09, 0.07, 0.09, 0.075, 0.07, 0.06, 0.06, 0.38],
                       ["Arsitektur", "Dim laten", "Waktu per epoch (s)", "Latensi (ms per batch 64)",
                        "Test MSE", "Test PSNR (dB)", "Test SSIM", "FID", "Kestabilan training"],
                       rows, fs=6.5)
        y = paragraph(fig, (
            f"N/A berarti metrik tidak berlaku (bukan nol): MSE, PSNR, dan SSIM hanya untuk model yang "
            f"merekonstruksi citra uji, dan FID hanya untuk model yang menyintesis dari derau. Latensi AE "
            f"diukur pada lintasan encode-decode, sedangkan VAE, GAN, dan DCGAN pada satu lintasan generator/"
            f"decoder dari prior; DDPM memakai seluruh {ddpm_steps} langkah reverse sampling "
            f"({idn(ddpm['latency_ms_batch64'], 0)} ms, sekitar {idn(ddpm_per_image, 0)} ms per citra "
            f"setelah diamortisasi dalam batch). Seluruh latensi diukur pada {gpu} dan tidak mewakili "
            f"perangkat mobile. Kolom kestabilan diisi dari kurva loss rata-rata per epoch (Figur 5), "
            f"bukan label baku: nilai setimbang ideal adalah G = ln2 ({idn(g_ideal)}) dan D = 2 ln2 "
            f"({idn(d_ideal)}), dan kedua GAN tidak mencapainya dalam {epochs} epoch. Karena kurva "
            f"dirata-ratakan per epoch, osilasi antar-iterasi tidak terlihat. Keterbatasan: satu seed "
            f"(tanpa interval kepercayaan), dan FID memakai {idn(fid_samples, 0)} sampel sedangkan fitur "
            f"Inception berdimensi 2.048, sehingga kovarians sampel berpangkat kurang dari dimensinya dan "
            f"FID cenderung bias; peringkat FID perlu dibaca bersama Figur 2."), y, fs=7.7)
        y = part_heading(fig, "Bagian V. Lima Visualisasi Komparatif Utama", y - 0.004)
        y = heading(fig, "Figur 1. Grid rekonstruksi side-by-side", y)
        y = put_image(fig, assets / "figure_1_reconstructions.png", y, 0.64,
                      "Figur 1. Delapan sampel Test Set. Baris dari atas: citra asli (ground truth), "
                      "Vanilla AE, Sparse AE, VAE.")
        y = paragraph(fig, (
            "Analisis. Vanilla AE mempertahankan siluet pakaian, tetapi detail kecil dan batas tepi tampak "
            "halus. Sparse AE memperlihatkan fidelitas rata-rata sedikit lebih baik sesuai MSE dan SSIM "
            f"pada tabel di atas (MSE turun {idn(mse_drop, 1)}% dibanding Vanilla AE). VAE tampak paling "
            "kabur, sejalan dengan objective rekonstruksi plus regularisasi KL yang menukar ketajaman dengan "
            "keteraturan laten."), y, fs=8.0)
        check(y, "halaman 5")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 6
        fig = new_page("Bagian V (lanjutan). Figur 2 dan Figur 3", 6,
                       "Sampling acak antarmodel dan bukti empiris batasan MLP classifier")
        y = 0.872
        y = heading(fig, "Figur 2. Random generative sampling", y)
        figure2 = Image.open(assets / "figure_2_generated_samples.png").convert("RGB")
        if figure2_is_grid:
            caption2 = ("Figur 2. 16 sampel per model dalam kisi 4x4. Panel A: VAE; Panel B: Vanilla GAN; "
                        "Panel C: DCGAN (semua dari z ~ N(0,I)); Panel D: DDPM (reverse denoising dari "
                        "x_T ~ N(0,I)).")
        else:
            caption2 = ("Figur 2. 16 sampel per model, satu baris per model. Dari atas: VAE, Vanilla GAN, "
                        "DCGAN (dari z ~ N(0,I)), dan DDPM (reverse denoising dari x_T ~ N(0,I)).")
        y = put_image(fig, figure2, y, 0.92, caption2)
        y = paragraph(fig, (
            "Analisis. Sampel VAE cenderung lembut. Vanilla GAN menampilkan sejumlah bentuk yang terputus "
            "dan bertekstur kasar. DCGAN menghasilkan siluet pakaian yang lebih jelas. DDPM membentuk "
            "beberapa objek yang dapat dikenali dengan variasi antarcontoh. Urutan FID "
            f"({fid_text}) sejalan dengan kesan visual ini, tetapi keberagaman mode hanya dinilai secara "
            "visual dari 16 sampel, bukan diukur terpisah."), y, fs=8.2)
        y = heading(fig, "Figur 3. Bukti empiris batasan bobot dan latent features MLP classifier", y)
        row_scale = 0.74
        width_b = row_scale * fig3_row_b.width / fig3_row_a.width
        y = put_image(fig, fig3_row_a, y, row_scale, None)
        y += 0.006
        y = put_image(fig, fig3_row_b, y, width_b,
                      "Figur 3. Baris atas: citra asli, proyeksi balik W^T dari vektor kelas y_c, proyeksi balik "
                      "W^T dari z_MLP, rekonstruksi Frozen Latent Decoder. Baris bawah: rekonstruksi Vanilla AE "
                      "(z_AE), hasil prior sampling pada ruang laten MLP, dan rekonstruksi VAE.",
                      x=LEFT + (CONTENT_WIDTH - row_scale) / 2)
        y = paragraph(fig, (
            "Analisis. Proyeksi y_c menghasilkan tekstur abu-abu berbercak tanpa kontur pakaian, sesuai dugaan "
            "superposisi template bobot kasar. Proyeksi z_MLP menghasilkan pola blok hampir biner yang tidak "
            "menyerupai citra asli, sehingga bobot maju tidak berfungsi sebagai decoder. Frozen Latent Decoder "
            "memulihkan siluet tetapi lebih halus daripada Vanilla AE (galat lebih besar pada Bagian III). "
            "Prior sampling pada ruang laten MLP menghasilkan bentuk tidak teratur dengan garis-garis "
            "horizontal, bukan objek utuh."), y, fs=8.2)
        check(y, "halaman 6")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 7
        fig = new_page("Bagian V (lanjutan). Figur 4", 7,
                       "Interpolasi linier pada ruang laten lima model")
        y = 0.872
        y = heading(fig, "Figur 4. Linear latent space interpolation", y)
        y = put_image(fig, assets / "figure_4_interpolation.png", y, 0.80,
                      "Figur 4. Interpolasi linier z(alpha) = (1-alpha) z_A + alpha z_B dengan alpha = 0, 0,2, "
                      "0,4, 0,6, 0,8, 1. Baris dari atas: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN. "
                      "Endpoint AE/Sparse AE/VAE adalah encoding dua citra test berkelas berbeda; endpoint GAN "
                      "dan DCGAN adalah dua vektor noise acak.")
        y = paragraph(fig, (
            "Analisis. Pada Vanilla AE dan Sparse AE perubahan antar-alpha relatif bertahap, tetapi objek di "
            "antara endpoint dapat kehilangan ciri kelas yang tegas. VAE juga berubah lembut. Pada Vanilla GAN "
            "dan DCGAN endpoint adalah dua vektor noise acak karena generator tidak memiliki encoder untuk "
            "memetakan citra endpoint ke ruang laten, sehingga kedua GAN tidak dapat dibandingkan langsung "
            "dengan baris AE pada pasangan citra yang sama."), y, fs=8.4)
        y = paragraph(fig, (
            "Figur ini hanya memuat satu pasangan sampel, sehingga mendukung tetapi tidak membuktikan "
            "bahwa seluruh ruang laten bebas dari void atau artefak. Mekanisme penyebab artefak pada AE "
            "deterministik dibahas pada jawaban Q4 (Bagian VI)."), y, fs=8.4)
        check(y, "halaman 7")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 8
        fig = new_page("Bagian V (lanjutan). Figur 5", 8,
                       "Kurva training loss keenam model dan lintasan reverse diffusion DDPM")
        y = 0.872
        y = heading(fig, "Figur 5. Training loss curves dan DDPM denoising trajectory", y)
        y = put_image(fig, training_figure, y, 0.92,
                      "Figur 5a. Kurva loss tiap objective selama training. Garis putus-putus abu-abu pada "
                      "panel GAN menandai nilai setimbang ideal (G = ln2, D = 2 ln2).")
        y = put_image(fig, trajectory_image, y, 0.86,
                      "Figur 5b. Sekuens reverse diffusion bertahap dari derau acak x_T menuju citra x_0 "
                      "(langkah t = 999, 500, 250, 0).")
        y = paragraph(fig, (
            f"Analisis. Loss rekonstruksi Vanilla AE turun dari {idn(ae_first, 4)} ke {idn(ae_last, 4)} dan "
            f"Sparse AE dari {idn(sae_first, 4)} ke {idn(sae_last, 4)}; loss VAE (BCE + KL, skala berbeda dari "
            f"MSE) turun dari {idn(vae_first, 1)} ke {idn(vae_last, 1)}; loss derau DDPM turun dari "
            f"{idn(dd_first, 3)} ke {idn(dd_last, 3)}. Keempatnya turun mulus. Pada Vanilla GAN, loss G "
            f"{trend_word(gg_first, gg_last)} dari {idn(gg_first)} ke {idn(gg_last)} dan loss D "
            f"{trend_word(gd_first, gd_last)} dari {idn(gd_first)} ke {idn(gd_last)}; pada DCGAN, loss G "
            f"{trend_word(dg_first, dg_last)} dari {idn(dg_first)} ke {idn(dg_last)} dan loss D "
            f"{trend_word(dd2_first, dd2_last)} dari {idn(dd2_first)} ke {idn(dd2_last)}. Kurva GAN bergerak "
            f"berlawanan arah dan tidak menuju nilai setimbang, berbeda dengan kurva DDPM yang turun "
            f"monoton. Lintasan reverse diffusion memperlihatkan derau acak yang berangsur membentuk siluet "
            f"lalu detail pakaian."), y, fs=8.2)
        check(y, "halaman 8")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 9
        fig = new_page("Bagian VI. Pembahasan Kritis dan Jawaban Q1-Q3", 9,
                       "Jawaban pertanyaan penelitian berdasarkan hasil empiris Bagian III-V")
        y = 0.872
        body_fs, para_gap = 8.4, 0.012
        y = heading(fig, "Q1. Reconstruction fidelity versus generative quality", y)
        y = paragraph(fig, (
            f"Rekonstruksi pada Test Set: Vanilla AE memperoleh MSE {idn(ae['test_mse'], 5)}, PSNR "
            f"{idn(ae['test_psnr_db'])} dB, SSIM {idn(ae['test_ssim'], 3)}; Sparse AE {idn(sae['test_mse'], 5)}, "
            f"{idn(sae['test_psnr_db'])} dB, {idn(sae['test_ssim'], 3)}; VAE {idn(vae['test_mse'], 5)}, "
            f"{idn(vae['test_psnr_db'])} dB, {idn(vae['test_ssim'], 3)}. Sparse AE lebih baik daripada Vanilla "
            f"AE pada ketiga metrik (MSE turun {idn(mse_drop, 1)}%), sedangkan VAE paling rendah fidelitasnya "
            f"({idn(vae['test_mse'] / ae['test_mse'], 1)} kali MSE Vanilla AE) karena objective-nya menukar "
            f"fidelitas dengan ruang laten yang teratur."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"Pengaruh penalti sparsity: secara teori lambda * |h| membatasi kapasitas efektif lapisan "
            f"tersembunyi sehingga cenderung menaikkan galat dibanding representasi padat Vanilla AE. Hasil di "
            f"sini berlawanan, yang dapat dijelaskan bila sparsity ringan bertindak sebagai regularizer. Karena "
            f"hanya satu seed dan satu nilai lambda ({sparse_weight:g}), selisih ini belum dapat dinyatakan "
            f"signifikan. Pada FID dari prior z ~ N(0,I) (DDPM dari x_T ~ N(0,I)), urutan terbaik ke terburuk: "
            f"{fid_text}."), y, fs=body_fs, gap=para_gap)
        y = heading(fig, "Q2. Blurriness versus sharp edges dan struktur fitur", y)
        y = paragraph(fig, (
            "Sebab blur. Minimizer MSE untuk decoder adalah ekspektasi kondisional: x-hat*(z) = E[x|z]. Bila "
            "beberapa citra yang masuk akal berbagi kode z yang sama atau berdekatan, rata-ratanya melumatkan "
            "tepi yang posisinya bervariasi; BCE per piksel juga berminimizer rata-rata (probabilitas) "
            "kondisional. VAE menambah sumber blur karena z disampel dengan derau dari q(z|x) dan suku KL "
            "menekan informasi di z, sehingga decoder memetakan wilayah z yang tumpang tindih ke rata-rata "
            "citra-citranya."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            "Sebab ketajaman GAN. Discriminator memberi sinyal nyata-atau-palsu, bukan jarak piksel. Pada "
            "discriminator optimal D*(x) = p_data / (p_data + p_g), loss generator minimax setara "
            "2 D_JS(p_data || p_g) - log 4, sehingga generator didorong menaruh massa pada manifold data; "
            "satu sampel adalah satu titik tajam, bukan rata-rata. Konsekuensinya sampel tajam tetapi tanpa "
            "jaminan cakupan semua mode. Implementasi di sini memakai generator non-saturating, bukan "
            "menghitung D_JS secara langsung."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            "Struktur fitur Sparse AE. Penalti L1 membuat sebagian besar unit tersembunyi bernilai nol bagi "
            "satu input, sehingga tiap citra dijelaskan oleh himpunan kecil unit yang spesifik dan bobot "
            "cenderung menjadi detektor bagian atau goresan lokal; Vanilla AE memakai kode padat tempat banyak "
            "unit aktif bersamaan dan fitur bercampur. Catatan: struktur fitur per-unit (proporsi unit nol, "
            "histogram aktivasi, visualisasi bobot) tidak diukur dalam eksperimen ini; uraian ini teoretis."),
            y, fs=body_fs, gap=para_gap)
        y = heading(fig, "Q3. Stabilitas training, dinamika arsitektur, dan Generative Trilemma", y)
        y = paragraph(fig, (
            "Vanishing gradient dan mode collapse. Loss generator asli log(1 - D(G(z))) menjenuh bila D yakin "
            "sampel palsu (D(G(z)) mendekati 0), sehingga gradien ke G mengecil; mode collapse muncul ketika G "
            "memetakan banyak z ke sedikit keluaran yang paling mengecoh D. Kedua GAN di sini memakai loss G "
            "non-saturating, -log D(G(z)), yang memitigasi saturasi."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            "Prinsip DCGAN [3] yang diterapkan: konvolusi strided/transposed menggantikan pooling dan lapisan "
            "fully-connected tersembunyi, BatchNorm pada G dan D, ReLU dengan Tanh keluaran pada G, serta "
            "LeakyReLU pada D. Karena Vanilla GAN di sini sudah memakai BatchNorm pada generatornya, selisih "
            "Vanilla GAN dan DCGAN terutama berasal dari bias konvolusional (weight sharing dan struktur "
            "lokal), bukan dari normalisasi saja."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"Dinamika kurva (Figur 5). Vanilla GAN: G {trend_word(gg_first, gg_last)} {idn(gg_first)} ke "
            f"{idn(gg_last)}, D {trend_word(gd_first, gd_last)} {idn(gd_first)} ke {idn(gd_last)}; DCGAN: G "
            f"{trend_word(dg_first, dg_last)} {idn(dg_first)} ke {idn(dg_last)}, D "
            f"{trend_word(dd2_first, dd2_last)} {idn(dd2_first)} ke {idn(dd2_last)}. Keduanya jauh dari "
            f"setimbang; karena dirata-ratakan per epoch, osilasi dan mode collapse tidak dapat disimpulkan "
            f"dari kurva. Permainan minimax bersifat non-stasioner (target tiap jaringan bergerak mengikuti "
            f"lawan), sedangkan DDPM meregresi derau dengan target dan jadwal derau tetap tanpa lawan, "
            f"sehingga kurvanya turun mulus ({idn(dd_first, 3)} ke {idn(dd_last, 3)})."),
            y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"Trade-off kecepatan. Generator satu lintasan (VAE {idn(vae['latency_ms_batch64'])} ms, Vanilla "
            f"GAN {idn(gan['latency_ms_batch64'])} ms, DCGAN {idn(dcgan['latency_ms_batch64'])} ms per batch "
            f"64) berbanding DDPM {idn(ddpm['latency_ms_batch64'], 0)} ms, sekitar {idn(ddpm_vs_dcgan, 0)} "
            f"kali DCGAN."), y, fs=body_fs, gap=para_gap)
        check(y, "halaman 9")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 10
        fig = new_page("Bagian VI (lanjutan). Jawaban Q4-Q6", 10,
                       "Latent space, batasan classifier, dan rekomendasi industri")
        y = 0.872
        body_fs, para_gap = 8.3, 0.011
        y = heading(fig, "Q4. Struktur latent space dan interpolasi", y)
        y = paragraph(fig, (
            "Autoencoder deterministik. Encoder memetakan tiap citra latih ke satu titik dan decoder hanya "
            "dilatih pada titik-titik itu; tidak ada suku loss yang membentuk kerapatan di antara atau di "
            "luar titik-titik tersebut. Garis lurus z(alpha) antara z_A dan z_B dapat melewati wilayah kosong "
            "(latent voids) yang tidak pernah dilatih, sehingga decoder mengekstrapolasi tanpa pengawasan "
            "dan menghasilkan citra berhantu (superposisi dua objek) atau terdistorsi. Pada Sparse AE kode "
            "non-negatif dan jarang, sehingga interpolasi linier juga dapat meninggalkan dukungan (support) "
            "kode yang pernah dilihat."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            "VAE dan GAN. Pada VAE, noise reparameterization dan suku KL membuat posterior q(z|x) tiap citra "
            "melebar dan tumpang tindih serta mendekati N(0,I), sehingga wilayah di antara dua kelas tercakup "
            "dan transisi lebih kontinu, meski tidak dijamin semantik di semua pasangan. Pada GAN, generator "
            "dilatih pada seluruh massa prior; namun interpolasi lurus antara dua vektor Gaussian berdimensi "
            "tinggi melewati titik tengah bernorma lebih kecil (sekitar 1/akar-2 pada alpha = 0,5) yang "
            "berada di wilayah berkerapatan rendah. Figur 4 hanya satu pasangan sampel sehingga mendukung, "
            "bukan membuktikan, mekanisme tersebut."), y, fs=body_fs, gap=para_gap)
        y = heading(fig, "Q5. Mengapa bobot dan fitur laten MLP classifier tidak dapat men-generate citra", y)
        y = paragraph(fig, (
            f"Pemodelan P(y|x). Classifier memaksimalkan log p(y|x) sehingga representasi hanya perlu "
            f"membedakan kelas (information bottleneck): variasi intra-kelas dibuang tanpa penalti, jadi z_MLP "
            f"tidak memuat cukup informasi untuk x. Bukti empiris: decoder yang dilatih pada z_MLP beku "
            f"mencapai MSE {idn(frozen['test_mse'], 5)}, PSNR {idn(frozen['test_psnr_db'])} dB, SSIM "
            f"{idn(frozen['test_ssim'], 4)}, lebih buruk dibanding Vanilla AE ({idn(ae['test_mse'], 5)}; "
            f"{idn(ae['test_psnr_db'])} dB; {idn(ae['test_ssim'], 4)}); catatan ketidaksetaraan epoch pada "
            f"Bagian III tetap berlaku."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"Matriks bobot. W1 (256x784) memiliki null space minimal {784 - weights['input_to_hidden']['rank']} "
            f"dimensi, W^T W tidak sama dengan I (||W^T W - I||_F / sqrt(n) = "
            f"{idn(weights['input_to_hidden']['normalized_gram_error'], 3)}; "
            f"{idn(weights['hidden_to_latent']['normalized_gram_error'], 3)}; "
            f"{idn(weights['latent_to_class']['normalized_gram_error'], 3)} untuk W1, W2, W3), dan ReLU "
            f"membuang nilai negatif; karena itu proyeksi transpos hanya menghasilkan bercak dan pola blok "
            f"(Figur 3 panel 2 dan 3)."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            "Ketiadaan regularisasi P(z). Tidak ada KL atau discriminator yang menyatakan sebaran z, sehingga "
            "sampling Gaussian dari statistik latent tidak jatuh pada manifold data dan menghasilkan bentuk "
            "tidak teratur (Figur 3 panel 6)."), y, fs=body_fs, gap=para_gap)
        y = heading(fig, "Q6. Sintesis rekayasa dan rekomendasi industri", y)
        y = paragraph(fig, (
            f"(1) Deteksi anomali tanpa supervisi: skor anomali adalah galat rekonstruksi, jadi model ideal "
            f"merekonstruksi data normal dengan baik tetapi gagal pada anomali. Vanilla AE berkapasitas besar "
            f"cenderung ikut merekonstruksi pola baru (kebocoran identitas), sedangkan penalti sparsity "
            f"membatasi kapasitas efektif. Karena Sparse AE memiliki galat normal lebih rendah (MSE "
            f"{idn(sae['test_mse'], 5)} berbanding {idn(ae['test_mse'], 5)}) dan SSIM lebih tinggi, ia "
            f"direkomendasikan sebagai kandidat utama; namun eksperimen ini tidak memuat citra anomali, "
            f"sehingga sensitivitas dan threshold harus divalidasi pada data inspeksi manufaktur berlabel."),
            y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"(2) Filter avatar waktu nyata (<30 ms): pada {gpu}, VAE {idn(vae['latency_ms_batch64'])} ms, "
            f"Vanilla GAN {idn(gan['latency_ms_batch64'])} ms, dan DCGAN {idn(dcgan['latency_ms_batch64'])} ms "
            f"per batch 64 semuanya di bawah anggaran. Dengan FID {idn(dcgan['fid'])} (DCGAN) berbanding "
            f"{idn(vae['fid'])} (VAE) dan {idn(gan['fid'])} (Vanilla GAN), DCGAN dipilih: generatornya hanya "
            f"sekitar 0,76 juta parameter, konvolusinya efisien, dan bias konvolusional menghasilkan tekstur "
            f"lebih tajam; VAE adalah cadangan bila dibutuhkan encoder dan interpolasi yang mulus. Klaim untuk "
            f"ponsel menunggu pengukuran pada perangkat mobile."), y, fs=body_fs, gap=para_gap)
        y = paragraph(fig, (
            f"(3) Aset visual beresolusi dan berfidelitas tinggi: DDPM memiliki FID terendah "
            f"({idn(ddpm['fid'])}) berbanding DCGAN ({idn(dcgan['fid'])}), pelatihan regresi yang stabil, dan "
            f"objective berbasis likelihood yang secara teori lebih baik dalam cakupan mode; harganya "
            f"{idn(ddpm_per_image, 0)} ms per citra (diamortisasi dalam batch 64) pada 28x28. DDPM "
            f"direkomendasikan untuk produksi aset offline, sedangkan DCGAN untuk iterasi cepat; untuk "
            f"resolusi tinggi diperlukan percepatan sampling dan pengujian di luar benchmark 28x28 ini."),
            y, fs=body_fs, gap=para_gap)
        check(y, "halaman 10")
        pdf.savefig(fig)
        plt.close(fig)

        # ===================================================== Halaman 11
        fig = new_page("Bagian VI (lanjutan). Verifikasi Trilemma dan Kesimpulan", 11,
                       "Generative Trilemma [8], keterbatasan, dan referensi")
        y = 0.872
        y = heading(fig, "Verifikasi Generative Trilemma", y)
        y = draw_table(fig, y, [0.16, 0.22, 0.24, 0.38],
                       ["Model", "Kualitas (FID, lebih rendah lebih baik)", "Kecepatan (ms/batch 64)",
                        "Cakupan mode"], [
            ["VAE", idn(vae["fid"]), idn(vae["latency_ms_batch64"]),
             "Secara teori baik (objective berbasis likelihood); tidak diukur"],
            ["Vanilla GAN", idn(gan["fid"]), idn(gan["latency_ms_batch64"]),
             "Berisiko mode dropping (objective adversarial); tidak diukur"],
            ["DCGAN", idn(dcgan["fid"]), idn(dcgan["latency_ms_batch64"]),
             "Berisiko mode dropping; tidak diukur"],
            ["DDPM", idn(ddpm["fid"]), idn(ddpm["latency_ms_batch64"], 0),
             "Secara teori baik (regresi derau, likelihood bound); tidak diukur"],
        ], fs=7.2)
        y = paragraph(fig, (
            f"Dua sumbu trilemma terverifikasi secara empiris: kualitas dan kecepatan. DDPM mencapai FID terbaik "
            f"({idn(ddpm['fid'])}) tetapi paling lambat, sekitar {idn(ddpm_vs_dcgan, 0)} kali DCGAN; Vanilla GAN "
            f"dan VAE tercepat tetapi FID jauh lebih tinggi; DCGAN berada di titik tengah kualitas-kecepatan. "
            f"Sumbu ketiga, cakupan mode, tidak diukur dalam eksperimen ini (misalnya lewat precision-recall "
            f"atau klasifikasi kelas sampel), sehingga hanya dapat dibahas secara teoretis; jadi trilemma "
            f"terbukti sebagian: pola 'kualitas tinggi mengorbankan kecepatan' terlihat, pola cakupan mode belum "
            f"teruji."), y, fs=8.3)
        y = heading(fig, "Kesimpulan dan batasan", y)
        y = paragraph(fig, (
            f"Sparse AE paling baik untuk rekonstruksi di antara dua AE (MSE {idn(sae['test_mse'], 5)}); DDPM "
            f"unggul dalam FID dengan biaya sampling iteratif; DCGAN memberi kompromi kualitas-kecepatan "
            f"terbaik. MLP classifier mencapai akurasi {pct(accuracy)} namun fitur dan bobotnya tidak dapat "
            f"dipakai merekonstruksi atau menyintesis citra: terbukti lewat proyeksi transpos yang gagal, "
            f"decoder beku yang galatnya {idn(frozen_ratio, 1)} kali AE, dan prior sampling yang tidak "
            f"membentuk objek utuh. Batasan: satu seed tanpa interval kepercayaan; Fashion-MNIST 28x28; FID "
            f"{idn(fid_samples, 0)} sampel dengan extractor ImageNet; satu nilai lambda; Tahap 2, 3, 5 pada "
            f"satu citra; tidak ada uji anomali, pengukuran cakupan mode, maupun perangkat mobile; latensi "
            f"berlaku untuk {gpu} dan bukan jaminan deployment."), y, fs=8.3)
        y = heading(fig, "Referensi", y, 10)
        references = (
            "[1] Goodfellow, I. et al. (2014). Generative Adversarial Nets. NeurIPS.\n"
            "[2] Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.\n"
            "[3] Radford, A., Metz, L. & Chintala, S. (2016). Unsupervised Representation Learning with "
            "Deep Convolutional Generative Adversarial Networks. ICLR.\n"
            "[4] Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.\n"
            "[5] Xiao, H., Rasul, K. & Vollgraf, F. (2017). Fashion-MNIST: a Novel Image Dataset for "
            "Benchmarking Machine Learning Algorithms. arXiv:1708.07747.\n"
            "[6] Tishby, N. & Zaslavsky, N. (2015). Deep Learning and the Information Bottleneck Principle. "
            "IEEE Information Theory Workshop.\n"
            "[7] Heusel, M. et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local "
            "Nash Equilibrium. NeurIPS.\n"
            "[8] Xiao, Z., Kreis, K. & Vahdat, A. (2022). Tackling the Generative Learning Trilemma with "
            "Denoising Diffusion GANs. ICLR.\n"
            "[9] Wang, Z. et al. (2004). Image Quality Assessment: From Error Visibility to Structural "
            "Similarity. IEEE Transactions on Image Processing, 13(4)."
        )
        y = paragraph(fig, references, y, fs=7.4, leading=1.18)
        check(y, "halaman 11")
        pdf.savefig(fig)
        plt.close(fig)

    from pypdf import PdfReader

    pages = len(PdfReader(str(output_path)).pages)
    print(output_path)
    print("PDF pages:", pages)
    if not 10 <= pages <= 12:
        print("PERINGATAN: jumlah halaman di luar 10-12.")
    shutil.rmtree(assets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Fashion-MNIST report PDF.")
    parser.add_argument("--archive", default=r"C:\Users\User\Downloads\DL2-hasil-laporan.zip")
    parser.add_argument("--output", default="TUGAS_DL_SESI07_2802477383_Samuel_Rafin_Djunaidi.pdf")
    parser.add_argument("--name", default="Samuel Rafin Djunaidi")
    parser.add_argument("--nim", default="2802477383")
    parser.add_argument("--class-name", default="LB01")
    parser.add_argument("--repository", default="https://github.com/SamuelRafin/GSLC-DL-2")
    parser.add_argument("--figures-dir", default=None,
                        help="Folder berisi figure_*.png yang menggantikan figur di arsip "
                             "(mis. hasil src/regenerate_figures.py).")
    arguments = parser.parse_args()
    build_report(arguments.archive, arguments.output, arguments.name, arguments.nim,
                 arguments.class_name, arguments.repository, arguments.figures_dir)
