import argparse
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT = 0.075, 0.075
CONTENT_WIDTH = 1 - LEFT - RIGHT


def new_page(title, number, subtitle=None):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor="white")
    fig.text(
        LEFT,
        0.975,
        "DEEP LEARNING (D6421)  |  STUDI GENERATIVE MODELS",
        fontsize=7.5,
        color="#486581",
        va="top",
    )
    fig.text(
        1 - RIGHT,
        0.975,
        "Fashion-MNIST",
        fontsize=7.5,
        color="#486581",
        ha="right",
        va="top",
    )
    fig.text(
        LEFT,
        0.935,
        title,
        fontsize=16,
        weight="bold",
        color="#102a43",
        va="top",
    )
    if subtitle:
        fig.text(
            LEFT,
            0.902,
            subtitle,
            fontsize=9,
            color="#486581",
            va="top",
        )
    fig.add_artist(
        plt.Line2D(
            [LEFT, 1 - RIGHT],
            [0.89, 0.89],
            transform=fig.transFigure,
            color="#bcccdc",
            lw=0.8,
        )
    )
    fig.text(
        0.5,
        0.025,
        f"Samuel Rafin Djunaidi  |  LB01  |  {number}",
        fontsize=7.5,
        color="#627d98",
        ha="center",
        va="bottom",
    )
    return fig


def paragraph(fig, text, y, fontsize=9.3, width=102, leading=1.28):
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
        else:
            lines.extend(
                textwrap.wrap(
                    raw,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [""]
            )
    fig.text(
        LEFT,
        y,
        "\n".join(lines),
        fontsize=fontsize,
        color="#243b53",
        va="top",
        linespacing=leading,
    )
    return y - max(1, len(lines)) * fontsize / (PAGE_H * 72) * leading - 0.012


def heading(fig, text, y, fontsize=11.2):
    fig.text(
        LEFT,
        y,
        text,
        fontsize=fontsize,
        weight="bold",
        color="#0b7285",
        va="top",
    )
    return y - 0.032


def add_image(fig, path, rect, caption=None):
    axis = fig.add_axes(rect)
    axis.imshow(Image.open(path).convert("RGB"))
    axis.axis("off")
    if caption:
        fig.text(
            rect[0],
            rect[1] - 0.018,
            caption,
            fontsize=7.8,
            color="#486581",
            va="top",
        )


def add_table(fig, rows, columns, rect, fontsize=7.6, widths=None, row_height=0.045):
    axis = fig.add_axes(rect)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        cell.set_linewidth(0.5)
        cell.set_height(row_height)
        if row == 0:
            cell.set_facecolor("#e6f4f1")
            cell.set_text_props(weight="bold", color="#102a43")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "white")
            cell.set_text_props(color="#243b53")


def make_training_figure(history, original_figure, output_path):
    source = Image.open(original_figure).convert("RGB")
    trajectory = source.crop(
        (int(source.width * 0.52), int(source.height * 0.17), source.width, int(source.height * 0.85))
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    figure.suptitle(
        "Training loss per objective dan reverse diffusion",
        fontsize=14,
        weight="bold",
    )
    epochs = range(1, 31)
    for key, label in (("vanilla_ae", "Vanilla AE"), ("sparse_ae", "Sparse AE")):
        axes[0, 0].plot(
            epochs,
            [row["reconstruction"] for row in history[key]],
            label=label,
        )
    axes[0, 0].set_title("Reconstruction MSE")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        epochs, [row["loss"] for row in history["vae"]], color="#2a9d8f"
    )
    axes[0, 1].set_title("VAE: BCE + KL")
    axes[0, 1].set_ylabel("Loss per sample")
    axes[0, 1].grid(alpha=0.25)

    for key, axis, title in (
        ("gan", axes[0, 2], "Vanilla GAN"),
        ("dcgan", axes[1, 0], "DCGAN"),
    ):
        axis.plot(
            epochs,
            [row["generator"] for row in history[key]],
            label="Generator",
        )
        axis.plot(
            epochs,
            [row["discriminator"] for row in history[key]],
            label="Discriminator",
        )
        axis.set_title(title)
        axis.set_ylabel("BCE loss")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)

    axes[1, 1].plot(
        epochs,
        [row["loss"] for row in history["ddpm"]],
        color="#6a4c93",
    )
    axes[1, 1].set_title("DDPM: noise prediction")
    axes[1, 1].set_ylabel("MSE(epsilon, epsilon_theta)")
    axes[1, 1].grid(alpha=0.25)

    axes[1, 2].imshow(trajectory)
    axes[1, 2].set_title("Reverse trajectory (t: 999 -> 0)")
    axes[1, 2].axis("off")
    for axis in axes.flat:
        if axis.has_data() and axis is not axes[1, 2]:
            axis.set_xlabel("Epoch")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_report(archive_path, output_path, name, nim, class_name, repository):
    output_path = Path(output_path)
    assets = output_path.parent / "_report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as bundle:
        metrics = json.loads(bundle.read("DL2-final/metrics.json"))
        history = json.loads(bundle.read("DL2-final/training_history.json"))
        figure_names = [
            "figure_1_reconstructions.png",
            "figure_2_generated_samples.png",
            "figure_3_classifier_limits.png",
            "figure_4_interpolation.png",
            "figure_5_training_and_diffusion.png",
        ]
        for figure_name in figure_names:
            (assets / figure_name).write_bytes(
                bundle.read(f"DL2-final/figures/{figure_name}")
            )

    training_figure = assets / "figure_5_training_and_diffusion_readable.png"
    make_training_figure(
        history,
        assets / "figure_5_training_and_diffusion.png",
        training_figure,
    )
    model_names = [
        ("Vanilla AE", "vanilla_ae"),
        ("Sparse AE", "sparse_ae"),
        ("VAE", "vae"),
        ("Vanilla GAN", "gan"),
        ("DCGAN", "dcgan"),
        ("DDPM", "ddpm"),
    ]
    epoch_seconds = {
        key: sum(row["seconds"] for row in history[key]) / len(history[key])
        for _, key in model_names
        if history.get(key)
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    page_width, page_height = PAGE_W, PAGE_H

    with PdfPages(output_path) as pdf:
        # Page 1: title and abstract
        fig = new_page(
            "Studi Komparatif Model Generatif",
            1,
            "Autoencoder, GAN, VAE, dan DDPM pada Fashion-MNIST",
        )
        fig.text(
            0.5,
            0.79,
            "LAPORAN EKSPERIMEN",
            ha="center",
            fontsize=10,
            color="#627d98",
            weight="bold",
        )
        fig.text(
            0.5,
            0.735,
            "AE  |  Sparse AE  |  VAE  |  GAN  |  DCGAN  |  DDPM",
            ha="center",
            fontsize=12,
            color="#102a43",
        )
        fig.text(
            0.5,
            0.655,
            "Identitas Penulis",
            ha="center",
            fontsize=11,
            weight="bold",
            color="#0b7285",
        )
        fig.text(0.5, 0.615, name, ha="center", fontsize=11, color="#243b53")
        fig.text(
            0.5,
            0.58,
            f"NIM {nim}  |  Kelas {class_name}",
            ha="center",
            fontsize=10,
            color="#243b53",
        )
        fig.text(
            0.5,
            0.535,
            f"Repositori: {repository}",
            ha="center",
            fontsize=8.2,
            color="#16697a",
            url=repository,
        )
        y = heading(fig, "Abstrak", 0.46, 12)
        abstract = (
            "Eksperimen ini membandingkan enam model generatif pada Fashion-MNIST "
            "dengan pembagian data 80:10:10, seed 42, batch 128, dan 30 epoch untuk "
            "model generatif. Sparse Autoencoder memperoleh MSE test 0,00769, "
            "sedikit lebih rendah daripada Vanilla Autoencoder (0,00818), sedangkan "
            "VAE memperoleh MSE 0,01337. Pada evaluasi FID dengan 1.024 sampel, "
            "DDPM menghasilkan skor terendah (37,20), disusul DCGAN (46,20), VAE "
            "(102,43), dan Vanilla GAN (151,76). MLP classifier dari checkpoint "
            "mencapai akurasi test 89,09%, melampaui batas minimum 88%. Keunggulan "
            "kualitas tersebut datang dengan biaya sampling DDPM sekitar 12,37 "
            "detik per batch 64 pada Tesla T4. Hasil dibatasi resolusi Fashion-MNIST, "
            "satu seed, dan penggunaan Inception v3 pra-latih ImageNet untuk FID."
        )
        paragraph(fig, abstract, y, fontsize=9.4, width=105)
        y = heading(fig, "Kata kunci", 0.215, 10)
        fig.text(
            LEFT,
            0.183,
            "representasi laten; rekonstruksi; GAN; diffusion; Fashion-MNIST",
            fontsize=8.8,
            color="#486581",
            va="top",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: introduction
        fig = new_page("1. Pendahuluan dan Tujuan", 2)
        y = 0.855
        for title, body in [
            (
                "Latar belakang",
                "Model generatif berupaya mempelajari struktur data agar dapat "
                "merekonstruksi atau menghasilkan contoh baru. Autoencoder "
                "mengoptimalkan rekonstruksi, VAE memberi prior probabilistik "
                "pada kode laten, GAN belajar melalui permainan generator-"
                "discriminator, sedangkan DDPM mempelajari prediksi derau untuk "
                "membalik proses difusi.",
            ),
            (
                "Tujuan",
                "Studi ini mengukur fidelitas rekonstruksi, kualitas sampling, "
                "latensi, dan perilaku ruang laten pada enam arsitektur. Eksperimen "
                "classifier terpisah digunakan untuk menunjukkan bahwa fitur yang "
                "berguna bagi prediksi label tidak otomatis menjadi representasi "
                "generatif.",
            ),
            (
                "Pertanyaan penelitian",
                "Q1 - Bagaimana perbandingan MSE, PSNR, SSIM, dan FID?\n"
                "Q2 - Bagaimana objective pixel-wise dan adversarial berkaitan "
                "dengan blur, tepi, dan sparsity fitur?\n"
                "Q3 - Apa perbedaan dinamika training GAN/DCGAN dengan DDPM, dan "
                "trade-off kualitas-kecepatan?\n"
                "Q4 - Bagaimana interpolasi antar titik laten pada kelima model "
                "yang dapat disampling/didekode?\n"
                "Q5 - Mengapa bobot dan latent classifier tidak dapat langsung "
                "membangkitkan citra?\n"
                "Q6 - Model mana yang layak dipertimbangkan untuk anomali, avatar "
                "waktu nyata, dan aset visual?",
            ),
            (
                "Ruang lingkup",
                "Benchmark yang dipakai adalah Fashion-MNIST grayscale 28x28, "
                "bukan data inspeksi manufaktur, avatar ponsel, atau aset resolusi "
                "tinggi. Karena itu, rekomendasi industri di bagian akhir merupakan "
                "penilaian awal berbasis karakteristik model dan hasil benchmark "
                "ini, bukan bukti validasi deployment.",
            ),
        ]:
            y = heading(fig, title, y)
            y = paragraph(fig, body, y, fontsize=9.2, width=105)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: architectures and objectives
        fig = new_page(
            "2. Arsitektur dan Formulasi Objektif",
            3,
            "Skema ringkas arsitektur dan loss yang digunakan",
        )
        y = paragraph(
            fig,
            "Seluruh citra masuk berukuran 28x28x1. AE/VAE memakai piksel [0,1]; "
            "GAN dan DDPM memakai rentang [-1,1]. Dimensi laten AE/VAE/classifier "
            "adalah 64 dan noise GAN 100.",
            0.855,
            fontsize=9.1,
            width=105,
        )
        rows = [
            [
                "Vanilla AE",
                "784 -> 256 ReLU -> 64 ReLU -> 256 ReLU -> 784 Sigmoid",
                "MSE(x, xhat)",
            ],
            [
                "Sparse AE",
                "Encoder AE; penalti pada hidden h 256-dimensi",
                "MSE + lambda mean(|h|), lambda=0.001",
            ],
            [
                "VAE",
                "784 -> 256; mu, logvar 64 -> z -> 256 -> 784",
                "BCE-sum/sample + beta KL(q(z|x)||N(0,I)); beta=1",
            ],
            [
                "Vanilla GAN",
                "G: 100->256->512->784; D: 784->512->256->1",
                "BCEWithLogits; generator non-saturating",
            ],
            [
                "DCGAN",
                "G: ConvTranspose 100x1x1->128x7x7->64x14x14->1x28x28; D konvolusional",
                "BCEWithLogits; generator non-saturating",
            ],
            [
                "DDPM",
                "U-Net residual konvolusional dengan embedding waktu",
                "MSE(epsilon, epsilon_theta(x_t,t)); beta 1e-4->0.02",
            ],
            [
                "MLP classifier",
                "784 -> 256 ReLU -> 64 ReLU -> 10 logits",
                "Cross-entropy; bukan objective P(x)",
            ],
        ]
        add_table(
            fig,
            rows,
            ["Model", "Skema", "Loss utama"],
            [LEFT, 0.37, CONTENT_WIDTH, 0.37],
            fontsize=7.3,
            widths=[0.17, 0.48, 0.35],
            row_height=0.045,
        )
        y = 0.32
        y = heading(fig, "Makna matematis", y)
        paragraph(
            fig,
            "Untuk AE, Lrec = E[||x-xhat||^2]. Sparse AE menambah penalti L1 pada "
            "hidden activation sebelum bottleneck. VAE menggunakan "
            "z=mu+exp(0.5 logvar)*epsilon, epsilon~N(0,I), dan KL terhadap prior "
            "normal. GAN meminimalkan loss discriminator pada data nyata/palsu "
            "serta BCE generator terhadap label nyata. DDPM membentuk "
            "x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon dan belajar "
            "memperkirakan epsilon.",
            y,
            fontsize=8.6,
            width=108,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: methods
        fig = new_page("3. Metode Eksperimen dan Evaluasi", 4)
        y = 0.855
        for title, body in [
            (
                "Dataset dan split",
                "Fashion-MNIST memuat 70.000 citra. Partisi resmi train dan test "
                "digabung lalu dibagi deterministik menjadi 56.000 train, 7.000 "
                "validation, dan 7.000 test menggunakan seed 42. Preprocessing "
                "memakai ToTensor tanpa augmentasi. Split yang dipakai adalah "
                "split eksperimen 80:10:10, bukan test resmi dataset.",
            ),
            (
                "Konfigurasi dan perangkat",
                "Training generatif memakai batch 128 dan 30 epoch. Optimizer "
                "Adam: 1e-3 untuk AE, Sparse AE, dan VAE; 2e-4 untuk Vanilla GAN, "
                "DCGAN, dan DDPM. GAN memakai beta1=0.5, beta2=0.999. DDPM memakai "
                "1.000 langkah. Eksperimen dijalankan pada Tesla T4, Python 3.13.16, "
                "PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0.",
            ),
            (
                "Metrik",
                "MSE dihitung sebagai rata-rata galat piksel; PSNR=10 log10(1/MSE), "
                "rentang piksel [0,1]. SSIM dihitung dari statistik lokal jendela "
                "7x7. FID dihitung dari 1.024 sampel real dan sintetis memakai "
                "Inception v3 pra-latih ImageNet; citra grayscale diulang menjadi "
                "RGB dan di-resize ke 299x299. FID lebih rendah menunjukkan jarak "
                "fitur lebih kecil, tetapi representasi ImageNet mungkin kurang "
                "sesuai untuk Fashion-MNIST.",
            ),
            (
                "Latensi",
                "Latensi adalah waktu inferensi satu batch 64 pada T4, dengan "
                "warm-up dan sinkronisasi CUDA. Generator AE/VAE/GAN/DCGAN dihitung "
                "satu lintasan; DDPM memakai seluruh 1.000 langkah reverse sampling. "
                "Angka ini tidak mewakili latensi ponsel.",
            ),
            (
                "Reproduksibilitas",
                "Seed utama 42, satu kali pengulangan. MLP classifier dilatih terlebih "
                "dahulu pada train split yang sama (20 epoch, Adam lr 1e-3, batch "
                "128, bobot epoch terakhir); prosedurnya tersedia di "
                "src/train_classifier.py dan checkpoint disertakan di repositori. "
                "Pipeline menguji kembali checkpoint pada test split yang sama. "
                "Decoder frozen dilatih 10 epoch pada latent classifier beku.",
            ),
        ]:
            y = heading(fig, title, y)
            y = paragraph(fig, body, y, fontsize=9.1, width=105)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: classifier experiment
        fig = new_page("4. Uji Batasan MLP Classifier", 5)
        y = 0.855
        y = paragraph(
            fig,
            "Classifier memodelkan P(y|x), bukan P(x). Cross-entropy hanya memberi "
            "insentif mempertahankan informasi yang membantu prediksi kelas; detail "
            "variasi intra-kelas dapat terabaikan, meski tidak berarti setiap "
            "classifier pasti membuang seluruh detail. Bobot forward juga umumnya "
            "bukan matriks ortogonal/invertibel, dan ReLU memotong nilai negatif.",
            y,
            fontsize=9.1,
            width=105,
        )
        y = heading(fig, "Hasil bobot dan akurasi", y)
        y = paragraph(
            fig,
            "Akurasi test MLP adalah 89,09%, melewati target 88%. Ketiga matriks "
            "berbentuk persegi panjang: W1 (256x784, rank 256), W2 (64x256, rank "
            "64), W3 (10x64, rank 10). Diagnostik ||W^T W-I||F/sqrt(n) masing-masing "
            "16,462; 3,925; dan 1,453. Karena dimensi tidak sama dan rank lebih "
            "rendah dari dimensi input, transpose tidak dapat dianggap sebagai "
            "inverse. Proyeksi juga mengabaikan bias dan tidak membalik ReLU secara unik.",
            y,
            fontsize=8.8,
            width=108,
        )
        y = heading(fig, "Rekonstruksi dari latent beku", y)
        paragraph(
            fig,
            "Decoder yang dilatih pada z_MLP beku menghasilkan MSE 0,01860, PSNR "
            "18,02 dB, dan SSIM 0,7085; AE berdimensi laten sama menghasilkan MSE "
            "0,00818, PSNR 21,91 dB, dan SSIM 0,8433. Pada gambar, proyeksi balik "
            "bobot tampak berupa bercak/struktur rusak, sedangkan prior sampling "
            "latent MLP menghasilkan bentuk yang tidak teratur. Temuan ini "
            "mendukung bahwa fitur classifier bukan decoder generatif siap pakai.",
            y,
            fontsize=8.8,
            width=108,
        )
        add_image(
            fig,
            assets / "figure_3_classifier_limits.png",
            [LEFT, 0.105, CONTENT_WIDTH, 0.205],
            "Figur 3. Dari kiri: citra asli, dua proyeksi transpose, decoder latent MLP, AE, prior latent MLP, dan VAE.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: metric table
        fig = new_page("5. Hasil Kuantitatif dan Q1", 6)
        rows = []
        for label, key in model_names:
            result = metrics["models"][key]
            mse = f"{result['test_mse']:.5f}" if "test_mse" in result else "-"
            psnr = f"{result['test_psnr_db']:.2f}" if "test_psnr_db" in result else "-"
            ssim = f"{result['test_ssim']:.3f}" if "test_ssim" in result else "-"
            fid = f"{result['fid']:.2f}" if "fid" in result else "-"
            rows.append(
                [
                    label,
                    f"{epoch_seconds[key]:.2f}",
                    f"{result['latency_ms_batch64']:.2f}",
                    mse,
                    psnr,
                    ssim,
                    fid,
                ]
            )
        add_table(
            fig,
            rows,
            ["Model", "s/epoch", "ms/batch 64", "MSE", "PSNR dB", "SSIM", "FID"],
            [LEFT, 0.40, CONTENT_WIDTH, 0.34],
            fontsize=7.6,
            widths=[0.19, 0.12, 0.15, 0.13, 0.13, 0.11, 0.17],
            row_height=0.043,
        )
        y = 0.345
        y = paragraph(
            fig,
            "MSE/PSNR/SSIM hanya bermakna untuk rekonstruksi; FID hanya dihitung "
            "untuk model sampling. Tanda '-' berarti tidak berlaku, bukan nol. "
            "Latensi DDPM adalah 12.368 ms (sekitar 12,37 s) untuk 1.000 langkah; "
            "nilai lain diukur untuk satu lintasan.",
            y,
            fontsize=8.2,
            width=110,
        )
        y = heading(fig, "Jawaban Q1", y)
        y = paragraph(
            fig,
            "Sparse AE mengungguli Vanilla AE pada ketiga metrik rekonstruksi di "
            "eksperimen ini: MSE turun sekitar 6,0%, PSNR naik 0,29 dB, dan SSIM "
            "naik 0,008. Jadi penalti lambda=0,001 tidak menurunkan fidelitas pada "
            "konfigurasi ini; generalisasi ke nilai lambda atau dataset lain belum "
            "diuji. VAE memiliki galat rekonstruksi lebih tinggi, tetapi ruang laten "
            "berprior memudahkan sampling. Pada FID, DDPM (37,20) terbaik, DCGAN "
            "(46,20) berikutnya, lalu VAE (102,43) dan Vanilla GAN (151,76). FID ini "
            "berbasis 1.024 sampel dan Inception ImageNet, sehingga peringkat perlu "
            "dibaca bersama figur dan batas domain.",
            y,
            fontsize=8.5,
            width=108,
        )
        y = heading(fig, "Waktu training", y)
        paragraph(
            fig,
            "Rata-rata waktu per epoch: AE 5,25 s; Sparse AE 5,34 s; VAE 5,39 s; "
            "Vanilla GAN 5,24 s; DCGAN 12,79 s; DDPM 31,84 s. Durasi training "
            "classifier tidak dicatat pada checkpoint yang dipakai.",
            y,
            fontsize=8.5,
            width=108,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: reconstruction figure
        fig = new_page("6. Rekonstruksi dan Karakteristik Visual", 7)
        y = paragraph(
            fig,
            "Figur 1 menggunakan delapan citra test yang sama pada empat baris: "
            "asli, Vanilla AE, Sparse AE, dan VAE. AE mempertahankan siluet pakaian, "
            "tetapi detail kecil dan batas tepi tampak halus. Sparse AE memperlihatkan "
            "fidelitas rata-rata sedikit lebih baik sesuai MSE/SSIM. VAE lebih kabur, "
            "sejalan dengan objective rekonstruksi plus regularisasi KL.",
            0.855,
            fontsize=9.0,
            width=105,
        )
        add_image(
            fig,
            assets / "figure_1_reconstructions.png",
            [LEFT, 0.39, CONTENT_WIDTH, 0.345],
            "Figur 1. Rekonstruksi delapan sampel test. Baris dari atas: asli, Vanilla AE, Sparse AE, VAE.",
        )
        y = 0.31
        y = heading(fig, "Jawaban Q2: loss dan ketajaman", y)
        paragraph(
            fig,
            "MSE pixel-wise menghukum perbedaan intensitas di setiap piksel. Bila "
            "satu representasi kompatibel dengan beberapa detail yang masuk akal, "
            "prediksi rata-rata kondisional E[x|z] cenderung meratakan tepi. BCE "
            "memberi penalti pixel-wise probabilistik tetapi tidak secara langsung "
            "memaksa tekstur tampak realistis. GAN menilai realisme melalui "
            "discriminator; pada GAN ideal dengan discriminator optimal, objective "
            "minimax terkait dengan Jensen-Shannon divergence. Implementasi "
            "generator non-saturating yang dipakai mengoptimalkan BCE generator, "
            "bukan menghitung D_JS secara langsung. Penalti L1 Sparse AE mendorong "
            "hidden units yang aktif lebih jarang; hasil test di sini menunjukkan "
            "kompromi sparsity-fidelitas yang menguntungkan, tetapi struktur fitur "
            "tidak dianalisis per-unit.",
            y,
            fontsize=8.4,
            width=110,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 8: random generation
        fig = new_page("7. Sampling Acak dan Kualitas Generasi", 8)
        y = paragraph(
            fig,
            "Figur 2 menampilkan 16 sampel per model. Baris dari atas: VAE, "
            "Vanilla GAN, DCGAN, dan DDPM. Sampel VAE cenderung lembut; Vanilla GAN "
            "menampilkan sejumlah bentuk yang terputus/bertekstur kasar; DCGAN "
            "menghasilkan siluet pakaian yang lebih jelas; DDPM membentuk beberapa "
            "objek yang dapat dikenali dengan variasi antarcontoh.",
            0.855,
            fontsize=9.0,
            width=105,
        )
        add_image(
            fig,
            assets / "figure_2_generated_samples.png",
            [LEFT, 0.52, CONTENT_WIDTH, 0.22],
            "Figur 2. Grid sampling acak, 16 sampel per model.",
        )
        y = 0.45
        y = heading(fig, "Interpretasi FID", y)
        y = paragraph(
            fig,
            "Urutan skor FID - DDPM 37,20; DCGAN 46,20; VAE 102,43; Vanilla GAN "
            "151,76 - sejalan dengan impresi umum bahwa sampel DDPM/DCGAN lebih "
            "dekat ke fitur data daripada sampel Vanilla GAN. Namun, FID tidak "
            "mengukur mode coverage secara terpisah dan sensitif pada extractor "
            "fitur. Fashion-MNIST kecil dan grayscale bukan domain alami yang "
            "menjadi sasaran Inception v3; angka ini adalah pembanding internal "
            "dengan protokol sama, bukan penilaian universal kualitas.",
            y,
            fontsize=8.8,
            width=108,
        )
        y = heading(fig, "Hubungan dengan Q1 dan Q3", y)
        paragraph(
            fig,
            "VAE menunjukkan trade-off: latent prior memudahkan sampling kontinu, "
            "tetapi rekonstruksi dan FID masih tertinggal dari DDPM/DCGAN. Vanilla "
            "GAN paling cepat di antara model sampling yang diuji, tetapi memiliki "
            "FID tertinggi. Kecepatan satu lintasan saja tidak cukup untuk memilih "
            "generator.",
            y,
            fontsize=8.8,
            width=108,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 9: interpolation
        fig = new_page("8. Interpolasi Ruang Laten dan Q4", 9)
        y = paragraph(
            fig,
            "Dua kode laten diinterpolasi pada enam nilai alpha=0, 0,2, ..., 1. "
            "Baris gambar berurutan: Vanilla AE, Sparse AE, VAE, Vanilla GAN, dan "
            "DCGAN. Pada AE/Sparse AE/VAE, endpoint berasal dari encoding dua citra "
            "test dengan kelas berbeda. Pada GAN, endpoint adalah dua vektor noise "
            "acak, karena generator tidak memiliki encoder untuk memetakan citra "
            "endpoint ke latent.",
            0.855,
            fontsize=9.0,
            width=105,
        )
        add_image(
            fig,
            assets / "figure_4_interpolation.png",
            [LEFT, 0.34, CONTENT_WIDTH, 0.425],
            "Figur 4. Interpolasi linier pada kode laten. Alpha meningkat dari kiri ke kanan.",
        )
        y = 0.27
        y = heading(fig, "Analisis", y)
        paragraph(
            fig,
            "Pada sampel ini, perubahan AE dan Sparse AE relatif bertahap tetapi "
            "objek di antara endpoint dapat kehilangan ciri kelas yang tegas. VAE "
            "juga berubah lembut; regularisasi KL mendorong q(z|x) mendekati prior "
            "normal, namun tidak menjamin semua interpolasi semantik mulus. "
            "Interpolasi GAN berada di ruang noise yang dipelajari, bukan encoding "
            "endpoint gambar. Figur ini menunjukkan satu pasangan sampel, bukan "
            "bukti bahwa seluruh ruang laten bebas dari void atau artefak.",
            y,
            fontsize=8.7,
            width=108,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 10: training dynamics
        fig = new_page("9. Dinamika Training dan Generative Trilemma", 10)
        y = paragraph(
            fig,
            "Kurva pada Figur 5 dipisahkan menurut skala objective agar perbedaan "
            "unit tidak menyamarkan kurva AE/GAN. VAE memakai BCE yang dijumlahkan "
            "atas piksel per sampel ditambah KL; karena skala loss berbeda, kurvanya "
            "tidak dibandingkan tinggi absolutnya dengan MSE AE atau loss GAN. "
            "Riwayat MLP classifier tidak termasuk karena classifier dibaca dari "
            "checkpoint.",
            0.855,
            fontsize=8.9,
            width=105,
        )
        add_image(
            fig,
            training_figure,
            [LEFT, 0.345, CONTENT_WIDTH, 0.39],
            "Figur 5. Loss per objective dan beberapa tahap reverse diffusion.",
        )
        y = 0.29
        y = heading(fig, "Jawaban Q3: stabilitas dan kecepatan", y)
        y = paragraph(
            fig,
            "Loss rekonstruksi AE dan prediksi derau DDPM menurun selama 30 epoch. "
            "Loss generator/discriminator GAN berfluktuasi; kurva yang tersedia "
            "tidak cukup untuk menyatakan mode collapse, tetapi dinamika adversarial "
            "memang mengoptimalkan dua jaringan yang saling berlawanan. DCGAN "
            "memakai bias konvolusional dan normalisasi sehingga sampel/kurva di run "
            "ini lebih baik daripada Vanilla GAN, tanpa menjamin kestabilan pada "
            "semua konfigurasi. DDPM mengoptimalkan supervised noise prediction, "
            "bukan permainan minimax.",
            y,
            fontsize=8.5,
            width=110,
        )
        y = heading(fig, "Trade-off kualitas-kecepatan-coverage", y)
        paragraph(
            fig,
            "DDPM memberi FID terendah (37,20), tetapi memerlukan 12,37 s per batch "
            "64. DCGAN mencapai FID 46,20 dengan latensi 4,31 ms; Vanilla GAN "
            "0,30 ms namun FID 151,76. Mode coverage tidak diukur secara eksplisit, "
            "sehingga tidak dapat disimpulkan hanya dari FID atau grid 16 sampel.",
            y,
            fontsize=8.5,
            width=110,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 11: implications and references
        fig = new_page("10. Sintesis, Rekomendasi, dan Kesimpulan", 11)
        y = 0.855
        y = heading(fig, "Jawaban Q5: mengapa classifier bukan generator", y)
        y = paragraph(
            fig,
            "Classifier dilatih untuk P(y|x); ia tidak mempelajari density P(x) atau "
            "prior laten P(z). Ukuran bobot non-square dan rank yang dilaporkan "
            "meniadakan inverse dua sisi, W^T bukan W^-1, dan ReLU memetakan semua "
            "nilai negatif ke nol sehingga informasinya tidak dapat dipulihkan unik. "
            "Hasil transpose menghasilkan bercak/artefak. Decoder terpisah pada "
            "z_MLP yang dibekukan memang merekonstruksi sebagian bentuk, tetapi "
            "MSE/PSNR/SSIM lebih buruk daripada AE. Sampling Gaussian dari statistik "
            "latent test juga tidak menjamin titik jatuh pada manifold data.",
            y,
            fontsize=8.5,
            width=108,
        )
        y = heading(fig, "Jawaban Q6: rekomendasi bersyarat", y)
        y = paragraph(
            fig,
            "(1) Deteksi anomali: Sparse AE layak diuji lanjut karena memperoleh "
            "galat normal test terendah di antara dua AE. Namun eksperimen ini tidak "
            "memuat citra anomali; sensitivitas dan threshold belum tervalidasi. "
            "Uji pada data inspeksi normal/anomali berlabel sebelum deployment. "
            "(2) Avatar waktu nyata: pada T4, VAE, Vanilla GAN, dan DCGAN masing-"
            "masing sekitar 0,11; 0,30; dan 4,31 ms/batch 64, semuanya di bawah "
            "30 ms pada perangkat uji. DCGAN memberi kompromi kualitas/kecepatan "
            "yang lebih menarik pada benchmark ini; klaim untuk ponsel menunggu "
            "pengukuran pada perangkat mobile. (3) Aset visual: DDPM memiliki FID "
            "terendah, cocok sebagai kandidat generasi offline, tetapi lambat dan "
            "diuji hanya pada 28x28. Untuk output cepat, DCGAN menjadi alternatif.",
            y,
            fontsize=8.2,
            width=108,
        )
        y = heading(fig, "Kesimpulan dan batasan", y)
        y = paragraph(
            fig,
            "Pada run ini Sparse AE paling baik untuk rekonstruksi di antara AE "
            "yang dibandingkan; DDPM unggul dalam FID, dengan biaya sampling "
            "iteratif yang besar. MLP memenuhi ambang akurasi tetapi latent "
            "classifier tetap kurang cocok untuk rekonstruksi dibanding AE. "
            "Kesimpulan dibatasi satu seed, Fashion-MNIST resolusi rendah, 1.024 "
            "sampel FID berbasis Inception ImageNet, dan tidak adanya uji anomali "
            "atau perangkat mobile. Latensi adalah hasil konfigurasi T4 ini, "
            "bukan jaminan deployment.",
            y,
            fontsize=8.0,
            width=110,
        )
        y = heading(fig, "Referensi", y, 10.5)
        references = (
            "[1] Goodfellow, I. et al. (2014). Generative Adversarial Nets. NeurIPS.\n"
            "[2] Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.\n"
            "[3] Radford, A., Metz, L. & Chintala, S. (2016). Unsupervised "
            "Representation Learning with Deep Convolutional Generative Adversarial "
            "Networks. ICLR.\n"
            "[4] Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion "
            "Probabilistic Models. NeurIPS.\n"
            "[5] Xiao, H., Rasul, K. & Vollgraf, F. (2017). Fashion-MNIST: a Novel "
            "Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747."
        )
        paragraph(fig, references, y, fontsize=7.4, width=115, leading=1.16)
        pdf.savefig(fig)
        plt.close(fig)

    print(output_path)
    print("PDF pages:", len(__import__("pypdf").PdfReader(str(output_path)).pages))
    shutil.rmtree(assets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Fashion-MNIST report PDF.")
    parser.add_argument(
        "--archive",
        default=r"C:\Users\User\Downloads\DL2-hasil-laporan.zip",
    )
    parser.add_argument(
        "--output",
        default="TUGAS_DL_SESI07_2802477383_Samuel_Rafin_Djunaidi.pdf",
    )
    parser.add_argument("--name", default="Samuel Rafin Djunaidi")
    parser.add_argument("--nim", default="2802477383")
    parser.add_argument("--class-name", default="LB01")
    parser.add_argument(
        "--repository",
        default="https://github.com/SamuelRafin/GSLC-DL-2",
    )
    arguments = parser.parse_args()
    build_report(
        arguments.archive,
        arguments.output,
        arguments.name,
        arguments.nim,
        arguments.class_name,
        arguments.repository,
    )
