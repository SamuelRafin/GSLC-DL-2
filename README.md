# Studi Generative Models pada Fashion-MNIST

Repositori ini berisi pipeline eksperimen untuk membandingkan Vanilla AE, Sparse AE,
VAE, Vanilla GAN, DCGAN, dan DDPM. Satu eksperimen juga melatih MLP classifier,
memeriksa proyeksi balik bobotnya, lalu membandingkan decoder yang dilatih pada fitur
MLP beku dengan decoder Autoencoder.

Kode menyiapkan pembagian data 80:10:10, seed tetap, checkpoint, kurva training,
metrik test, dan lima gambar komparatif. Angka pada laporan harus diambil dari berkas
hasil eksperimen, bukan diisi dengan perkiraan.

## Menyiapkan lingkungan

Gunakan versi Python yang didukung oleh build PyTorch yang dipilih (contoh berikut
menggunakan Python 3.12) dan lingkungan virtual. Contoh di PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika menggunakan GPU, pasang build PyTorch yang sesuai dengan versi CUDA di komputer
melalui petunjuk instalasi resmi PyTorch, lalu cek `torch.cuda.is_available()`.
Jalankan semua eksperimen pada perangkat yang sama ketika membandingkan latensi.

## Menjalankan eksperimen

Training mengambil Fashion-MNIST secara otomatis dan membuat split deterministik
80% train, 10% validation, dan 10% test dari gabungan data resmi train dan test.
Test split tidak dipakai untuk optimasi model. Default 10 epoch ditujukan untuk
mengecek pipeline; jalankan lebih banyak epoch sebelum menarik kesimpulan kualitas.

```powershell
python src\run_experiment.py --device auto --epochs 30 --classifier-epochs 25 --decoder-epochs 10 --ddpm-steps 1000
```

Jika sudah memiliki checkpoint `mlp_classifier.pt` yang dibuat dengan
`torch.save(model.state_dict(), "mlp_classifier.pt")` dari arsitektur
784 → 256 → 64 → 10, classifier itu dapat dipakai tanpa training ulang:

```powershell
python src\run_experiment.py --device auto --classifier-checkpoint "C:\path\to\mlp_classifier.pt" --epochs 30 --ddpm-steps 1000
```

Checkpoint hanya berisi bobot, jadi pastikan berasal dari Fashion-MNIST dengan
preprocessing `ToTensor()` dan split 80:10:10 seed 42 yang sama. Pipeline tetap
mengukur ulang akurasi pada test split-nya dan melatih frozen latent decoder
menggunakan train split yang sama.

Jika training selesai tetapi proses berhenti saat evaluasi, checkpoint hasil
training dapat dimuat ulang tanpa mengulangi training:

```powershell
python src\run_experiment.py --device auto --load-checkpoints "outputs\checkpoints" --output "outputs"
```

Folder tersebut harus berisi checkpoint keenam model, `classifier.pt`, dan
`frozen_decoder.pt`. Training history sebelumnya hanya tersedia jika proses sudah
sempat menulis `training_history.json`; pemuatan checkpoint saja tidak dapat
merekonstruksi kurva loss yang belum tersimpan.

## Membuat PDF laporan dari hasil eksperimen

Setelah mengunduh ZIP keluaran eksperimen ke komputer, buat laporan 11 halaman
dengan data metrik dan figur dari ZIP tersebut:

```powershell
python build_report.py --archive "C:\path\to\DL2-hasil-laporan.zip" --output "TUGAS_DL_SESI07_2802477383_Samuel_Rafin_Djunaidi.pdf"
```

Argumen `--name`, `--nim`, `--class-name`, dan `--repository` dapat dipakai untuk
mengubah identitas pada halaman judul.

DDPM dengan 1.000 langkah dan pengukuran FID memerlukan waktu lebih lama. Untuk
memeriksa bahwa kode berjalan, kurangi epoch/langkah dan gunakan `--skip-fid`;
hasil konfigurasi singkat itu jangan dicampur dengan hasil eksperimen final. Opsi
lain dapat dilihat dengan:

```powershell
python src\run_experiment.py --help
```

FID memakai Inception v3 pra-latih ImageNet. Bobotnya akan diunduh saat FID pertama
kali dihitung; karena Fashion-MNIST berupa citra grayscale 28×28, setiap citra
diubah menjadi tiga kanal dan di-resize untuk masukan Inception.

## Berkas keluaran

- `outputs\metrics.json`: MSE, PSNR, SSIM, FID, latensi batch 64, akurasi classifier,
  statistik latent, dan diagnostik matriks bobot.
- `outputs\training_history.json` dan `outputs\training_history.csv`: loss per epoch
  dan waktu training.
- `outputs\checkpoints\`: bobot hasil training.
- `outputs\figures\figure_1_...png` sampai `figure_5_...png`: visualisasi yang diminta.

Sebelum menulis laporan, periksa akurasi test MLP (target minimal 88%), kestabilan
kurva, FID, latensi, dan apakah gambar yang dihasilkan benar-benar mendukung
kesimpulan. Bila hasil tidak memenuhi target, tambah epoch atau perbaiki konfigurasi,
kemudian jalankan ulang dan simpan konfigurasi yang dipakai.

## Catatan reproduktibilitas

Seed default adalah 42. Split data dan urutan batch training dibuat deterministik,
tetapi operasi GPU tertentu dapat tetap menghasilkan selisih kecil antarperangkat.
Simpan versi Python/PyTorch/CUDA bersama hasil final. Jangan mengunggah data,
checkpoint besar, atau berkas keluaran sebelum memeriksa ukuran dan lisensinya.
