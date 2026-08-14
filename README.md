# Training Ulang Model Hangeul

Paket ini melatih VGG16, ResNet50, MobileNetV2, EfficientNetB0, dan Xception
sesuai konfigurasi skripsi Erlangga Bagas Yunanta. Dataset asli tetap berjumlah
1.760 citra. Augmentasi dibuat secara otomatis hanya pada batch training.

## Konfigurasi utama

- Pembagian subject-wise: train 38 responden, validation 8, test 9.
- Input 224 x 224 RGB.
- Augmentasi training: rotasi 15 derajat, zoom 0,20, pergeseran horizontal dan
  vertikal 0,15, shear 0,20, fill mode nearest.
- Classifier: GlobalAveragePooling2D, Dense 512 ReLU, Dropout 0,5, Dense 32
  Softmax.
- Feature extraction: Adam, learning rate 0,001, maksimal 50 epoch.
- Fine-tuning: Adam, learning rate 0,00001, maksimal 20 epoch.
- Batch size 16.
- EarlyStopping patience 8, ReduceLROnPlateau factor 0,2 dan patience 4.

## Menjalankan lokal dengan RTX 4060, VS Code, dan WSL2

TensorFlow versi modern tidak mendukung CUDA secara native di Windows. Buka
folder Windows melalui VS Code yang terhubung ke WSL2.

Dataset pada contoh berikut berada di `E:\skripsi_nanta` dan langsung berisi
folder `train`, `validation`, dan `test`.

Di terminal Ubuntu WSL jalankan:

```bash
cd /mnt/e/skripsi_nanta
code .
```

Salin atau ekstrak folder `hangeul_training` dari paket ini ke dalam
`E:\skripsi_nanta`. Struktur akhirnya:

```text
E:\skripsi_nanta
  train
  validation
  test
  hangeul_training
```

Buka terminal WSL di VS Code, lalu jalankan:

```bash
cd /mnt/e/skripsi_nanta
nvidia-smi
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r hangeul_training/requirements.txt
```

Verifikasi TensorFlow dan CUDA:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Hasil yang benar harus menampilkan minimal satu perangkat GPU. Setelah itu,
jalankan smoke test MobileNetV2:

```bash
python hangeul_training/train_hangeul.py \
  --dataset-dir /mnt/e/skripsi_nanta \
  --output-dir /mnt/e/skripsi_nanta/training_output_smoke \
  --models mobilenetv2 \
  --smoke-test
```

Setelah smoke test selesai tanpa error, jalankan training penuh:

```bash
python hangeul_training/train_hangeul.py \
  --dataset-dir /mnt/e/skripsi_nanta \
  --output-dir /mnt/e/skripsi_nanta/training_output
```

Untuk memantau penggunaan GPU dari terminal WSL kedua:

```bash
watch -n 1 nvidia-smi
```

## Menjalankan di Google Colab T4

1. Aktifkan `Runtime > Change runtime type > T4 GPU`.
2. Unggah `hangeul_training_bundle.zip` dan
   `dataset_tf-20260806T133946Z-1-001.zip` ke Colab.
3. Jalankan perintah berikut pada satu sel:

```bash
!unzip -q hangeul_training_bundle.zip
!pip install -q -r hangeul_training/requirements.txt
!python hangeul_training/train_hangeul.py \
  --dataset-zip dataset_tf-20260806T133946Z-1-001.zip \
  --output-dir /content/training_output
```

Untuk menyimpan hasil langsung ke Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Kemudian ubah `--output-dir` menjadi:

```text
/content/drive/MyDrive/hangeul_training_output
```

## Uji pipeline sebelum training penuh

Jalankan satu model dengan satu epoch pada setiap tahap:

```bash
!python hangeul_training/train_hangeul.py \
  --dataset-zip dataset_tf-20260806T133946Z-1-001.zip \
  --output-dir /content/smoke_output \
  --models mobilenetv2 \
  --smoke-test
```

Jika uji ini selesai tanpa error, jalankan seluruh model tanpa opsi
`--smoke-test`.

## Menjalankan model satu per satu

Contoh VGG16:

```bash
!python hangeul_training/train_hangeul.py \
  --dataset-zip dataset_tf-20260806T133946Z-1-001.zip \
  --output-dir /content/drive/MyDrive/hangeul_training_output_vgg16 \
  --models vgg16
```

Nama model yang tersedia:

```text
vgg16 resnet50 mobilenetv2 efficientnetb0 xception
```

## Arti steps multiplier

Default `--steps-multiplier 1` berarti setiap epoch menjalankan jumlah batch
yang setara dengan 1.216 citra training. Gambar tetap diaugmentasi secara acak
pada setiap pengambilan batch.

Jika ingin meniru klaim 1.216 citra menjadi sekitar 6.080 paparan augmentasi per
epoch, tambahkan:

```text
--steps-multiplier 5
```

Pilihan tersebut membuat waktu training sekitar lima kali lebih lama. File fisik
tetap berjumlah 1.216 karena augmentasi dilakukan on-the-fly.

## Hasil yang dibuat

Folder output berisi:

- `dataset_audit.json`: bukti jumlah data dan tidak adanya kebocoran responden.
- `run_config.json`: konfigurasi eksperimen.
- `model_comparison.csv`: perbandingan seluruh model.
- Model terbaik format `.keras`.
- Riwayat training format CSV dan grafik PNG.
- Accuracy, precision, recall, dan F1-score.
- Classification report.
- Confusion matrix CSV dan PNG.
- Prediksi setiap gambar testing.
- Waktu feature extraction dan fine-tuning.

## Catatan metodologi

Skripsi memiliki konflik parameter. Bab III menulis batch size 32 dan learning
rate feature extraction 0,001. Bab IV menulis batch size 16 dan learning rate
0,00001. Paket ini memakai batch size 16, learning rate 0,001 untuk feature
extraction, dan 0,00001 untuk fine-tuning. Konfigurasi ini mempertahankan detail
implementasi Bab IV sekaligus membedakan learning rate kedua tahap secara aman.

Pembagian dataset yang sudah tersedia tidak diacak ulang. Hal ini menjaga
subject-wise split dan mencegah data leakage.
