# seventx_prototype: Penilai Kualitas Stroberi

Prototipe hackathon: model klasifikasi gambar kecil yang menilai foto **satu buah stroberi** sebagai **baik** (`good`) atau **rusak** (`damaged`). Model ini cukup kecil untuk berjalan **offline di ponsel**. Sasaran penggunanya adalah petani stroberi (misalnya di Rancabali), supaya penyortiran pascapanen lebih cepat dan konsisten.

Folder ini berisi seluruh alur kerja **model**: penyiapan data, pelatihan, evaluasi, dan ekspor ke TFLite. Aplikasi Android (Flutter) belum dibuat; rancangannya ada di [`plans/04_app.md`](plans/04_app.md).

## Hasil singkat

| Hal | Hasil |
| --- | --- |
| Data yang dipakai | 1.123 foto (574 baik, 549 rusak) dari dua dataset publik |
| Model terpilih | **EfficientNet-B0** (pra-latih ImageNet), diekspor sebagai `strawberry_2class.tflite`, float16, **8,1 MB** |
| Akurasi pada data uji (161 foto) | **96,9%**; 81 dari 84 stroberi rusak terdeteksi (ambang 0,261) |
| Perbandingan model (validasi) | EfficientNet-B0 92,1%, MobileNetV3-Small 89,7%, CNN dilatih dari nol 72,7% |
| Uji lintas dataset (latih di satu dataset, uji di dataset lain) | **63–75%**: akurasi turun pada kamera atau kondisi foto yang berbeda |

Rincian lengkap ada di [`MODEL_PLAN.md`](MODEL_PLAN.md) dan di folder [`plans/`](plans/).

## Struktur folder

```
seventx_prototype/
  README.md                       berkas ini
  MODEL_PLAN.md                   rencana utama dan keputusan (sumber kebenaran)
  CLAUDE.md                       catatan untuk asisten AI (Claude Code)
  plans/
    01_data_and_splits.md         data, pembagian train/val/test, eksperimen
    02_train_eval.md              pelatihan, evaluasi, log tuning, hasil
    03_export.md                  ekspor TFLite, ambang, test vector
    04_app.md                     rancangan aplikasi Flutter (belum dibuat)
  notebooks/
    01_manifest_and_split.ipynb   langkah 1: bersihkan data dan bagi train/val/test
    02_train_eval.ipynb           langkah 2: latih dan evaluasi 3 model
    03_export.ipynb               langkah 3: ekspor ke TFLite untuk aplikasi
    lib/preprocess.py, .js        crop ke buah + ubah ukuran (dipakai pelatihan dan aplikasi)
    build/make_nb0X.py            skrip pembuat notebook
  data/
    manifest.csv                  daftar foto yang dipakai beserta split-nya (WAJIB, jangan diubah manual)
    excluded.csv                  foto yang dibuang beserta alasannya
  papers/                         catatan dua makalah rujukan (Sustika 2018, Mahendra 2018)
  models/export/                  hasil ekspor untuk aplikasi (diunduh dari Google Drive)
  dataset/, dataset.zip           foto mentah (TIDAK ada di GitHub, lihat langkah 1)
```

## Alur kerja

```
dataset/ (foto mentah)
   │  notebook 01: buang salinan, kelompokkan foto berulang, bagi 70/15/15
   ▼
data/manifest.csv
   │  notebook 02: latih EfficientNet-B0, MobileNetV3-Small, CNN dasar
   ▼
models/crop_lr1e-4/*.keras + results/crop_lr1e-4/ (di Google Drive)
   │  notebook 03: ubah ke TFLite, cek akurasi, tetapkan ambang
   ▼
models/export/ → aplikasi Flutter
```

Ada tiga aturan penting di seluruh alur:
- **Setiap foto di-crop ke buahnya** sebelum diubah ke 224×224, baik saat pelatihan maupun di aplikasi (`notebooks/lib/preprocess.py`).
- **Input model bernilai piksel 0–255.** Jangan dibagi 255.
- **Semua keputusan** (model, pengaturan, ambang) **diambil dari data validasi.** Data uji hanya dilihat sekali, di akhir.

## Cara menjalankan di komputer Anda

### Yang dibutuhkan

- **Python 3.10+** dengan `pillow`, `numpy`, `nbformat` dan `nbconvert`, untuk notebook 01, yang bisa jalan di laptop:
  ```
  pip install pillow numpy nbformat nbconvert ipykernel
  ```
- **Akun Google** dengan Google Drive dan **Google Colab**, untuk notebook 02 dan 03. Keduanya butuh TensorFlow, dan notebook 02 butuh GPU. Colab sudah menyediakan TensorFlow, jadi **jangan** `pip install tensorflow` di Colab.
- **Opsional:** **VS Code** dengan ekstensi **Google Colab**. Dengan ini, notebook tetap berada di laptop dan hanya kernelnya (beserta GPU) yang memakai Colab. Membuka notebook langsung di colab.research.google.com juga bisa.
- **Opsional:** **Node.js**, untuk pengecekan `preprocess.js` di notebook 03.

### Langkah 1: siapkan data

Foto mentah tidak ikut di GitHub karena terlalu besar. Unduh dari sumber aslinya, lalu susun seperti ini:

```
dataset/
  color/good/              382 foto, dari Mahendra 2018
  color/damaged/           350 foto
  FreshStrawberry/         737 foto, dari Mendeley "Fresh and Rotten Fruits"
  RottenStrawberry/        737 foto
```

| Dataset | Unduh dari | Lisensi |
| --- | --- | --- |
| Mahendra et al. 2018 | https://github.com/okamahendra123/dataset_strawberry (folder `color/`) | GPL |
| Fresh and Rotten Fruits | https://data.mendeley.com/datasets/bdd69gyhv8/1 (ambil hanya folder `FreshStrawberry` dan `RottenStrawberry`) | CC BY 4.0 |

Lalu buat `dataset.zip` untuk diunggah ke Drive. Jalankan di folder proyek:

```
python -c "import zipfile, pathlib; z = zipfile.ZipFile('dataset.zip', 'w'); [z.write(p, p.as_posix()) for p in sorted(pathlib.Path('dataset').rglob('*.jpg'))]; z.close()"
```

### Langkah 2: notebook 01, data dan pembagian (bisa di laptop)

`data/manifest.csv` **sudah ada di repositori**, jadi langkah ini hanya perlu diulang kalau datanya berubah. Notebook ini:
- membuang 1.074 salinan hasil augmentasi (file Mendeley nomor 201–737);
- mengelompokkan foto berulang dari buah yang sama, supaya tidak bocor antara train dan test;
- membagi data 70/15/15 per kelompok.

```
cd notebooks
python -m nbconvert --to notebook --execute --inplace 01_manifest_and_split.ipynb
```

Hasilnya: `data/manifest.csv` (1.123 baris) dan `data/excluded.csv`. Semua pemeriksaan di notebook harus lulus ("all checks passed").

### Langkah 3: siapkan Google Drive

Buat folder `seventx_prototype` di **My Drive**, lalu unggah:

```
My Drive/seventx_prototype/
  dataset.zip
  data/manifest.csv
```

Notebook tidak perlu diunggah kalau Anda memakai VS Code dengan ekstensi Colab. Kalau memakai Colab di browser, unggah juga folder `notebooks/`. Notebook mencari Drive di `/content/drive/MyDrive/seventx_prototype`; ubah baris `PROJECT = ...` di sel pertama kalau Anda memakai nama atau lokasi lain.

### Langkah 4: notebook 02, latih dan evaluasi (Colab + GPU)

Pilih runtime **GPU**, misalnya T4. Jalankan beberapa kali dengan pengaturan berbeda di sel pertama:

| Urutan | `SMOKE` | `FIRST_ONLY` | `SHOW_TEST` | Gunanya |
| --- | --- | --- | --- | --- |
| 1 | `True` | (diabaikan) | `False` | Uji cepat bahwa semua jalan (skornya tidak bermakna) |
| 2 | `False` | `True` | `False` | Satu model (MobileNetV3, E1, seed 42); lihat skor validasi saja |
| 3 | `False` | `False` | `False` | Semua 15 run: 3 model × (E1 dengan 3 seed + E2a + E2b) |
| 4 | `False` | `False` | `True` | Sekali saja, setelah model dipilih: tampilkan skor uji |

- Pengaturan terakhir yang dipakai: `CROP_TO_BERRY = True`, `RUN_TAG = 'lr1e-4'`. Hasilnya tersimpan di Drive, di `models/crop_lr1e-4/` dan `results/crop_lr1e-4/`.
- **Kalau koneksi atau Colab terputus,** jalankan ulang semua sel. Run yang sudah selesai akan dilewati, dan run yang terpotong diulang dari awal tanpa memengaruhi hasil.
- **Kalau Anda mengubah pengaturan pelatihan,** ganti juga `RUN_TAG`, supaya hasil lama tidak tertimpa.

### Langkah 5: notebook 03, ekspor untuk aplikasi (Colab, CPU cukup)

Tidak perlu GPU. Jalankan semua sel; tidak ada pengaturan yang perlu diubah. Notebook ini:
- memastikan crop hasil ekspor sama persis dengan crop saat pelatihan;
- mengubah EfficientNet-B0 ke 4 varian TFLite, lalu memakai **float16** (`SHIP_VARIANT`);
- menetapkan **ambang dari data validasi** (≥95% stroberi rusak terdeteksi);
- menulis 10 foto uji (*test vector*), `metadata.json` dan `labels.txt`.

Hasilnya ada di Drive, di `models/export/`. **Unduh folder itu ke `models/export/` di proyek ini.** Folder `tflite_variants/` di dalamnya tidak perlu diunduh.

### Langkah 6: aplikasi

Belum dibuat. Rancangan aplikasi Flutter (Android, offline, bahasa Indonesia dan Inggris) ada di [`plans/04_app.md`](plans/04_app.md). Yang dibutuhkan aplikasi dari `models/export/`:

| Berkas | Fungsi |
| --- | --- |
| `strawberry_2class.tflite` | modelnya |
| `metadata.json` | ambang (0,261), aturan input, skor, dan kredit dataset |
| `labels.txt` | `good`, `damaged` |
| `preprocess.py` / `preprocess.js` | crop rujukan, yang akan di-*port* ke Dart |
| `test_vectors/` | foto uji untuk memastikan aplikasi memproses foto dengan benar |

Ada tiga hal yang wajib di aplikasi: **crop ke buah**, **jangan bagi 255**, dan **ambil ambang dari `metadata.json`**. Stroberi dinilai rusak kalau P ≥ 0,261, bukan 0,5.

## Mengubah kode crop atau notebook 03

`notebooks/lib/preprocess.py` adalah rujukan crop, dan notebook 03 menyalin isinya. Setelah mengubahnya, buat ulang notebook 03:

```
python notebooks/build/make_nb03.py notebooks/03_export.ipynb
```

Notebook 01 dan 02 sudah disunting langsung, jadi **jangan** membuatnya ulang dari `notebooks/build/`: keluaran run yang tersimpan di dalamnya akan hilang.

## Keterbatasan

- **Satu buah per foto.** Menilai satu nampan dari satu foto belum didukung.
- **Hanya 2 kelas,** baik dan rusak. Belum ada "grade kedua".
- **Hanya permukaan buah.** Kerusakan di dalam buah tidak terlihat kamera.
- **Dilatih pada dua dataset publik,** tanpa foto dari Rancabali. Uji lintas dataset menunjukkan akurasi turun pada kondisi foto baru.

## Lisensi dan kredit data

- **Fresh and Rotten Fruits Dataset,** Mendeley Data, DOI 10.17632/bdd69gyhv8.1, CC BY 4.0.
- **Mahendra et al. 2018,** dataset stroberi (GPL), https://github.com/okamahendra123/dataset_strawberry. Foto-foto ini **tidak boleh** dibundel di dalam aplikasi.
- Makalah rujukan: Sustika et al. 2018 dan Mahendra et al. 2018 (catatannya di `papers/`). Angka akurasi mereka hanya pembanding, bukan perbandingan langsung, karena data dan metodenya berbeda.
