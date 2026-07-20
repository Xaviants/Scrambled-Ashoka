<<<<<<< HEAD
# Scrambled-Ashoka
=======
# Ashoka - Scrambled AI

Repository ini berisi eksperimen klasifikasi citra untuk dataset Ashoka dengan beberapa skenario hold-out, termasuk model hybrid CNN + C-Swin, ViT dengan multi-patch selection dan Dilated SE-DenseNet.

## Tujuan

- Membangun model klasifikasi biner yang mampu membedakan citra buried penis dan normal   
  dengan akurasi tinggi
- Menentukan metode yang terbaik dalam menangani dataset Ashoka yang terbatas

## Struktur Repo

- `Ashoka_With_Hybrid_CNN_and_C_Swin.ipynb` — eksperimen hybrid CNN + C-Swin
- `DilatedSEDenseNet_WithHoldOut.ipynb` — eksperimen Dilated SE-DenseNet
- `holdout_patch16.ipynb` — hold-out patch 16
- `holdout_patch32.ipynb` — hold-out patch 32
- `holdout_patch8.ipynb` — hold-out patch 8
- `Dilated_SEDenseNet_model.py` — implementasi model Dilated SE-DenseNet
- `run_all_notebooks.py` — executor notebook
- `results/` — hasil run

## Persyaratan

- Python 3.10+ (disarankan 3.11)
- File `requirements.txt` saat ini menggunakan paket PyTorch dengan build CUDA 12.4. Jika 
  sistem Anda memakai versi CUDA lain, sesuaikan paket `torch`, `torchvision`, dan `torchaudio` agar sesuai.

## Setup Environment

### 1) Buat virtual environment

Contoh untuk Windows, macOS, dan Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Jika Anda menggunakan Windows PowerShell, perintah aktivasi bisa juga:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependensi

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes untuk `requirements.txt`:
- `torch`, `torchvision`, `torchaudio` — untuk model deep learning dan preprocessing citra
- `numpy` — komputasi numerik dan manipulasi array
- `matplotlib`, `seaborn` — visualisasi hasil eksperimen
- `scikit-learn` — metrik evaluasi seperti accuracy, confusion matrix, ROC-AUC
- `pillow` — membuka dan memproses citra gambar
- `nbformat`, `nbclient`, `jupyter_client`, `jupyter_core` — menjalankan notebook secara 
  terprogram lewat `run_all_notebooks.py`

## Menjalankan Eksperimen

### Cek daftar notebook yang akan dijalankan

```bash
python run_all_notebooks.py --dry-run
```

### Jalankan sekali

```bash
python run_all_notebooks.py --runs 1
```

### Jalankan dan lanjutkan dari run terakhir

```bash
python run_all_notebooks.py --resume --runs 1
```

### Jalankan dengan timeout tertentu

```bash
python run_all_notebooks.py --timeout 7200 --runs 1
```

Setiap run akan menyimpan hasil ke folder:

```text
results/run_01/<notebook_name>/executed_notebook.ipynb
```

## Notes

Agar hasil lebih konsisten:

- Pastikan Anda menggunakan environment yang sama.
- Jangan menghapus atau mengubah blok seed di notebook.
- Jalankan notebook dari direktori repo yang sama.
- Jika ingin membandingkan hasil antar run, gunakan `--run-index` atau simpan hasil dari 
  tiap run secara terpisah.
>>>>>>> 7656483 (Initial commit)
