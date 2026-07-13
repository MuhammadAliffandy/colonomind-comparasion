# 🧬 ColonoMind — Super Agent Evaluator

Aplikasi evaluasi gambar kolonoskopi berbasis **Streamlit** yang menggunakan pipeline *Hybrid Super Agent* (Wavelet + GLCM + LightGBM). Aplikasi ini mendukung pemilihan dataset dan model eksperimen secara dinamis untuk keperluan perbandingan performa antar eksperimen.

---

## 🏗️ Arsitektur Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Feature Extraction (Handcrafted)               │
│  ├── Wavelet Decomposition (db1)                │
│  │     LL, LH, HL, HH → Mean, Std, Var, Entropy│
│  └── GLCM (distances=[1,3,5], angles=4)         │
│        Contrast, Dissimilarity, Homogeneity      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Scaler (StandardScaler via joblib .pkl)        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  LightGBM Super Agent (.txt)                    │
│  Output: MES0 / MES1 / MES2 / MES3             │
└─────────────────────────────────────────────────┘
```

---

## 📁 Struktur Direktori

```
ColonomindComparasionWeb/
├── app.py                          # ← Entrypoint utama Streamlit
├── requirements.txt                # ← Semua dependencies Python
├── .gitignore
├── Result/                         # ← File model (TIDAK di-push ke git)
│   ├── Intra_LIMUC/
│   │   ├── ResNet-50_Experiment/
│   │   │   ├── ResNet-50_scaler.pkl
│   │   │   └── ResNet-50_agent.txt
│   │   ├── DenseNet-121_Experiment/
│   │   ├── EfficientNet-B4_Experiment/
│   │   ├── ConvNeXt-Tiny_Experiment/
│   │   └── ViT-B-16_Experiment/
│   ├── Intra_TMC-UCM/
│   │   └── (struktur sama)
│   └── Intra_NTUH/
│       └── (struktur sama)
└── ColonoMind_Comparison_.ipynb    # ← Notebook analisis (legacy)
```

> **⚠️ Penting:** Folder `Result/` berisi file model biner yang besar dan **tidak masuk ke Git**.  
> Saat deploy ke server baru, salin folder `Result/` secara manual ke root direktori project.

---

## 🔧 Dataset & Model yang Didukung

| Dataset Key       | Nama Tampil       |
|-------------------|-------------------|
| `Intra_LIMUC`     | LIMUC (Intra)     |
| `Intra_TMC-UCM`   | TMC-UCM (Intra)   |
| `Intra_NTUH`      | NTUH (Intra)      |

| Model Backbone  |
|-----------------|
| ResNet-50       |
| DenseNet-121    |
| EfficientNet-B4 |
| ConvNeXt-Tiny   |
| ViT-B-16        |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/MuhammadAliffandy/colonomind-comparasion.git
cd colonomind-comparasion
```

### 2. Salin Folder Model (manual)
Pastikan folder `Result/` dengan semua file `.pkl` dan `_agent.txt` sudah tersedia di direktori root project.

```
Result/
  ├── Intra_LIMUC/
  ├── Intra_TMC-UCM/
  └── Intra_NTUH/
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Streamlit
```bash
streamlit run app.py
```

Buka browser dan akses: **`http://localhost:8501`**  
Atau untuk akses dari jaringan lain: **`http://<IP_SERVER>:8501`**

---

## ⚙️ Konvensi Penamaan File Model

Aplikasi membaca model dari path berikut secara dinamis:

```
./Result/{DATASET_KEY}/{MODEL_NAME}_Experiment/{MODEL_NAME}_scaler.pkl
./Result/{DATASET_KEY}/{MODEL_NAME}_Experiment/{MODEL_NAME}_agent.txt
```

**Contoh:**
```
./Result/Intra_LIMUC/DenseNet-121_Experiment/DenseNet-121_scaler.pkl
./Result/Intra_LIMUC/DenseNet-121_Experiment/DenseNet-121_agent.txt
```

---

## 🏷️ Label Kelas (Mayo Endoscopic Score)

| Label  | Deskripsi                  | Warna  |
|--------|----------------------------|--------|
| `MES0` | Normal Mucosa              | 🟢 Hijau  |
| `MES1` | Mild Inflammation          | 🟡 Kuning |
| `MES2` | Moderate Inflammation      | 🟠 Oranye |
| `MES3` | Severe Inflammation        | 🔴 Merah  |

> Jika confidence prediksi **< 70%**, sistem akan memberikan flag **"Referral Needed"**  
> dan merekomendasikan rujukan ke dokter.

---

## 📦 Tech Stack

- **UI:** Streamlit ≥ 1.30
- **ML Agent:** LightGBM
- **Feature Extraction:** PyWavelets, scikit-image (GLCM)
- **Image Processing:** OpenCV (headless), Pillow
- **Scaler:** scikit-learn (StandardScaler via joblib)
