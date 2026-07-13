import streamlit as st
import os
import cv2
import joblib
import numpy as np
import pywt
import scipy.stats
import warnings
import lightgbm as lgb
from PIL import Image

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="ColonoMind Evaluator", layout="wide", page_icon="\U0001f9ec")

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1d2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
}
.main-header h1 { color: white !important; font-size: 2rem; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85) !important; margin: 0.5rem 0 0; font-size: 0.95rem; }
.path-badge {
    background: #1e2a3a; border: 1px solid #2d4a6b; border-radius: 8px;
    padding: 0.6rem 1rem; font-family: monospace; font-size: 0.8rem;
    color: #58a6ff; margin-top: 0.5rem; word-break: break-all;
}
.result-box {
    background: #1a1d2e; border: 1px solid #2d4a6b;
    border-radius: 12px; padding: 1.5rem; margin-top: 1rem; text-align: center;
}
.label-mes0 { color: #2ea043; font-weight: 700; font-size: 2.5rem; }
.label-mes1 { color: #d4a017; font-weight: 700; font-size: 2.5rem; }
.label-mes2 { color: #fd7e14; font-weight: 700; font-size: 2.5rem; }
.label-mes3 { color: #f85149; font-weight: 700; font-size: 2.5rem; }
</style>
""", unsafe_allow_html=True)

DATASET_CHOICES = {
    "Intra_LIMUC":   "LIMUC (Intra)",
    "Intra_TMC-UCM": "TMC-UCM (Intra)",
    "Intra_NTUH":    "NTUH (Intra)",
}
MODEL_CHOICES = ["ResNet-50", "DenseNet-121", "EfficientNet-B4", "ConvNeXt-Tiny", "ViT-B-16"]
CLASS_NAMES   = ["MES0", "MES1", "MES2", "MES3"]
IMG_SIZE      = (224, 224)
WAVELET       = "db1"
BASE_DRIVE    = "./Result"

LABEL_CSS  = {"MES0": "label-mes0", "MES1": "label-mes1", "MES2": "label-mes2", "MES3": "label-mes3"}
LABEL_DESC = {
    "MES0": "Normal Mucosa \U0001f7e2",
    "MES1": "Mild Inflammation \U0001f7e1",
    "MES2": "Moderate Inflammation \U0001f7e0",
    "MES3": "Severe Inflammation \U0001f534",
}
FEAT_NAMES = [
    "LL_Mean","LL_Std","LL_Var","LL_Ent",
    "LH_Mean","LH_Std","LH_Var","LH_Ent",
    "HL_Mean","HL_Std","HL_Var","HL_Ent",
    "HH_Mean","HH_Std","HH_Var","HH_Ent","HH_Energy",
    "GLCM_Contrast","GLCM_Dissimilarity","GLCM_Homogeneity",
]

with st.sidebar:
    st.markdown("## \u2699\ufe0f Configuration")
    st.markdown("---")
    selected_dataset_key = st.selectbox(
        "\U0001f4c2 Dataset / Experiment",
        list(DATASET_CHOICES.keys()),
        format_func=lambda x: DATASET_CHOICES[x],
    )
    selected_model = st.selectbox("\U0001f916 Model", MODEL_CHOICES)
    model_path = f"{BASE_DRIVE}/{selected_dataset_key}/{selected_model}_Experiment"
    st.markdown("**Model Path:**")
    st.markdown(f"<div class='path-badge'>{model_path}</div>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded_file = st.file_uploader("\U0001f5bc\ufe0f Upload Image (PNG/JPG)", type=["png","jpg","jpeg"])
    st.markdown("---")
    st.caption("ColonoMind Super Agent  |  Hybrid LightGBM Pipeline")

@st.cache_resource
def load_model_components(dataset_key, model_name):
    base_dir    = f"{BASE_DRIVE}/{dataset_key}/{model_name}_Experiment"
    scaler_path = os.path.join(base_dir, f"{model_name}_scaler.pkl")
    agent_path  = os.path.join(base_dir, f"{model_name}_agent.txt")
    missing = []
    if not os.path.exists(scaler_path): missing.append(scaler_path)
    if not os.path.exists(agent_path):  missing.append(agent_path)
    if missing:
        return None, "File tidak ditemukan:\n" + "\n".join(missing)
    try:
        return {"scaler": joblib.load(scaler_path), "agent": lgb.Booster(model_file=agent_path)}, None
    except Exception as e:
        return None, str(e)

def extract_features(img_rgb):
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    coeffs2 = pywt.dwt2(gray, WAVELET)
    LL, (LH, HL, HH) = coeffs2
    def stats(sb):
        return [np.mean(sb), np.std(sb), np.var(sb),
                scipy.stats.entropy(np.abs(sb.flatten()) + 1e-6)]
    feats  = stats(LL) + stats(LH) + stats(HL) + stats(HH)
    feats += [np.sum(np.square(HH)) / HH.size]
    glcm   = graycomatrix(gray, distances=[1,3,5],
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=256, symmetric=True, normed=True)
    feats += [
        np.mean(graycoprops(glcm, "contrast")),
        np.mean(graycoprops(glcm, "dissimilarity")),
        np.mean(graycoprops(glcm, "homogeneity")),
    ]
    return feats

st.markdown("""
<div class="main-header">
  <h1>\U0001f9ec ColonoMind &mdash; Super Agent Evaluator</h1>
  <p>Hybrid Wavelet &middot; GLCM &middot; LightGBM Pipeline &nbsp;|&nbsp; Multi-Dataset Comparison</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("\U0001f448 Pilih Dataset, Model, lalu Upload Gambar dari sidebar untuk memulai analisis.")
else:
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("\U0001f5bc\ufe0f Gambar Input")
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, use_container_width=True)
        st.caption(f"Ukuran asli: {pil_img.size[0]}x{pil_img.size[1]} px -> di-resize 224x224")

    with col_res:
        st.subheader("\U0001f4ca Hasil Analisis")
        with st.spinner(f"Memuat {selected_model} dari {DATASET_CHOICES[selected_dataset_key]}..."):
            components, err = load_model_components(selected_dataset_key, selected_model)

        if err:
            st.error(f"Gagal memuat model:\n```\n{err}\n```")
        else:
            st.success(f"Model **{selected_model}** berhasil dimuat dari `{selected_dataset_key}`")
            with st.spinner("Mengekstrak fitur & menjalankan prediksi..."):
                img_arr     = np.array(pil_img)
                img_resized = cv2.resize(img_arr, IMG_SIZE)
                raw_feats   = extract_features(img_resized)
                scaled      = components["scaler"].transform(np.array(raw_feats).reshape(1, -1))
                agent_input = np.hstack([[[0.5]], [[0.0]], [[0.0]], scaled])
                proba       = components["agent"].predict(agent_input)[0]
                if hasattr(proba, "__len__") and len(proba) > 1:
                    conf      = float(np.max(proba))
                    label_idx = int(np.argmax(proba))
                else:
                    label_idx = int(proba)
                    conf      = 1.0

            label_str = CLASS_NAMES[label_idx]
            is_ref    = conf < 0.70

            st.markdown(f"""
<div class="result-box">
  <div class="{LABEL_CSS[label_str]}">{label_str}</div>
  <div style="color:#aaa; margin-top:0.3rem;">{LABEL_DESC[label_str]}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("")
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{conf*100:.1f}%")
            c2.metric("Referral Needed", "Yes" if is_ref else "No")
            if is_ref:
                st.warning("Ketidakpastian terdeteksi - disarankan rujuk ke dokter.")
            else:
                st.success("Confidence tinggi - hasil dapat diandalkan.")
            with st.expander("Lihat Semua Fitur (20 fitur Wavelet + GLCM)"):
                feat_dict = {FEAT_NAMES[i]: round(float(raw_feats[i]), 6) for i in range(len(FEAT_NAMES))}
                st.dataframe(feat_dict, use_container_width=True)
