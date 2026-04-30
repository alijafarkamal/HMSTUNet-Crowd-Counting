import os
import shutil
import tempfile
import urllib.request
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import cv2

from model import HMSTUNet

# ──────────────────────────────────────────────
#  Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HMSTUNet Crowd Counter",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
}

.metric-badge {
    background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(96,165,250,0.2));
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 0.4rem;
}
.metric-badge .label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.metric-badge .value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1.2;
}
.metric-badge .unit {
    font-size: 0.85rem;
    color: #64748b;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.info-box {
    background: rgba(96,165,250,0.1);
    border-left: 3px solid #60a5fa;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    color: #bfdbfe;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Constants & helpers
# ──────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best.pth")


def _nonempty_str(value):
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _checkpoint_download_url():
    u = _nonempty_str(os.environ.get("CHECKPOINT_URL"))
    if u:
        return u
    try:
        sec = st.secrets
    except Exception:
        return None
    getter = getattr(sec, "get", None)
    if callable(getter):
        u = _nonempty_str(getter("CHECKPOINT_URL"))
        if u:
            return u
    try:
        u = _nonempty_str(sec["CHECKPOINT_URL"])
        if u:
            return u
    except (KeyError, TypeError):
        pass
    try:
        return _nonempty_str(sec["checkpoint"]["url"])
    except (KeyError, TypeError):
        return None


def ensure_checkpoint_file():
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.is_file():
        return CHECKPOINT_PATH
    url = _checkpoint_download_url()
    if not url:
        raise FileNotFoundError(
            "Missing checkpoints/best.pth. Either add the file under checkpoints/, or set "
            "CHECKPOINT_URL in Streamlit app secrets / environment to a direct download link."
        )
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; HMSTUNet/1.0)"}
    )
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=CHECKPOINT_PATH.parent, suffix=".part"
        ) as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(req, timeout=600) as resp:
                shutil.copyfileobj(resp, tmp)
            tmp.flush()
        tmp_path.replace(CHECKPOINT_PATH)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return CHECKPOINT_PATH


@st.cache_resource
def load_model():
    path = ensure_checkpoint_file()
    model = HMSTUNet(pretrained=False)
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_image(image):
    image = image.convert("RGB")
    w, h = image.size
    new_w = max(32, (w // 32) * 32)
    new_h = max(32, (h // 32) * 32)
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return image, transform(image).unsqueeze(0)


def density_to_heatmap(density_map: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    dm_norm = density_map / (density_map.max() + 1e-5)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * dm_norm), cmap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(orig_pil: Image.Image, heatmap_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    orig_np = np.array(orig_pil.convert("RGB"))
    hm_resized = cv2.resize(heatmap_rgb, (orig_np.shape[1], orig_np.shape[0]))
    blended = cv2.addWeighted(orig_np, 1 - alpha, hm_resized, alpha, 0)
    return blended


# ──────────────────────────────────────────────
#  Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>👥 HMSTUNet Crowd Counter</h1>
    <p>AI-powered crowd density estimation</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Load model
# ──────────────────────────────────────────────
try:
    with st.spinner("Loading model weights…"):
        model = load_model()
except Exception as e:
    st.error(
        f"**Failed to load model.** Add `checkpoints/best.pth` locally, or set "
        f"**CHECKPOINT_URL** in Streamlit Secrets to a direct `.pth` download link.\n\n`{e}`"
    )
    st.stop()

# ──────────────────────────────────────────────
#  File uploader
# ──────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📂 Upload a crowd image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload any image containing people to estimate the crowd count.",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        ℹ️ Upload an image above to start. The model will generate a full density map and
        estimate the total crowd count.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
#  Run inference (cached in session state)
# ──────────────────────────────────────────────
file_id = uploaded_file.file_id

if st.session_state.get("file_id") != file_id:
    image_pil = Image.open(uploaded_file)
    with st.spinner("🧠 Running HMSTUNet inference…"):
        orig_img, img_tensor = preprocess_image(image_pil)
        with torch.no_grad():
            dm = model(img_tensor)
        density_map = dm.squeeze().cpu().numpy()
        total_count = float(np.sum(density_map))

    st.session_state.file_id = file_id
    st.session_state.orig_img = orig_img
    st.session_state.density_map = density_map
    st.session_state.total_count = total_count

orig_img: Image.Image = st.session_state.orig_img
density_map: np.ndarray = st.session_state.density_map
total_count: float = st.session_state.total_count

img_w, img_h = orig_img.size

# ──────────────────────────────────────────────
#  Full Image Analysis
# ──────────────────────────────────────────────
st.markdown("""<div class="section-title">📊 Overall Crowd Metrics</div>""", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="metric-badge">
        <div class="label">Total Crowd Count</div>
        <div class="value">{int(total_count):,}</div>
        <div class="unit">people (estimated)</div>
    </div>""", unsafe_allow_html=True)
with col_m2:
    area_px = img_w * img_h
    density_per_100px = total_count / (area_px / 10_000)
    st.markdown(f"""
    <div class="metric-badge">
        <div class="label">Density per 100×100 px</div>
        <div class="value">{density_per_100px:.1f}</div>
        <div class="unit">people / 10k pixels</div>
    </div>""", unsafe_allow_html=True)
with col_m3:
    peak_val = float(density_map.max())
    st.markdown(f"""
    <div class="metric-badge">
        <div class="label">Peak Density Value</div>
        <div class="value">{peak_val:.3f}</div>
        <div class="unit">max cell activation</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">🖼️ Visualization</div>', unsafe_allow_html=True)

cmap_options = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "PLASMA": cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS
}
selected_cmap_name = st.radio(
    "🎨 Select Colormap",
    options=list(cmap_options.keys()),
    index=0,
    horizontal=True
)

heatmap_rgb = density_to_heatmap(density_map, cmap=cmap_options[selected_cmap_name])
overlay_rgb = overlay_heatmap(orig_img, heatmap_rgb, alpha=0.60)

col_img, col_hm, col_ov = st.columns(3)
with col_img:
    st.image(orig_img, caption="Original Image", use_container_width=True)
with col_hm:
    st.image(
        cv2.resize(heatmap_rgb, (img_w, img_h)),
        caption="Density Heatmap",
        use_container_width=True,
    )
with col_ov:
    st.image(overlay_rgb, caption="Heatmap Overlay", use_container_width=True)