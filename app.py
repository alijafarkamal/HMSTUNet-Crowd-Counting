import os
import shutil
import tempfile
import urllib.request
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --primary: #22d3ee;
    --primary-glow: rgba(34, 211, 238, 0.3);
    --accent: #0ea5e9;
    --bg-dark: #061d23;
    --card-bg: rgba(8, 51, 68, 0.6);
    --border: rgba(34, 211, 238, 0.2);
    /* These now use Streamlit's native variable so they auto-adapt to any theme */
    --text-main: var(--text-color, #f8fafc);
    --text-dim: var(--text-color, #94a3b8);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .stHeader {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    color: var(--text-main);
}
@media (prefers-color-scheme: dark) {
    .stApp {
        background: radial-gradient(circle at 50% 0%, #083344 0%, #061d23 100%);
    }
}

.hero-header {
    text-align: center;
    padding: 3rem 1rem 1rem;
}
.hero-header h1 {
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--primary);
    margin-bottom: 0.5rem;
}
.hero-header p {
    color: var(--text-dim);
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

/* Stepper Component */
.stepper-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 5rem;
    margin: 3rem 0;
    position: relative;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}
.stepper-line {
    position: absolute;
    top: 20px;
    left: 10%;
    right: 10%;
    height: 2px;
    background: var(--border);
    z-index: 0;
}
.step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 1;
    opacity: 0.5;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.step.active {
    opacity: 1;
    transform: translateY(-2px);
}
.step-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: var(--bg-dark);
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-bottom: 0.8rem;
    color: var(--text-dim);
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
}
.active .step-circle {
    background: var(--primary);
    border-color: var(--primary);
    color: #000;
    box-shadow: 0 0 25px var(--primary-glow);
}
.step-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.active .step-label {
    color: var(--primary);
    text-shadow: 0 0 10px var(--primary-glow);
}

[data-testid="stFileUploader"] {
    background: var(--card-bg);
    border: 2px dashed var(--border);
    border-radius: 24px;
    padding: 3rem 2rem 2rem;
    backdrop-filter: blur(20px);
    margin: 0.5rem 0 0.5rem;
    text-align: center;
    position: relative;
}
.info-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: var(--text-dim);
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 0;
    backdrop-filter: blur(10px);
}
.info-card b {
    color: var(--text-color);
    font-weight: 600;
}
.uploader-header {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--primary);
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    padding-left: 0.5rem;
    letter-spacing: 0.01em;
}
.uploader-header span {
    font-size: 1.3rem;
    filter: drop-shadow(0 0 8px var(--primary-glow));
}
.help-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border: 1.5px solid var(--text-dim);
    border-radius: 50%;
    font-size: 11px;
    color: var(--text-dim);
    cursor: help;
    margin-left: 8px;
    transition: all 0.3s ease;
}
.help-icon:hover {
    border-color: var(--primary);
    color: var(--primary);
    box-shadow: 0 0 8px var(--primary-glow);
}
.tooltip {
    position: relative;
    display: inline-block;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 220px;
    background-color: #ffffff;
    color: #1e293b;
    text-align: center;
    border-radius: 8px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1000;
    bottom: 150%;
    left: 50%;
    margin-left: -110px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 12px;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    line-height: 1.4;
}
.tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #ffffff transparent transparent transparent;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    padding: 3rem !important;
    margin-top: 1rem;
}
/* Style for uploaded file chips */
[data-testid="stFileUploaderUploadedFiles"] {
    padding-top: 1.5rem;
}
[data-testid="stUploadedFile"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stUploadedFile"] > div {
    color: var(--text-color) !important;
}
[data-testid="stFileUploaderDeleteBtn"], [data-testid="stFileUploaderAddBtn"] {
    background-color: var(--primary) !important;
    color: #000 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
/* Glassmorphism Table Styling */
.glass-table-container {
    max-height: 400px;
    overflow-y: auto;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(8, 51, 68, 0.4);
}
.glass-table {
    width: 100%;
    border-collapse: collapse;
    color: var(--text-main);
    font-size: 0.95rem;
}
.glass-table th {
    background: rgba(34, 211, 238, 0.15);
    color: var(--primary);
    font-weight: 600;
    text-align: left;
    padding: 12px 16px;
    position: sticky;
    top: 0;
    backdrop-filter: blur(8px);
    z-index: 1;
    border-bottom: 2px solid var(--primary);
}
.glass-table td {
    padding: 10px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s ease;
}
.glass-table tbody tr:hover td {
    background: rgba(34, 211, 238, 0.1);
    color: var(--text-color);
}
.glass-table tbody tr:first-child td {
    background: rgba(34, 211, 238, 0.2);
    font-weight: 700;
    color: var(--text-color);
    border-left: 3px solid var(--primary);
}
.glass-table-container::-webkit-scrollbar {
    width: 6px;
}
.glass-table-container::-webkit-scrollbar-track {
    background: transparent;
}
.glass-table-container::-webkit-scrollbar-thumb {
    background: var(--primary);
    border-radius: 10px;
}

[data-testid="stFileUploaderDeleteBtn"]:hover {
    background-color: #ef4444 !important; /* Red on hover for delete */
    color: #fff !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--primary) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.1);
    box-shadow: none !important;
}
.uploader-desc {
    color: var(--text-dim);
    font-size: 1.05rem;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
    text-align: center;
}

/* Override Streamlit Info Box */
div[data-testid="stNotification"] {
    background: rgba(8, 51, 68, 0.8) !important;
    border: 1px solid var(--border) !important;
    border-left: 5px solid var(--primary) !important;
    color: var(--text-main) !important;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border) !important;
    padding: 0 24px;
    color: var(--text-color) !important;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: #000 !important; /* Fixed contrast */
    font-weight: 700 !important;
    border: 3px solid #ffffff !important; /* Thick White boundary */
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
}

/* Hide default red underline */
div[data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* Analysis Section Styling */
.metric-badge {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease;
    margin-bottom: 1rem;
}
.metric-badge:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.06);
    border-color: var(--primary);
}
.metric-badge .label {
    color: var(--text-dim);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}
.metric-badge .value {
    color: var(--primary);
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-badge .unit {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-top: 0.4rem;
}
.section-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-color);
    margin: 3rem 0 1.5rem;
    padding: 0.5rem 1rem;
    border-left: 4px solid var(--primary);
    background: linear-gradient(90deg, rgba(34, 211, 238, 0.1) 0%, transparent 100%);
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.01em;
    border-radius: 0 8px 8px 0;
}
.info-box {
    background: rgba(34, 211, 238, 0.05);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    color: var(--text-main);
    font-size: 0.95rem;
    line-height: 1.5;
}
/* Custom Number Input Styling */
[data-testid="stNumberInput"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important; /* Increased distance from borders */
    margin: 1.2rem 0 !important;   /* Added spacing around the component */
}
[data-testid="stNumberInput"] label {
    color: var(--text-color) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stNumberInput"] input {
    color: var(--text-color) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
    background-color: rgba(34, 211, 238, 0.1) !important;
    color: var(--primary) !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}
[data-testid="stNumberInputStepUp"] {
    margin-left: 8px !important; /* Added gap between buttons */
}
[data-testid="stNumberInputStepDown"]:hover {
    background-color: #ef4444 !important; /* Red for minus */
    color: #fff !important;
}
[data-testid="stNumberInputStepUp"]:hover {
    background-color: #22c55e !important; /* Green for plus */
    color: #fff !important;
}

/* Custom Slider Thumb Styling */
[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 10px !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #3b82f6 !important; /* Blue center */
    border: 3px solid #ffffff !important; /* Thick white ring */
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.6) !important; /* Subtle glow */
    top: 2px !important; /* Center on track */
    color: transparent !important; /* Hide text on thumb */
    transition: box-shadow 0.2s ease !important;
}

[data-testid="stSlider"] [role="slider"]:hover,
[data-testid="stSlider"] [role="slider"]:active {
    box-shadow: 0 0 16px rgba(59, 130, 246, 1.0) !important;
}
[data-testid="stSliderTickBar"],
[data-testid="stThumbValue"] {
    display: none !important;
}
/* ── Colormap Segmented Control Fix (Matches Tabs) ── */

/* Container - make it transparent to show separate buttons */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"],
[data-testid="stSegmentedControl"] [role="group"],
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    background-color: transparent !important;
    background: transparent !important;
    gap: 12px !important; /* Gap like tabs */
    display: flex !important;
}

/* Individual Buttons - Unselected */
[data-testid="stSegmentedControl"] button {
    height: 48px !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    padding: 0 24px !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
    margin: 0 !important;
}

/* Fix for inner button layout to prevent shifting */
[data-testid="stSegmentedControl"] button > div {
    background: transparent !important;
}

/* Selected button - Cyan background, Black text, White border (Matches Tabs) */
[data-testid="stSegmentedControl"] button[aria-selected="true"],
[data-testid="stSegmentedControl"] button[data-selected="true"] {
    background: var(--primary) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.2) !important;
}

[data-testid="stSegmentedControl"] button[aria-selected="true"] *,
[data-testid="stSegmentedControl"] button[data-selected="true"] * {
    color: #000000 !important;
    background: transparent !important;
}

/* Colormap Label - White */
[data-testid="stSegmentedControl"] label p,
[data-testid="stSegmentedControl"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    opacity: 1 !important;
    background: transparent !important;
}

/* ══════════════════════════════════════════════
   UNIVERSAL VISIBILITY — THEME SAFE
   ══════════════════════════════════════════════ */

/* ALL widget labels (slider, number input, text input) */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stNumberInput"] label p,
[data-testid="stTextInput"] label p,
[data-testid="stSlider"] label p {
    color: var(--text-color) !important;
}

/* ALL input values */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    color: var(--text-color) !important;
}

/* Expander headers (ROI 1, ROI 2 etc.) - Force Dark Background & White Text */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stExpander"] summary {
    background-color: #0d3040 !important; /* Dark Teal */
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: #164e63 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
    color: #ffffff !important;
    font-weight: 800 !important; /* Bolder */
}

/* ALL widget labels (slider, number input, text input, zone name, x/y coords) */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label,
label[data-testid="stWidgetLabel"] p,
.stSlider label p,
.stTextInput label p,
.stNumberInput label p {
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 800 !important; /* Bolder */
    font-size: 0.95rem !important;
}

/* Uploader instructions (200MB line) */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* General Markdown / captions / custom classes */
.stMarkdown p, .stCaption, .uploader-desc, .step-label,
.metric-badge .label, .metric-badge .unit {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Header buttons — Cyan visible on both themes */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] button p,
header[data-testid="stHeader"] svg {
    color: #22d3ee !important;
    fill: #22d3ee !important;
}

/* Help / Hint Icons (Question Marks) - Make them Pop */
[data-testid="stHelpIcon"], 
[data-testid="stTooltipIcon"],
div[data-testid="stMarkdown"] svg[data-testid="stHelpIcon"] {
    color: #22d3ee !important;
    fill: #22d3ee !important;
    transform: scale(1.2); /* Slightly larger */
    transition: all 0.3s ease;
    opacity: 1 !important;
}
[data-testid="stHelpIcon"]:hover {
    filter: drop-shadow(0 0 8px rgba(34, 211, 238, 0.8));
    transform: scale(1.3);
}

/* Title always stays Cyan */
.hero-header h1, .hero-header h1 * {
    color: var(--primary) !important;
    background: transparent !important;
}

/* ══════════════════════════════════════════════
   LIGHT MODE SPECIFICS
   ══════════════════════════════════════════════ */
@media (prefers-color-scheme: light) {
    .stApp {
        background-image: radial-gradient(circle at 50% 0%, #f1f5f9 0%, #f8fafc 100%) !important;
        background-color: #f8fafc !important;
    }
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


def render_stepper(step_idx):
    steps = ["Upload", "Analysis", "Results"]
    html = '<div class="stepper-container">'
    html += '<div class="stepper-line"></div>'
    for i, label in enumerate(steps):
        active_class = "active" if i == step_idx else ""
        html += f'<div class="step {active_class}">'
        html += f'<div class="step-circle">{i+1}</div>'
        html += f'<div class="step-label">{label}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def overlay_heatmap(orig_pil: Image.Image, heatmap_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    orig_np = np.array(orig_pil.convert("RGB"))
    hm_resized = cv2.resize(heatmap_rgb, (orig_np.shape[1], orig_np.shape[0]))
    blended = cv2.addWeighted(orig_np, 1 - alpha, hm_resized, alpha, 0)
    return blended


def run_inference(image_pil: Image.Image, model):
    orig_img, img_tensor = preprocess_image(image_pil)
    with torch.no_grad():
        dm = model(img_tensor)
    density_map = dm.squeeze().cpu().numpy()
    total_count = float(np.sum(density_map))
    return orig_img, density_map, total_count


def upload_file_id(uploaded_file):
    fallback = f"{uploaded_file.name}-{uploaded_file.size}"
    return getattr(uploaded_file, "file_id", fallback)


def alert_level(total_count: float, capacity: int):
    ratio = total_count / max(capacity, 1)
    if ratio < 0.70:
        return "SAFE", "🟢", "#22c55e", ratio
    if ratio <= 0.90:
        return "MONITOR", "🟡", "#eab308", ratio
    return "ALERT", "🔴", "#ef4444", ratio


def compute_zone_stats(density_map: np.ndarray, rows: int, cols: int):
    h, w = density_map.shape
    total = float(np.sum(density_map)) + 1e-8
    stats = []
    for r in range(rows):
        y0 = int(round(r * h / rows))
        y1 = int(round((r + 1) * h / rows))
        for c in range(cols):
            x0 = int(round(c * w / cols))
            x1 = int(round((c + 1) * w / cols))
            zone_count = float(np.sum(density_map[y0:y1, x0:x1]))
            stats.append(
                {
                    "zone": f"R{r + 1}C{c + 1}",
                    "row": r + 1,
                    "col": c + 1,
                    "count": zone_count,
                    "share_pct": (zone_count / total) * 100,
                }
            )
    return sorted(stats, key=lambda item: item["count"], reverse=True)


def draw_zone_grid(image_pil: Image.Image, rows: int, cols: int, hotspot_zone=None):
    image = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for r in range(1, rows):
        y = int(round(r * h / rows))
        draw.line([(0, y), (w, y)], fill=(180, 180, 180), width=2)
    for c in range(1, cols):
        x = int(round(c * w / cols))
        draw.line([(x, 0), (x, h)], fill=(180, 180, 180), width=2)

    if hotspot_zone is not None:
        row_idx, col_idx = hotspot_zone
        x0 = int(round(col_idx * w / cols))
        x1 = int(round((col_idx + 1) * w / cols))
        y0 = int(round(row_idx * h / rows))
        y1 = int(round((row_idx + 1) * h / rows))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 72, 72), width=4)
    return image


def compute_roi_stats(density_map: np.ndarray, rois, total_count: float):
    total = max(float(total_count), 1e-8)
    rows = []
    for roi in rois:
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        roi_count = float(np.sum(density_map[y0:y1, x0:x1]))
        rows.append(
            {
                "zone": roi["name"],
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "count": roi_count,
                "share_pct": (roi_count / total) * 100,
            }
        )
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def draw_roi_overlay(image_pil: Image.Image, rois):
    image = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 99, 132),
        (54, 162, 235),
        (255, 206, 86),
        (75, 192, 192),
        (153, 102, 255),
    ]
    for i, roi in enumerate(rois):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = roi["x0"], roi["y0"], roi["x1"], roi["y1"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text((x0 + 5, max(0, y0 - 18)), roi["name"], fill=color)
    return image


def align_density_map(density_map: np.ndarray, target_shape):
    target_h, target_w = target_shape
    if density_map.shape == target_shape:
        return density_map

    resized = cv2.resize(density_map.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    original_sum = float(np.sum(density_map))
    resized_sum = float(np.sum(resized)) + 1e-8
    return resized * (original_sum / resized_sum)


def diff_to_heatmap(diff_map: np.ndarray):
    max_abs = float(np.max(np.abs(diff_map))) + 1e-8
    pos = np.clip(diff_map / max_abs, 0, 1)
    neg = np.clip(-diff_map / max_abs, 0, 1)
    rgb = np.zeros((*diff_map.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (pos * 255).astype(np.uint8)
    rgb[..., 2] = (neg * 255).astype(np.uint8)
    rgb[..., 1] = np.minimum(rgb[..., 0], rgb[..., 2]) // 3
    return rgb


# ──────────────────────────────────────────────
#  Hero header & Stepper
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--primary); margin-bottom: -5px; margin-right: 10px;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
        HMSTUNet Crowd Counter
    </h1>
    <p>Advanced neural crowd analysis using HMSTUNet architecture for real-time density estimation and spatial intelligence.</p>
</div>
""", unsafe_allow_html=True)

# Placeholder for stepper to keep it at the top
stepper_container = st.empty()

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
st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
st.markdown(f'''
    <div class="uploader-header">
        <span>📁</span> Select source image for crowd analysis
        <div class="tooltip">
            <div class="help-icon">?</div>
            <span class="tooltiptext">High resolution JPG or PNG files recommended.</span>
        </div>
    </div>
''', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Step tracking
current_step = 0
if uploaded_file is not None:
    current_step = 1
    file_id = upload_file_id(uploaded_file)
    if f"result_{file_id}" in st.session_state:
        current_step = 2

with stepper_container:
    render_stepper(current_step)

if uploaded_file is None:
    st.markdown("""
        <div class="info-card">
            <span>ℹ️</span> 
            <div><b>Getting Started:</b> Upload a crowd image above to begin the multi-scale density analysis.</div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
#  Run inference (cached in session state)
# ──────────────────────────────────────────────
file_id = upload_file_id(uploaded_file)

if st.session_state.get("file_id") != file_id:
    image_pil = Image.open(uploaded_file)
    with st.spinner("🧠 Running HMSTUNet inference…"):
        orig_img, density_map, total_count = run_inference(image_pil, model)

    st.session_state.file_id = file_id
    st.session_state.orig_img = orig_img
    st.session_state.density_map = density_map
    st.session_state.total_count = total_count
    st.session_state[f"result_{file_id}"] = True
else:
    # Ensure result state is set if already cached
    st.session_state[f"result_{file_id}"] = True

orig_img: Image.Image = st.session_state.orig_img
density_map: np.ndarray = st.session_state.density_map
total_count: float = st.session_state.total_count

img_w, img_h = orig_img.size

# ──────────────────────────────────────────────
#  Lane-safe multi-feature UI
# ──────────────────────────────────────────────
st.markdown("""
<div class="info-box">
✅ <b>Lane lock:</b> all features below use the same HMSTUNet density map output (single-image inference).
No new model and no video pipeline.
</div>
""", unsafe_allow_html=True)

tab_single, tab_alert, tab_zone, tab_compare = st.tabs(
    ["Single Image Analysis", "Overcrowding Alert", "Zone Analysis", "Comparative Analysis"]
)

with tab_single:
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
    
    # LOCAL CSS for this specific widget to ensure it updates
    st.markdown("""
        <style>
        /* Target the actual Button Group container */
        [data-testid="stButtonGroup"] {
            background-color: transparent !important;
            background: transparent !important;
            gap: 16px !important; /* Increased distance */
            display: flex !important;
            flex-wrap: wrap !important; /* Allow wrapping if small screen */
            padding: 10px 0 !important;
            width: 100% !important;
        }
        
        /* Individual Buttons - Base styling */
        [data-testid^="stBaseButton-segmented_control"] {
            height: 48px !important;
            border-radius: 12px !important;
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            padding: 0 25px !important; /* More horizontal padding */
            color: #ffffff !important;
            transition: all 0.3s ease !important;
            margin: 0 !important;
            flex: 0 1 auto !important; /* Don't force equal width, grow to fit text */
            min-width: 120px !important; /* Ensure enough room for text */
            white-space: nowrap !important; /* PREVENT TRUNCATION (PL...) */
            overflow: visible !important;
        }
        
        /* Force text inside to be full and visible */
        [data-testid^="stBaseButton-segmented_control"] div,
        [data-testid^="stBaseButton-segmented_control"] p {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: nowrap !important;
            width: auto !important;
        }
        
        /* ACTIVE / SELECTED Button - Matches Tabs (Cyan) */
        [data-testid="stBaseButton-segmented_controlActive"] {
            background: #22d3ee !important; /* Cyan */
            color: #000000 !important;
            font-weight: 700 !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 0 20px rgba(34, 211, 238, 0.4) !important;
        }
        
        /* Force black text on active button children */
        [data-testid="stBaseButton-segmented_controlActive"] * {
            color: #000000 !important;
        }
        
        /* Label visibility fix */
        [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
            margin-bottom: 12px !important;
        }

        /* Hover effect */
        [data-testid^="stBaseButton-segmented_control"]:hover {
            border-color: #22d3ee !important;
            background: rgba(34, 211, 238, 0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    selected_cmap_name = st.segmented_control(
        "🎨 Select Colormap",
        options=list(cmap_options.keys()),
        default="JET",
        selection_mode="single",
        key="single_cmap",
    )
    if not selected_cmap_name:
        selected_cmap_name = "JET"

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

with tab_alert:
    st.markdown('<div class="section-title">🚨 Capacity-Based Alert</div>', unsafe_allow_html=True)
    default_capacity = max(int(np.ceil(total_count * 1.2)), 1)
    capacity = st.number_input(
        "Venue capacity (people)",
        min_value=1,
        value=default_capacity,
        step=10,
        help="Alert level is computed from estimated count ÷ venue capacity.",
    )
    level_text, level_icon, level_color, ratio = alert_level(total_count, int(capacity))
    ratio_pct = ratio * 100

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border:1px solid {level_color};
                border-radius:12px;padding:1rem 1.2rem;margin:0.8rem 0 0.6rem;">
        <div style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;">
            Occupancy Status
        </div>
        <div style="font-size:1.8rem;font-weight:700;color:{level_color};">
            {level_icon} {level_text}
        </div>
        <div style="color:#cbd5e1;font-size:0.95rem;">
            Estimated occupancy: <b>{int(total_count):,}</b> / <b>{int(capacity):,}</b> ({ratio_pct:.1f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(ratio, 1.0))

    st.caption("Thresholds: SAFE < 70% | MONITOR 70–90% | ALERT > 90%")

with tab_zone:
    st.markdown('<div class="section-title">🧩 Zone-Based Density Analysis</div>', unsafe_allow_html=True)
    st.caption("Use grid zones for quick overview, or custom ROI zones for perspective-aware monitoring.")

    st.markdown("##### Grid overview")
    zc1, zc2 = st.columns(2)
    with zc1:
        if "grid_rows" not in st.session_state: st.session_state.grid_rows = 3
        c1, c2 = st.columns([1, 1])
        c1.markdown("<div style='color: var(--text-dim); font-size: 0.95rem; font-weight: 500;'>Grid rows</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='text-align: right; color: var(--text-main); font-weight: 600;'>{st.session_state.grid_rows}</div>", unsafe_allow_html=True)
        rows = st.slider("Grid rows", min_value=2, max_value=6, value=st.session_state.grid_rows, label_visibility="collapsed", key="grid_rows")
    with zc2:
        if "grid_cols" not in st.session_state: st.session_state.grid_cols = 3
        c1, c2 = st.columns([1, 1])
        c1.markdown("<div style='color: var(--text-dim); font-size: 0.95rem; font-weight: 500;'>Grid columns</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='text-align: right; color: var(--text-main); font-weight: 600;'>{st.session_state.grid_cols}</div>", unsafe_allow_html=True)
        cols = st.slider("Grid columns", min_value=2, max_value=6, value=st.session_state.grid_cols, label_visibility="collapsed", key="grid_cols")

    zone_stats = compute_zone_stats(density_map, rows, cols)
    hotspot = zone_stats[0]
    zone_grid_img = draw_zone_grid(orig_img, rows, cols, hotspot_zone=(hotspot["row"] - 1, hotspot["col"] - 1))

    v1, v2 = st.columns([1.1, 1.3])
    with v1:
        st.image(zone_grid_img, caption=f"Grid overlay (hotspot: {hotspot['zone']})", use_container_width=True)
    with v2:
        html_rows = ""
        for z in zone_stats:
            html_rows += f"<tr><td>{z['zone']}</td><td>{int(round(z['count']))}</td><td>{round(z['share_pct'], 2)}</td></tr>"
            
        st.markdown(f"""
        <div class="glass-table-container">
            <table class="glass-table">
                <thead>
                    <tr><th>Zone</th><th>Count (est.)</th><th>Share (%)</th></tr>
                </thead>
                <tbody>
                    {html_rows}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### Perspective-aware custom ROI zones")
    st.caption("Define meaningful areas (e.g., entrance, stage-left, exits) using pixel ranges.")

    roi_count = st.number_input(
        "Number of ROI zones",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Each ROI zone is a custom rectangle over the same HMSTUNet density map.",
    )

    rois = []
    for i in range(int(roi_count)):
        default_name = f"Zone {i + 1}"
        default_x0 = int(i * img_w / max(int(roi_count), 1))
        default_x1 = int((i + 1) * img_w / max(int(roi_count), 1))
        default_x1 = min(max(default_x1, default_x0 + 1), img_w)

        with st.expander(f"ROI {i + 1}", expanded=(i == 0)):
            zone_name = st.text_input("Zone name", value=default_name, key=f"roi_name_{i}").strip() or default_name
            c_left, c_right = st.columns(2)
            with c_left:
                x0 = st.slider("x_start", min_value=0, max_value=img_w - 1, value=default_x0, key=f"roi_x0_{i}")
                y0 = st.slider("y_start", min_value=0, max_value=img_h - 1, value=0, key=f"roi_y0_{i}")
            with c_right:
                x1 = st.slider("x_end", min_value=min(x0 + 1, img_w), max_value=img_w, value=default_x1, key=f"roi_x1_{i}")
                y1 = st.slider("y_end", min_value=min(y0 + 1, img_h), max_value=img_h, value=img_h, key=f"roi_y1_{i}")
            rois.append({"name": zone_name, "x0": x0, "y0": y0, "x1": x1, "y1": y1})

    roi_stats = compute_roi_stats(density_map, rois, total_count)
    roi_overlay = draw_roi_overlay(orig_img, rois)
    roi_hotspot = roi_stats[0]

    r1, r2 = st.columns([1.1, 1.3])
    with r1:
        st.image(roi_overlay, caption=f"Custom ROI overlay (hotspot: {roi_hotspot['zone']})", use_container_width=True)
    with r2:
        html_rows = ""
        for z in roi_stats:
            html_rows += f"<tr><td>{z['zone']}</td><td>{int(round(z['count']))}</td><td>{round(z['share_pct'], 2)}</td><td>({z['x0']},{z['y0']},{z['x1']},{z['y1']})</td></tr>"
            
        st.markdown(f"""
        <div class="glass-table-container">
            <table class="glass-table">
                <thead>
                    <tr><th>ROI Zone</th><th>Count (est.)</th><th>Share (%)</th><th>Box (x0,y0,x1,y1)</th></tr>
                </thead>
                <tbody>
                    {html_rows}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="section-title">🔁 Before/After Comparative Analysis</div>', unsafe_allow_html=True)
    compare_file = st.file_uploader(
        "Upload a second image of the same location (before/after)",
        type=["jpg", "jpeg", "png"],
        key="compare_uploader",
        help="For best results, keep camera angle and framing similar.",
    )

    if compare_file is None:
        st.info("Upload a second image to compute count change and hotspot differences.")
    else:
        compare_file_id = upload_file_id(compare_file)
        if st.session_state.get("compare_file_id") != compare_file_id:
            compare_image = Image.open(compare_file)
            with st.spinner("Running HMSTUNet on comparison image…"):
                compare_orig, compare_density_map, compare_total_count = run_inference(compare_image, model)
            st.session_state.compare_file_id = compare_file_id
            st.session_state.compare_orig_img = compare_orig
            st.session_state.compare_density_map = compare_density_map
            st.session_state.compare_total_count = compare_total_count

        compare_orig: Image.Image = st.session_state.compare_orig_img
        compare_density_map: np.ndarray = st.session_state.compare_density_map
        compare_total_count: float = st.session_state.compare_total_count

        count_delta = compare_total_count - total_count
        pct_delta = (count_delta / max(total_count, 1e-8)) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Before count", f"{int(total_count):,}")
        c2.metric("After count", f"{int(compare_total_count):,}")
        c3.metric("Delta", f"{int(round(count_delta)):+,}", f"{pct_delta:+.1f}%")

        aligned_compare = align_density_map(compare_density_map, density_map.shape)
        diff_density = aligned_compare - density_map
        diff_heatmap = diff_to_heatmap(diff_density)
        diff_overlay = overlay_heatmap(orig_img, diff_heatmap, alpha=0.60)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.image(orig_img, caption="Before image", use_container_width=True)
        with d2:
            st.image(compare_orig, caption="After image", use_container_width=True)
        with d3:
            st.image(diff_overlay, caption="Difference map (red=increase, blue=decrease)", use_container_width=True)
