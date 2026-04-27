import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

from model import HMSTUNet

st.set_page_config(page_title="HMSTUNet Crowd Counter", layout="wide")

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
    """Local file, or download once from CHECKPOINT_URL (env or Streamlit secrets)."""
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
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; HMSTUNet/1.0)"},
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
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

st.title("👥 HMSTUNet Crowd Counting")
st.markdown("Upload an image to estimate the crowd count and visualize the density map using the **HMSTUNet** model.")

try:
    with st.spinner("Loading model (first run may download weights)..."):
        model = load_model()
except Exception as e:
    st.error(
        f"Failed to load model. Add `checkpoints/best.pth` locally, or set **CHECKPOINT_URL** "
        f"in Streamlit **Secrets** to a direct `.pth` download link. Error: {e}"
    )
    st.info(
        "**Streamlit Cloud:** Open the app menu (**⋮**) → **Settings** → **Secrets**. Use a "
        "**top-level** key `CHECKPOINT_URL = \"https://…/best.pth\"` (must be a direct file URL, "
        "not a share page), click **Save**, then **Reboot** the app. You can instead use a nested "
        "table: `[checkpoint]` with `url = \"…\"`. "
        "**Local run:** put `best.pth` in `checkpoints/` or export `CHECKPOINT_URL` before starting Streamlit."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")
    
    # Resize to multiple of 32 for ViT/ConvNeXt compatibility
    w, h = image.size
    new_w = w // 32 * 32
    new_h = h // 32 * 32
    
    # Avoid resizing to 0
    new_w = max(32, new_w)
    new_h = max(32, new_h)
    
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return image, transform(image).unsqueeze(0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner('Running HMSTUNet inference...'):
        orig_img_resized, img_tensor = preprocess_image(image)
        
        with torch.no_grad():
            density_map = model(img_tensor)
            
        density_map = density_map.squeeze().cpu().numpy()
        # The sum of the density map gives the total crowd count
        count = np.sum(density_map)
        
        # Normalize density map for visualization
        density_map_norm = density_map / (np.max(density_map) + 1e-5)
        # Apply jet colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * density_map_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        st.success(f"### 🧑‍🤝‍🧑 Estimated Crowd Count: **{int(count)}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_img_resized, caption="Original Image", use_container_width=True)
        with col2:
            st.image(heatmap, caption="Density Heatmap", use_container_width=True)
