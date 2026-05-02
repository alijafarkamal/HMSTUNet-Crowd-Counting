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
file_id = upload_file_id(uploaded_file)

if st.session_state.get("file_id") != file_id:
    image_pil = Image.open(uploaded_file)
    with st.spinner("🧠 Running HMSTUNet inference…"):
        orig_img, density_map, total_count = run_inference(image_pil, model)

    st.session_state.file_id = file_id
    st.session_state.orig_img = orig_img
    st.session_state.density_map = density_map
    st.session_state.total_count = total_count

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
    selected_cmap_name = st.radio(
        "🎨 Select Colormap",
        options=list(cmap_options.keys()),
        index=0,
        horizontal=True,
        key="single_cmap",
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
        rows = st.slider("Grid rows", min_value=2, max_value=6, value=3)
    with zc2:
        cols = st.slider("Grid columns", min_value=2, max_value=6, value=3)

    zone_stats = compute_zone_stats(density_map, rows, cols)
    hotspot = zone_stats[0]
    zone_grid_img = draw_zone_grid(orig_img, rows, cols, hotspot_zone=(hotspot["row"] - 1, hotspot["col"] - 1))

    v1, v2 = st.columns([1.1, 1.3])
    with v1:
        st.image(zone_grid_img, caption=f"Grid overlay (hotspot: {hotspot['zone']})", use_container_width=True)
    with v2:
        st.dataframe(
            [
                {
                    "Zone": z["zone"],
                    "Count (est.)": int(round(z["count"])),
                    "Share (%)": round(z["share_pct"], 2),
                }
                for z in zone_stats
            ],
            use_container_width=True,
            hide_index=True,
        )

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
        st.dataframe(
            [
                {
                    "ROI Zone": z["zone"],
                    "Count (est.)": int(round(z["count"])),
                    "Share (%)": round(z["share_pct"], 2),
                    "Box (x0,y0,x1,y1)": f"({z['x0']},{z['y0']},{z['x1']},{z['y1']})",
                }
                for z in roi_stats
            ],
            use_container_width=True,
            hide_index=True,
        )

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
