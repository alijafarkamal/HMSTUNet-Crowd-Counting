# 👥 HMSTUNet Crowd Counting App

🚀 A lightweight, interactive Streamlit application for accurate crowd density estimation using the **HMSTUNet** architecture.

**Pretrained weights (`best.pth`):** [download](https://huggingface.co/aliJafar/hmstunet-weights/resolve/main/best.pth) · [Hugging Face repo](https://huggingface.co/aliJafar/hmstunet-weights/tree/main) — save as `checkpoints/best.pth` locally, or use the download URL as **`CHECKPOINT_URL`** in Streamlit Cloud Secrets.

---

## 🌟 Features
- **Upload & Analyze**: Drag and drop any image (JPG, PNG) to estimate the number of people.
- **Heatmap Visualization**: Instantly generates a jet-colormap density heatmap alongside the original image.
- **Overcrowding Alert**: Capacity-based risk level (Safe / Monitor / Alert) from the same predicted count.
- **Zone Analysis**: Grid-based per-zone counts to identify local hotspots.
- **Perspective-aware ROI Zones**: Define custom rectangular regions (entrance, exits, stage-side) and get per-region counts.
- **Comparative Analysis**: Before/after count delta and difference heatmap (red=increase, blue=decrease).
- **Lane-safe by design**: All tabs use the same HMSTUNet density map output (no extra model, no video pipeline).
- **Efficient Inference**: Pre-configured model logic handles resizing, normalization, and tensor conversions seamlessly.

## 📘 Added Functionality Documentation

### 1) Multi-tab GUI
The app now has four tabs:
1. **Single Image Analysis**
2. **Overcrowding Alert**
3. **Zone Analysis**
4. **Comparative Analysis**

All tabs operate from the same HMSTUNet inference output (single-image density map).

### 2) Overcrowding Alert
- Input: venue capacity (people)
- Output: occupancy status from `estimated_count / capacity`
  - **SAFE**: `< 70%`
  - **MONITOR**: `70–90%`
  - **ALERT**: `> 90%`
- Displays status card + occupancy progress bar.

### 3) Zone Analysis (Grid)
- User selects grid rows/columns.
- App computes estimated people count for each grid cell from density-map sums.
- Shows:
  - hotspot zone
  - per-zone count table
  - grid overlay visualization

### 4) Perspective-aware ROI Zones (Custom)
- User selects number of ROI zones (1–5).
- For each ROI:
  - set zone name
  - set rectangle bounds: `x_start`, `y_start`, `x_end`, `y_end`
- App computes per-ROI count directly from the same density map.
- Shows:
  - custom ROI overlay with labels
  - hotspot ROI
  - per-ROI table with count, share %, and box coordinates

This is useful when scene geometry is non-uniform (e.g., entrance appears smaller due to perspective).

### 5) Comparative Analysis (Before/After)
- User uploads a second image of the same location.
- App runs the same HMSTUNet model on both images.
- Outputs:
  - before count
  - after count
  - delta count and delta percentage
  - difference heatmap overlay:
    - **red** = increase
    - **blue** = decrease

### 6) Lane Discipline (What was intentionally not added)
- No new model
- No temporal/video pipeline
- No optical flow
- No task outside single-image density estimation

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alijafarkamal/HMSTUNet-Crowd-Counting.git
   cd HMSTUNet-Crowd-Counting
   ```

2. **Create a virtual environment (recommended) and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 🏋️ Training (Now Ready)

The repository now includes `train.py` so you can train/fine-tune directly (not only notebook-based).

### Lane-first recommendation
Pick one lane and stay there:
- **Part A lane (dense events)**: `--part A`
- **Part B lane (street/sparser scenes)**: `--part B`

Do not mix lanes during first training if you want stable behavior in one scenario.

### What `train.py` does
1. Detects ShanghaiTech part folder under `data/` (`part_A_final|part_A` or `part_B_final|part_B`)
2. Builds missing Gaussian density maps from GT `.mat` files
3. Trains HMSTUNet with:
   - Loss: `MSE(density) + 0.1 * MAE(count)`
   - Optimizer: Adam (lower LR for encoder, higher LR for decoder/head)
   - Scheduler: CosineAnnealingLR
4. Evaluates with MAE/RMSE each epoch
5. Saves:
   - `checkpoints/last.pth` every epoch
   - `checkpoints/best.pth` when validation MAE improves

### Quick start (local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train dense-crowd lane (Part A):
   ```bash
   python train.py --data-root data --part A --epochs 50 --batch-size 4
   ```
3. Or train sparse/street lane (Part B):
   ```bash
   python train.py --data-root data --part B --epochs 50 --batch-size 4
   ```

### Useful commands
- Generate density maps only:
  ```bash
  python train.py --data-root data --part A --generate-density-only
  ```
- Resume training:
  ```bash
  python train.py --data-root data --part A --resume checkpoints/last.pth
  ```
- Force regenerate density maps:
  ```bash
  python train.py --data-root data --part A --force-density
  ```

### Kaggle step-by-step (external work)

1. Create a new Kaggle notebook (GPU enabled: T4/P100).
2. Add your dataset so folder contains:
   - `.../data/part_A_final/...` (or `part_B_final/...`)
3. In notebook:
   ```bash
   !git clone https://github.com/alijafarkamal/HMSTUNet-Crowd-Counting.git
   %cd HMSTUNet-Crowd-Counting
   !pip install -r requirements.txt
   ```
4. Copy dataset into repo `data/` path (or point `--data-root` to mounted dataset path).
5. Train:
   ```bash
   !python train.py --data-root /kaggle/input/<your-dataset-folder> --part A --epochs 50 --batch-size 4 --num-workers 2
   ```
6. Save artifact:
   - Best weights will be at `checkpoints/best.pth`
   - In Kaggle, output files are preserved from `/kaggle/working/...`
7. Download `best.pth`, then place it in your app repo under:
   - `checkpoints/best.pth`

### Notes
- If GPU memory is tight, reduce `--batch-size` (e.g., 2).
- If training is too slow, reduce `--train-crop` and `--epochs` for first test runs.
- Streamlit app loads `checkpoints/best.pth` automatically.

## ☁️ Streamlit Community Cloud

Weights are not stored in git. On the hosted app you must either supply a **direct download URL** via Secrets or run locally with a file on disk.

### Secrets (required on Cloud)

1. Open [share.streamlit.io](https://share.streamlit.io) → your app → **⋮** → **Settings** → **Secrets**.
2. Use a **flat**, case-sensitive key at the top level of the TOML (recommended):

   ```toml
   CHECKPOINT_URL = "https://example.com/path/best.pth"
   ```

   Or a **nested** table (also supported by the app):

   ```toml
   [checkpoint]
   url = "https://example.com/path/best.pth"
   ```

3. Click **Save**, then **Reboot** the app. Secrets are not always applied until a reboot.

### URL must be a direct file link

Opening the URL in a browser should **download** `best.pth` (or start a binary download), not show HTML.

- **Hugging Face (public file):** `https://huggingface.co/<user>/<repo>/resolve/main/best.pth`
- **Dropbox:** use a link with `dl=1` (not the `dl=0` preview page).
- **Google Drive:** convert the share link to a direct-download form; a plain “share” URL often fails.

The first launch after a reboot may take a while while weights download; they are then cached on the runner.

### If it still says missing checkpoint / CHECKPOINT_URL

- Wrong or nested-under-wrong-name keys (e.g. `[weights]` instead of `[checkpoint]` with `url`, or typo `CHECKPOINT_URI`).
- Secret value empty or URL not in quotes when the URL contains `#` or special characters.
- App not **rebooted** after saving Secrets.
- URL is not direct (login page, HTML preview, virus-scan interstitial).

### Local run

Use `checkpoints/best.pth` on disk, or:

```bash
export CHECKPOINT_URL="https://…/best.pth"
streamlit run app.py
```

See [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) for a template.

## 📦 Checkpoints & Datasets

- **Model Weights**: Place your trained `best.pth` checkpoint inside the `checkpoints/` directory.
- **ShanghaiTech Dataset**: Used for training & evaluation.
  - 📖 [Dataset GitHub Repository](https://github.com/desenzhou/ShanghaiTechDataset)
  - 📥 [Direct Download (Dropbox)](https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=0)

---
*Built with ❤️ using PyTorch and Streamlit.*
