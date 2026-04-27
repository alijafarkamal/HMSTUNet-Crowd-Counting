# 👥 HMSTUNet Crowd Counting App

🚀 A lightweight, interactive Streamlit application for accurate crowd density estimation using the **HMSTUNet** architecture.

**Pretrained weights (`best.pth`):** [download](https://huggingface.co/aliJafar/hmstunet-weights/resolve/main/best.pth) · [Hugging Face repo](https://huggingface.co/aliJafar/hmstunet-weights/tree/main) — save as `checkpoints/best.pth` locally, or use the download URL as **`CHECKPOINT_URL`** in Streamlit Cloud Secrets.

---

## 🌟 Features
- **Upload & Analyze**: Drag and drop any image (JPG, PNG) to estimate the number of people.
- **Heatmap Visualization**: Instantly generates a jet-colormap density heatmap alongside the original image.
- **Efficient Inference**: Pre-configured model logic handles resizing, normalization, and tensor conversions seamlessly.

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
