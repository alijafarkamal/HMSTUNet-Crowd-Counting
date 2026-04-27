# 👥 HMSTUNet Crowd Counting App

🚀 A lightweight, interactive Streamlit application for accurate crowd density estimation using the **HMSTUNet** architecture.

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

Weights are not stored in git. On the hosted app, either:

1. Open your app on [share.streamlit.io](https://share.streamlit.io) → **App settings** (⋮) → **Secrets**, and add:
   ```toml
   CHECKPOINT_URL = "https://...direct link to best.pth..."
   ```
   Use a **direct file URL** (e.g. Hugging Face: `https://huggingface.co/<user>/<repo>/resolve/main/best.pth`, or a raw/static hosting link). The first visitor may wait while the file downloads once; it is cached on the machine after that.

2. Or keep using a local `checkpoints/best.pth` when you run `streamlit run app.py` on your computer (no secret needed).

See `.streamlit/secrets.toml.example` for a template.

## 📦 Checkpoints & Datasets

- **Model Weights**: Place your trained `best.pth` checkpoint inside the `checkpoints/` directory.
- **ShanghaiTech Dataset**: Used for training & evaluation.
  - 📖 [Dataset GitHub Repository](https://github.com/desenzhou/ShanghaiTechDataset)
  - 📥 [Direct Download (Dropbox)](https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=0)

---
*Built with ❤️ using PyTorch and Streamlit.*
