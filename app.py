import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

from model import HMSTUNet

st.set_page_config(page_title="HMSTUNet Crowd Counter", layout="wide")

@st.cache_resource
def load_model():
    model = HMSTUNet(pretrained=False)
    checkpoint = torch.load("checkpoints/best.pth", map_location='cpu')
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
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Make sure `checkpoints/best.pth` exists. Error: {e}")
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
