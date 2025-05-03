import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MiDaS model and transform
@st.cache_resource
def load_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    model.to(device)
    model.eval()
    return model, transform

midas, transform = load_model()

st.title("Depth Estimation using MiDaS")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    input_batch = transform(img_np).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_img = (depth_map * 255).astype(np.uint8)

    st.image(depth_img, caption="Estimated Depth Map", use_container_width=True)
