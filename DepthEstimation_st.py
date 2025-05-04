import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MiDaS model and transform
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        model.to(device)
        model.eval()
        return model, transform
    except Exception as e:
        st.error(f"Error loading MiDaS model: {str(e)}")
        return None, None

midas, transform = load_model()

if midas is None or transform is None:
    st.stop()

st.title("Depth Estimation using MiDaS")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
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
        depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        # Grayscale Plot
        st.subheader("Estimated Depth Map (Grayscale)")
        fig_gray, ax_gray = plt.subplots()
        gray_map = ax_gray.imshow(depth_map_norm, cmap="gray")
        ax_gray.set_title("Grayscale Depth (Near = Bright, Far = Dark)", fontsize=12)
        cbar_gray = plt.colorbar(gray_map, ax=ax_gray, orientation="vertical")
        cbar_gray.set_label("Depth Intensity", rotation=270, labelpad=15)
        cbar_gray.ax.set_yticklabels(["Far", "", "", "", "", "", "Near"])
        st.pyplot(fig_gray)

        # Heatmap Plot
        st.subheader("Estimated Depth Map (Heatmap)")
        fig_heat, ax_heat = plt.subplots()
        heat_map = ax_heat.imshow(depth_map_norm, cmap="inferno")
        ax_heat.set_title("Heatmap Depth (Near = Bright, Far = Dark)", fontsize=12)
        cbar_heat = plt.colorbar(heat_map, ax=ax_heat, orientation="vertical")
        cbar_heat.set_label("Depth Intensity", rotation=270, labelpad=15)
        cbar_heat.ax.set_yticklabels(["Far", "", "", "", "", "", "Near"])
        st.pyplot(fig_heat)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
