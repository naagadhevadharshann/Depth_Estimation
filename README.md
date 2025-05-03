# Depth_Estimation
This project demonstrates real-time monocular depth estimation using the MiDaS model from Intel ISL. Upload an image, and the app generates a depth map highlighting the relative distance of objects. Built with PyTorch, OpenCV, and Streamlit, it enables intuitive visualization of depth from a single image.

# 🌄 Depth Estimation with MiDaS

This project uses Intel-ISL's [MiDaS](https://github.com/isl-org/MiDaS) model to estimate depth from a single RGB image. You can use it either as a **CLI script** or an interactive **Streamlit web app**.

---

## 🚀 Features

- Use MiDaS-small for fast and accurate depth prediction.
- Interactive file upload via web browser using Streamlit.
- Normalized depth map output as grayscale image.
- Standalone Python script for terminal-based usage.

---

## 🧠 Model

- `MiDaS_small` (based on MobileNetV2) from `torch.hub` — optimized for speed and portability.
- Input: Single RGB image
- Output: 2D Depth map

---

## 🖥️ Usage

### 📌 1. CLI Version

```bash
python depth_cli.py
