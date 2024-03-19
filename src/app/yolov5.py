import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


def read_image(img_file_buffer=None):
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    return img_array


def detect(image):
    results = model([image])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()

    results.save(save_dir=tmpdir)
    print(list(tmpdir.glob("*")))
    im1 = Image.open(tmpdir / "image0.jpg")
    return im1


def draw_image(image):
    st.image(
        image,
        use_column_width=True,
    )


def upload_image():
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    return img_file_buffer


def run_yolov5():
    st.subheader("Yolov5")
    img_file_buffer = upload_image()

    col1, col2 = st.columns(2)

    if img_file_buffer is not None:
        image = read_image(img_file_buffer)
        with col1:
            st.button("Show Original Only")
            draw_image(image)

        with col2:
            button = st.button("Detect")
            if button:
                detected_image = detect(image)
                draw_image(detected_image)
