import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
from io import BytesIO
from streamlit.runtime.uploaded_file_manager import UploadedFile

model = torch.hub.load("ultralytics/yolov5", "yolov5s")


def read_image(img_file_buffer: BytesIO) -> np.array:
    """이미지를 읽습니다.

    Args:
        img_file_buffer (BytesIO): 이미지 파일 버퍼. Defaults to None.

    Returns:
        np.array: np.array 형식의 이미지
    """
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    return img_array


def detect(image: np.array) -> np.array:
    """yolov5 모델을 사용하여 이미지 내의 객체를 탐지하고 탐지 결과를 이미지로 반환합니다.

    Args:
        image (np.array): 이미지

    Returns:
        np.array: 객체의 위치가 상자로 표시된 이미지
    """
    results = model([image])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()

    results.save(save_dir=tmpdir)
    im1 = Image.open(tmpdir / "image0.jpg")
    return im1


def draw_image(image: np.array):
    """이미지를 streamlit 이미지로 표현합니다.

    Args:
        image (np.array): 이미지
    """
    st.image(
        image,
        use_column_width=True,
    )


def upload_image() -> UploadedFile:
    """이미지를 업로드합니다.

    Returns:
        UploadedFile: 업로드된 이미지
    """
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    return img_file_buffer


def main():
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
