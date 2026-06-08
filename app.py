import streamlit as st
import cv2
import numpy as np
import os
import urllib.request

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

if not os.path.exists(MODEL_FILE):
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

st.set_page_config(page_title="Face Detection", page_icon="📷", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Face Detection (YuNet)")

uploaded_image = st.camera_input("Ambil Foto")

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE,
        "",
        (w, h),
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000
    )

    detector.setInputSize((w, h))

    _, faces = detector.detect(image)

    count = 0

    if faces is not None:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)

            cv2.rectangle(
                image,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                3
            )

            cv2.putText(
                image,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            count += 1

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(
        image_rgb,
        caption=f"Terdeteksi {count} wajah",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<center><b>Developed by Galuh Adi Insani</b></center>",
    unsafe_allow_html=True
)
