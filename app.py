import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Detection",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Face Detection")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

uploaded_image = st.camera_input("Ambil Foto")

if uploaded_image is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_image.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Resize agar deteksi lebih mudah
    h, w = image.shape[:2]

    if w > 800:
        scale = 800 / w
        image = cv2.resize(
            image,
            (800, int(h * scale))
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, fw, fh) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            3
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        image_rgb,
        caption=f"Terdeteksi {len(faces)} wajah",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<center><b>Developed by Your Name</b></center>",
    unsafe_allow_html=True
)
