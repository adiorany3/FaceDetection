import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Face Detection",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hilangkan elemen Streamlit
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Face Detection")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Ambil gambar dari kamera
image = st.camera_input("Ambil Foto")

if image is not None:

    file_bytes = np.asarray(
        bytearray(image.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Kotak hijau
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        img_rgb,
        caption=f"Terdeteksi {len(faces)} wajah",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<center><b>Developed by Your Name</b></center>",
    unsafe_allow_html=True
)
