import streamlit as st
import cv2
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Face Detection",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# HIDE STREAMLIT ELEMENTS
# =========================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("Face Detection")

# =========================
# LOAD FACE CASCADE
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================
# CAMERA INPUT
# =========================
uploaded_image = st.camera_input("Ambil Foto")

if uploaded_image is not None:

    # Convert image bytes
    file_bytes = np.asarray(
        bytearray(uploaded_image.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # =========================
    # RESIZE IMAGE
    # =========================
    h, w = image.shape[:2]

    if w > 1000:
        ratio = 1000 / w
        image = cv2.resize(
            image,
            (1000, int(h * ratio))
        )

    # =========================
    # PREPROCESSING
    # =========================
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # =========================
    # FACE DETECTION
    # =========================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # =========================
    # DRAW RESULTS
    # =========================
    for (x, y, fw, fh) in faces:

        # Green rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            4
        )

        # Label
        cv2.putText(
            image,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # =========================
    # SHOW RESULT
    # =========================
    st.image(
        image_rgb,
        caption=f"Terdeteksi {len(faces)} wajah",
        use_container_width=True
    )

    if len(faces) == 0:
        st.warning(
            "Wajah tidak terdeteksi. Coba gunakan pencahayaan yang lebih terang, posisi wajah menghadap kamera, dan jarak yang tidak terlalu jauh."
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center><b>Developed by Galuh Adi Insani</b></center>",
    unsafe_allow_html=True
)
