import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Face Detection",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# HIDE STREAMLIT ELEMENTS
# ==========================
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stToolbar"] {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Face Detection")

# ==========================
# MEDIAPIPE FACE DETECTION
# ==========================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ==========================
# CAMERA INPUT
# ==========================
image_file = st.camera_input("Ambil Foto")

if image_file is not None:

    file_bytes = np.asarray(
        bytearray(image_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    detected_faces = 0

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:

        results = face_detection.process(image_rgb)

        if results.detections:

            h, w, _ = image.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    3
                )

                detected_faces += 1

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        image_rgb,
        caption=f"Terdeteksi {detected_faces} wajah",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<center><b>Developed by Your Name</b></center>",
    unsafe_allow_html=True
)
