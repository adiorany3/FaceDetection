import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2

st.set_page_config(page_title="Realtime Face Detection")
st.title("Realtime Face Detection (Kotak Hijau)")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class FaceDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return img

webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
)
