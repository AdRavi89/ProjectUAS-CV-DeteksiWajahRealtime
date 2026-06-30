"""
app.py
Entry point aplikasi Streamlit - Realtime Smart Face Detection.

Menggabungkan:
- streamlit-webrtc untuk akses webcam realtime di browser
- YOLOv8 (detector.py) untuk deteksi wajah
- DeepFace (gender.py) untuk analisis gender (dengan frame skipping)
- utils.py untuk FPS counter dan timestamp
"""

import os
import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from detector import FaceDetector
from gender import GenderAnalyzer
from age import AgeAnalyzer
from utils import FPSCounter, get_indonesian_date, get_current_time, timestamp_filename

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Realtime Smart Face Detection",
    page_icon="📸",
    layout="wide",
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ----------------------------------------------------------------------
# Sidebar - Panel Kontrol
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Panel Kontrol")

st.sidebar.subheader("Kamera")
camera_enabled = st.sidebar.checkbox("Aktifkan Kamera", value=True)

st.sidebar.subheader("Model")
st.sidebar.radio("Pilih Model", ["YOLOv8 Face"], index=0, disabled=True)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)

st.sidebar.subheader("Informasi Tampilan")
show_fps = st.sidebar.checkbox("Tampilkan FPS", value=True)
show_gender = st.sidebar.checkbox("Tampilkan Gender", value=True)
show_age = st.sidebar.checkbox("Tampilkan Umur", value=True)
show_time = st.sidebar.checkbox("Tampilkan Jam", value=True)
show_date = st.sidebar.checkbox("Tampilkan Tanggal", value=True)

st.sidebar.subheader("Performa")
skip_frames = st.sidebar.slider("Analisis Gender Tiap N Frame", 1, 30, 10)

st.sidebar.markdown("---")
snapshot_clicked = st.sidebar.button("📷 Simpan Snapshot", use_container_width=True)

# ----------------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------------
st.title("📸 Realtime Smart Face Detection")
st.caption("YOLOv8 Face Detection + DeepFace Gender Analysis — powered by Streamlit")

col_video, col_info = st.columns([3, 1])

with col_info:
    st.subheader("📊 Informasi Realtime")
    fps_placeholder = st.empty()
    date_placeholder = st.empty()
    time_placeholder = st.empty()
    face_count_placeholder = st.empty()
    st.markdown("---")
    snapshot_status_placeholder = st.empty()


# ----------------------------------------------------------------------
# Video Processor
# ----------------------------------------------------------------------
class FaceDetectionProcessor:
    def __init__(self):
        self.detector = FaceDetector(conf_threshold=conf_threshold)
        self.gender_analyzer = GenderAnalyzer(analyze_every_n_frames=skip_frames)
        self.age_analyzer = AgeAnalyzer(analyze_every_n_frames=skip_frames)
        self.fps_counter = FPSCounter()
        self.face_count = 0
        self.last_frame = None
        self.lock = threading.Lock()
        self.snapshot_requested = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        faces = self.detector.detect(img)
        self.face_count = len(faces)

        for idx, face in enumerate(faces):
            box = face["box"]
            det_conf = face["confidence"]

            label_parts = []
            face_crop = self.detector.crop_face(img, box) if (show_gender or show_age) else None

            if show_gender:
                gender, gender_conf = self.gender_analyzer.get_cached_or_analyze(idx, face_crop)
                label_parts.append(GenderAnalyzer.format_label(gender, gender_conf))

            if show_age:
                age = self.age_analyzer.get_cached_or_analyze(idx, face_crop)
                label_parts.append(AgeAnalyzer.format_label(age))

            if not label_parts:
                label_parts.append(f"Face {det_conf*100:.0f}%")

            label = " | ".join(label_parts)
            self.detector.draw_box(img, box, label)

        fps = self.fps_counter.update()

        # Overlay info di bagian bawah frame
        overlay_lines = []
        if show_fps:
            overlay_lines.append(f"FPS: {fps}")
        if show_date:
            overlay_lines.append(f"Tanggal: {get_indonesian_date()}")
        if show_time:
            overlay_lines.append(f"Jam: {get_current_time()}")
        overlay_lines.append(f"Wajah: {self.face_count}")

        y_offset = img.shape[0] - 15
        for line in reversed(overlay_lines):
            cv2.putText(img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset -= 25

        with self.lock:
            self.last_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------------------------------------------------
# Webcam Stream
# ----------------------------------------------------------------------
with col_video:
    if camera_enabled:
        ctx = webrtc_streamer(
            key="realtime-face-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FaceDetectionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        ctx = None
        st.info("Aktifkan kamera dari sidebar untuk memulai deteksi.")

# ----------------------------------------------------------------------
# Update info panel & handle snapshot
# ----------------------------------------------------------------------
if ctx and ctx.video_processor:
    processor = ctx.video_processor

    if show_fps:
        fps_placeholder.metric("FPS", processor.fps_counter.fps)
    if show_date:
        date_placeholder.write(f"**Tanggal:** {get_indonesian_date()}")
    if show_time:
        time_placeholder.write(f"**Jam:** {get_current_time()}")
    face_count_placeholder.metric("Jumlah Wajah Terdeteksi", processor.face_count)

    if snapshot_clicked:
        with processor.lock:
            frame_to_save = processor.last_frame.copy() if processor.last_frame is not None else None

        if frame_to_save is not None:
            filename = timestamp_filename(prefix="hasil")
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame_to_save)
            snapshot_status_placeholder.success(f"✅ Snapshot disimpan: `{filepath}`")
        else:
            snapshot_status_placeholder.warning("⚠️ Belum ada frame untuk disimpan.")
else:
    fps_placeholder.write("**FPS:** -")
    date_placeholder.write(f"**Tanggal:** {get_indonesian_date()}")
    time_placeholder.write(f"**Jam:** {get_current_time()}")
    face_count_placeholder.write("**Jumlah Wajah:** -")

st.markdown("---")
st.caption(
    "💡 Tips performa: gunakan slider 'Analisis Gender Tiap N Frame' di sidebar "
    "untuk menyeimbangkan akurasi vs kecepatan (FPS)."
)
