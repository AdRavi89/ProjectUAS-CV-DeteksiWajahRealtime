"""
app.py
Realtime Smart Face Detection — Entry point Streamlit.

Jalankan dengan:
    streamlit run app.py

JANGAN jalankan dengan:
    python app.py
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

# ── Konfigurasi halaman ───────────────────────────────────────────────
st.set_page_config(
    page_title="Realtime Smart Face Detection",
    page_icon="📸",
    layout="wide",
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Panel Kontrol")

    st.subheader("📷 Kamera")
    camera_on = st.checkbox("Aktifkan Kamera", value=True)

    st.subheader("🔍 Model")
    st.radio("Pilih Model", ["YOLOv8 Face"], index=0, disabled=True)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.45, 0.05)

    st.subheader("📊 Informasi Tampilan")
    show_fps    = st.checkbox("Tampilkan FPS",     value=True)
    show_gender = st.checkbox("Tampilkan Gender",  value=True)
    show_age    = st.checkbox("Tampilkan Umur",    value=True)
    show_time   = st.checkbox("Tampilkan Jam",     value=True)
    show_date   = st.checkbox("Tampilkan Tanggal", value=True)

    st.subheader("⚡ Performa")
    skip_n = st.slider("Analisis Gender & Umur Tiap N Frame", 1, 30, 10)

    st.markdown("---")
    snap_btn = st.button("📷 Simpan Snapshot", use_container_width=True)
    snap_status = st.empty()

# ── Main page ─────────────────────────────────────────────────────────
st.title("📸 Realtime Smart Face Detection")
st.caption("YOLOv8 · DeepFace Gender & Age · Streamlit WebRTC")
st.markdown("---")

col_video, col_info = st.columns([3, 1])

with col_info:
    st.subheader("📊 Info Realtime")
    ph_fps   = st.empty()
    ph_date  = st.empty()
    ph_time  = st.empty()
    ph_faces = st.empty()


# ── Video Processor ───────────────────────────────────────────────────
class FaceProcessor:
    def __init__(self):
        self.detector = FaceDetector(conf_threshold=conf_threshold)
        self.gender   = GenderAnalyzer(analyze_every_n_frames=skip_n)
        self.age      = AgeAnalyzer(analyze_every_n_frames=skip_n)
        self.fps      = FPSCounter()
        self.lock     = threading.Lock()
        self.last_frame = None
        self.face_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        faces = self.detector.detect(img)
        self.face_count = len(faces)

        for idx, face in enumerate(faces):
            box  = face["box"]
            conf = face["confidence"]

            # Crop sekali, pakai bersama gender + umur
            crop = self.detector.crop_face(img, box) if (show_gender or show_age) else None

            parts = []
            if show_gender and crop is not None:
                g, gc = self.gender.get(idx, crop)
                parts.append(GenderAnalyzer.label(g, gc))
            if show_age and crop is not None:
                a = self.age.get(idx, crop)
                parts.append(AgeAnalyzer.label(a))
            if not parts:
                parts.append(f"{conf*100:.0f}%")

            label = " | ".join(parts)
            self.detector.draw_box(img, box, label)

        # ── Overlay teks bawah frame ──────────────────────────────────
        current_fps  = self.fps.update()
        current_date = get_indonesian_date()
        current_time = get_current_time()

        overlay = []
        if show_fps:    overlay.append(f"FPS      : {current_fps}")
        if show_date:   overlay.append(f"Tanggal  : {current_date}")
        if show_time:   overlay.append(f"Jam      : {current_time}")
        overlay.append(f"Wajah    : {self.face_count}")

        y = img.shape[0] - 15
        for line in reversed(overlay):
            # Shadow hitam dulu agar terbaca di background apapun
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y -= 26

        with self.lock:
            self.last_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Webcam Stream ─────────────────────────────────────────────────────
with col_video:
    if camera_on:
        ctx = webrtc_streamer(
            key="face-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=FaceProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        ctx = None
        st.info("☝️ Aktifkan kamera dari sidebar untuk memulai deteksi.")


# ── Update info panel + snapshot ──────────────────────────────────────
if ctx and ctx.video_processor:
    proc = ctx.video_processor

    ph_fps.metric("FPS", proc.fps.fps if hasattr(proc.fps, "fps") else "-")
    ph_date.write(f"**Tanggal:** {get_indonesian_date()}")
    ph_time.write(f"**Jam:** {get_current_time()}")
    ph_faces.metric("Jumlah Wajah", proc.face_count)

    if snap_btn:
        with proc.lock:
            frame_copy = proc.last_frame.copy() if proc.last_frame is not None else None
        if frame_copy is not None:
            fname = timestamp_filename()
            fpath = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(fpath, frame_copy)
            snap_status.success(f"✅ Tersimpan: `{fpath}`")
        else:
            snap_status.warning("⚠️ Belum ada frame. Pastikan kamera aktif.")
else:
    ph_fps.write("**FPS:** —")
    ph_date.write(f"**Tanggal:** {get_indonesian_date()}")
    ph_time.write(f"**Jam:** {get_current_time()}")
    ph_faces.write("**Jumlah Wajah:** —")

st.markdown("---")
st.caption(
    "💡 **Tips:** Naikkan nilai 'Analisis Gender & Umur Tiap N Frame' di sidebar "
    "untuk meningkatkan FPS. Semakin besar nilainya, semakin ringan prosesnya."
)
