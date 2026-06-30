"""
detector.py
Logika deteksi wajah menggunakan YOLOv8 (ultralytics).
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.path.join("models", "yolov8n-face.pt")
# Fallback ke model deteksi objek umum bila model wajah khusus belum diunduh.
# Catatan: untuk hasil akurat, unduh model 'yolov8n-face.pt' dari repo
# pihak ketiga (mis. akanametov/yolov8-face) dan letakkan di folder models/.
FALLBACK_MODEL_PATH = "yolov8n.pt"


class FaceDetector:
    """Wrapper YOLOv8 khusus untuk deteksi wajah."""

    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = 0.45):
        self.conf_threshold = conf_threshold

        path_to_load = model_path if os.path.exists(model_path) else FALLBACK_MODEL_PATH
        if path_to_load == FALLBACK_MODEL_PATH:
            print(
                f"[WARNING] Model wajah '{model_path}' tidak ditemukan. "
                f"Menggunakan fallback '{FALLBACK_MODEL_PATH}' (deteksi objek umum, "
                f"bukan model khusus wajah). Silakan unduh model wajah untuk hasil terbaik."
            )

        self.model = YOLO(path_to_load)

    def detect(self, frame: np.ndarray):
        """
        Mendeteksi wajah pada satu frame.

        Returns:
            List of dict: [{"box": (x1, y1, x2, y2), "confidence": float}, ...]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
        )

        faces = []
        if len(results) == 0:
            return faces

        boxes = results[0].boxes
        if boxes is None:
            return faces

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            faces.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
            })

        return faces

    @staticmethod
    def crop_face(frame: np.ndarray, box) -> np.ndarray:
        """Memotong area wajah dari frame berdasarkan bounding box."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def draw_box(frame: np.ndarray, box, label: str = "", color=(60, 145, 230), thickness: int = 2):
        """Menggambar bounding box beserta label di atas frame."""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if label:
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 12), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

        return frame
