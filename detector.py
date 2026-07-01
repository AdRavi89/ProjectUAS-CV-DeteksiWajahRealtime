"""
detector.py
Deteksi wajah menggunakan YOLOv8 (ultralytics).

Model khusus wajah: letakkan 'yolov8n-face.pt' di folder models/.
Jika tidak ada, otomatis fallback ke 'yolov8n.pt' (model umum).
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_FACE   = os.path.join("models", "yolov8n-face.pt")
MODEL_GENERAL = "yolov8n.pt"   # diunduh otomatis oleh ultralytics


class FaceDetector:

    def __init__(self, conf_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        path = MODEL_FACE if os.path.exists(MODEL_FACE) else MODEL_GENERAL
        if path == MODEL_GENERAL:
            print(f"[INFO] Model wajah tidak ditemukan, menggunakan fallback: {MODEL_GENERAL}")
        self.model = YOLO(path)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list:
        """
        Deteksi wajah pada satu frame BGR.
        Returns: [{"box": (x1,y1,x2,y2), "confidence": float}, ...]
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        faces = []
        if not results:
            return faces
        boxes = results[0].boxes
        if boxes is None:
            return faces
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            faces.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(b.conf[0]),
            })
        return faces

    # ------------------------------------------------------------------
    @staticmethod
    def crop_face(frame: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    @staticmethod
    def draw_box(
        frame: np.ndarray,
        box: tuple,
        label: str = "",
        color: tuple = (50, 180, 50),
        thickness: int = 2,
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
