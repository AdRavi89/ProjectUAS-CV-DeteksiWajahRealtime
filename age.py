"""
age.py
Prediksi umur menggunakan DeepFace dengan strategi frame-skipping.
"""

import numpy as np
from deepface import DeepFace


class AgeAnalyzer:

    def __init__(self, analyze_every_n_frames: int = 10):
        self._every = analyze_every_n_frames
        self._counter = 0
        self._cache: dict = {}   # {face_id: int (umur)}

    # ------------------------------------------------------------------
    def _tick(self) -> bool:
        self._counter += 1
        if self._counter >= self._every:
            self._counter = 0
            return True
        return False

    # ------------------------------------------------------------------
    def analyze(self, face_img: np.ndarray):
        if face_img is None or face_img.size == 0:
            return None
        try:
            res = DeepFace.analyze(
                img_path=face_img,
                actions=["age"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            data = res[0] if isinstance(res, list) else res
            age = data.get("age")
            return int(age) if age is not None else None
        except Exception as e:
            print(f"[WARNING] Age: {e}")
            return None

    # ------------------------------------------------------------------
    def get(self, face_id: int, face_img: np.ndarray):
        if self._tick():
            age = self.analyze(face_img)
            if age is not None:
                self._cache[face_id] = age
        return self._cache.get(face_id, None)

    # ------------------------------------------------------------------
    @staticmethod
    def label(age) -> str:
        if age is None:
            return "..."
        return f"~{age} th"

    @staticmethod
    def age_group(age) -> str:
        if age is None:
            return "Unknown"
        if age < 13:  return "Anak-anak"
        if age < 20:  return "Remaja"
        if age < 60:  return "Dewasa"
        return "Lansia"
