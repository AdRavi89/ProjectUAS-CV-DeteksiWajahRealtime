"""
gender.py
Analisis gender menggunakan DeepFace dengan strategi frame-skipping.
"""

import numpy as np
from deepface import DeepFace


class GenderAnalyzer:

    def __init__(self, analyze_every_n_frames: int = 10):
        self._every = analyze_every_n_frames
        self._counter = 0
        self._cache: dict = {}   # {face_id: (gender, confidence)}

    # ------------------------------------------------------------------
    def _tick(self) -> bool:
        """Kembalikan True jika saatnya analisis ulang."""
        self._counter += 1
        if self._counter >= self._every:
            self._counter = 0
            return True
        return False

    # ------------------------------------------------------------------
    def analyze(self, face_img: np.ndarray):
        if face_img is None or face_img.size == 0:
            return None, None
        try:
            res = DeepFace.analyze(
                img_path=face_img,
                actions=["gender"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            data = res[0] if isinstance(res, list) else res
            scores = data.get("gender", {})
            if not scores:
                return None, None
            best = max(scores, key=scores.get)
            return best, round(scores[best], 1)
        except Exception as e:
            print(f"[WARNING] Gender: {e}")
            return None, None

    # ------------------------------------------------------------------
    def get(self, face_id: int, face_img: np.ndarray):
        if self._tick():
            g, c = self.analyze(face_img)
            if g is not None:
                self._cache[face_id] = (g, c)
        return self._cache.get(face_id, (None, None))

    # ------------------------------------------------------------------
    @staticmethod
    def label(gender, confidence) -> str:
        if gender is None:
            return "Detecting..."
        label_id = "Pria" if gender.lower() == "man" else "Wanita"
        return f"{label_id} ({confidence:.0f}%)"
