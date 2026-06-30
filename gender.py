"""
gender.py
Logika analisis gender menggunakan DeepFace, dengan strategi
frame-skipping agar performa realtime tetap terjaga.
"""

import numpy as np
from deepface import DeepFace


class GenderAnalyzer:
    """
    Melakukan analisis gender pada wajah yang sudah di-crop.

    Karena DeepFace cukup berat, gunakan `should_analyze()` untuk
    membatasi frekuensi analisis (mis. 1x setiap N frame), dan simpan
    hasil terakhir per wajah agar label tetap tampil di frame lain.
    """

    def __init__(self, analyze_every_n_frames: int = 10):
        self.analyze_every_n_frames = analyze_every_n_frames
        self._frame_counter = 0
        # Cache hasil terakhir berdasarkan index wajah pada frame
        # (mis. {0: ("Male", 98.2), 1: ("Female", 95.6)})
        self._last_results = {}

    def should_analyze(self) -> bool:
        """Menentukan apakah frame saat ini perlu dianalisis penuh."""
        self._frame_counter += 1
        if self._frame_counter >= self.analyze_every_n_frames:
            self._frame_counter = 0
            return True
        return False

    def analyze(self, face_img: np.ndarray):
        """
        Menganalisis gender dari satu gambar wajah (hasil crop).

        Returns:
            (gender_label: str, confidence: float) atau (None, None) jika gagal.
        """
        if face_img is None or face_img.size == 0:
            return None, None

        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["gender"],
                enforce_detection=False,
                detector_backend="skip",  # wajah sudah di-crop oleh YOLO
                silent=True,
            )

            # DeepFace bisa mengembalikan list atau dict tergantung versi
            data = result[0] if isinstance(result, list) else result

            gender_scores = data.get("gender", {})
            if not gender_scores:
                return None, None

            best_gender = max(gender_scores, key=gender_scores.get)
            confidence = gender_scores[best_gender]
            return best_gender, round(confidence, 1)

        except Exception as e:
            print(f"[WARNING] Gagal menganalisis gender: {e}")
            return None, None

    def get_cached_or_analyze(self, face_id: int, face_img: np.ndarray):
        """
        Mengembalikan hasil cache untuk face_id tertentu, atau
        menjalankan analisis baru jika sudah waktunya (frame skipping).
        """
        if self.should_analyze():
            gender, confidence = self.analyze(face_img)
            if gender is not None:
                self._last_results[face_id] = (gender, confidence)

        return self._last_results.get(face_id, (None, None))

    @staticmethod
    def format_label(gender: str, confidence: float) -> str:
        """Memformat label tampilan, mis: 'Male (98%)'."""
        if gender is None:
            return "Detecting..."
        return f"{gender} ({confidence:.0f}%)"
