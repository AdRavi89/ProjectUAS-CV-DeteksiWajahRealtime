"""
age.py
Logika prediksi umur (age estimation) menggunakan DeepFace, dengan
strategi frame-skipping yang sama seperti gender.py agar performa
realtime tetap terjaga.
"""

import numpy as np
from deepface import DeepFace


class AgeAnalyzer:
    """
    Memprediksi estimasi umur pada wajah yang sudah di-crop.

    Sama seperti GenderAnalyzer, gunakan `should_analyze()` untuk
    membatasi frekuensi analisis (mis. 1x setiap N frame), dan simpan
    hasil terakhir per wajah agar label tetap tampil di frame lain.
    """

    def __init__(self, analyze_every_n_frames: int = 10):
        self.analyze_every_n_frames = analyze_every_n_frames
        self._frame_counter = 0
        # Cache hasil terakhir berdasarkan index wajah pada frame
        # (mis. {0: 24, 1: 31})
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
        Memprediksi umur dari satu gambar wajah (hasil crop).

        Returns:
            age: int (estimasi umur dalam tahun) atau None jika gagal.
        """
        if face_img is None or face_img.size == 0:
            return None

        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["age"],
                enforce_detection=False,
                detector_backend="skip",  # wajah sudah di-crop oleh YOLO
                silent=True,
            )

            # DeepFace bisa mengembalikan list atau dict tergantung versi
            data = result[0] if isinstance(result, list) else result
            age = data.get("age")
            return int(age) if age is not None else None

        except Exception as e:
            print(f"[WARNING] Gagal memprediksi umur: {e}")
            return None

    def get_cached_or_analyze(self, face_id: int, face_img: np.ndarray):
        """
        Mengembalikan hasil cache untuk face_id tertentu, atau
        menjalankan analisis baru jika sudah waktunya (frame skipping).
        """
        if self.should_analyze():
            age = self.analyze(face_img)
            if age is not None:
                self._last_results[face_id] = age

        return self._last_results.get(face_id, None)

    @staticmethod
    def format_label(age) -> str:
        """Memformat label tampilan, mis: '~24 th'."""
        if age is None:
            return "Detecting..."
        return f"~{age} th"

    @staticmethod
    def age_group(age) -> str:
        """Mengelompokkan umur ke kategori sederhana, berguna untuk log/analytics."""
        if age is None:
            return "Unknown"
        if age < 13:
            return "Anak-anak"
        if age < 20:
            return "Remaja"
        if age < 60:
            return "Dewasa"
        return "Lansia"
