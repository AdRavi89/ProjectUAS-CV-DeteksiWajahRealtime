"""
utils.py
Fungsi-fungsi bantuan: penghitung FPS dan timestamp (tanggal & jam).
"""

import time
from datetime import datetime


class FPSCounter:
    """Penghitung FPS sederhana berbasis moving average."""

    def __init__(self, smoothing: int = 10):
        self.smoothing = smoothing
        self._last_time = time.time()
        self._fps = 0.0

    def update(self) -> float:
        """Panggil sekali per frame. Mengembalikan nilai FPS terkini."""
        now = time.time()
        delta = now - self._last_time
        self._last_time = now

        if delta > 0:
            current_fps = 1.0 / delta
            # Exponential moving average agar angka FPS tidak "loncat-loncat"
            alpha = 2 / (self.smoothing + 1)
            self._fps = (current_fps * alpha) + (self._fps * (1 - alpha))

        return round(self._fps, 1)

    @property
    def fps(self) -> float:
        return round(self._fps, 1)


def get_current_date(fmt: str = "%d %B %Y") -> str:
    """Mengembalikan tanggal hari ini, default format: 30 June 2026."""
    return datetime.now().strftime(fmt)


def get_current_time(fmt: str = "%H:%M:%S") -> str:
    """Mengembalikan jam saat ini, default format: 20:45:12."""
    return datetime.now().strftime(fmt)


def get_indonesian_date() -> str:
    """Mengembalikan tanggal dalam format Indonesia, mis: 30 Juni 2026."""
    bulan_indonesia = {
        "January": "Januari", "February": "Februari", "March": "Maret",
        "April": "April", "May": "Mei", "June": "Juni",
        "July": "Juli", "August": "Agustus", "September": "September",
        "October": "Oktober", "November": "November", "December": "Desember",
    }
    now = datetime.now()
    bulan_en = now.strftime("%B")
    bulan_id = bulan_indonesia.get(bulan_en, bulan_en)
    return f"{now.strftime('%d')} {bulan_id} {now.strftime('%Y')}"


def timestamp_filename(prefix: str = "hasil", ext: str = "jpg") -> str:
    """Membuat nama file unik berbasis timestamp, mis: hasil_20260630_204512.jpg"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}.{ext}"
