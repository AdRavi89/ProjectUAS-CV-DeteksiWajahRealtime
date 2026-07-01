"""
utils.py
Fungsi-fungsi bantuan: FPS counter, timestamp (tanggal & jam),
dan helper nama file snapshot.
"""

import time
from datetime import datetime


class FPSCounter:
    """Penghitung FPS berbasis exponential moving average."""

    def __init__(self, smoothing: int = 10):
        self.smoothing = smoothing
        self._last_time = time.time()
        self._fps = 0.0

    def update(self) -> float:
        now = time.time()
        delta = now - self._last_time
        self._last_time = now
        if delta > 0:
            alpha = 2 / (self.smoothing + 1)
            self._fps = (1.0 / delta) * alpha + self._fps * (1 - alpha)
        return round(self._fps, 1)

    @property
    def fps(self) -> float:
        return round(self._fps, 1)


def get_indonesian_date() -> str:
    """Contoh: '30 Juni 2026'"""
    bulan = {
        "January": "Januari",   "February": "Februari", "March": "Maret",
        "April": "April",       "May": "Mei",            "June": "Juni",
        "July": "Juli",         "August": "Agustus",     "September": "September",
        "October": "Oktober",   "November": "November",  "December": "Desember",
    }
    now = datetime.now()
    return f"{now.day:02d} {bulan[now.strftime('%B')]} {now.year}"


def get_current_time() -> str:
    """Contoh: '20:45:12'"""
    return datetime.now().strftime("%H:%M:%S")


def timestamp_filename(prefix: str = "hasil", ext: str = "jpg") -> str:
    """Contoh: 'hasil_20260630_204512.jpg'"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
