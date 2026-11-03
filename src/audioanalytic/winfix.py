"""Мелочи, без которых конвейер падает именно на Windows.

1) Консоль Windows (cp1251/cp866) может отказаться печатать юникод
   -> UnicodeEncodeError. Включаем UTF-8 для stdout/stderr.
2) WhisperX при диаризации вызывает torch.load, а чекпойнт pyannote хранит
   объекты вроде ``pyannote.core.segment.Segmentation``. Начиная с torch 2.6
   это приводит к UnpicklingError — разрешаем загрузку таких весов.
3) Проверка ffmpeg: понятная подсказка вместо непонятной ошибки внутри whisperx.
"""
from __future__ import annotations

import shutil
import sys


def setup_console() -> None:
    """UTF-8 в консоли Windows (Python 3.7+). Безопасно и на Linux/macOS."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # уже переопределено / не tty
                pass


def allow_pyannote_unpickling() -> None:
    """Разрешает torch.load грузить чекпойнты pyannote (WhisperX + torch>=2.6).

    Без этого диаризация падает с UnpicklingError: в весах pyannote лежат
    собственные объекты (Segmentation и т.п.), которые безопасная загрузка
    по умолчанию блокирует.
    """
    try:
        import torch
    except ImportError:
        return
    # pyannote должен быть импортирован до десериализации весов
    try:
        import pyannote.core  # noqa: F401
        import pyannote.audio  # noqa: F401
    except Exception:
        pass
    if getattr(torch.load, "_audioanalytic_patched", False):
        return
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    patched._audioanalytic_patched = True  # type: ignore[attr-defined]
    torch.load = patched  # type: ignore[assignment]


def check_ffmpeg() -> None:
    """ffmpeg должен быть в PATH — иначе whisperx.load_audio падает без внятной ошибки."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установите его, например: winget install Gyan.FFmpeg "
            "(после установки перезапустите терминал), либо скачайте сборку с ffmpeg.org "
            "и добавьте папку bin\\ в переменную PATH."
        )
