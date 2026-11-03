"""Настройки конвейера распознавания речи."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный загрузчик .env (без зависимостей): HF_TOKEN и т.п.

    Читает файл вида KEY=VALUE (строки с '#' и пустые пропускаются), значения в
    кавычках снимаются. Уже установленные переменные окружения НЕ перезаписывает:
    приоритет у `set HF_TOKEN=...` / системных переменных Windows.
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and val and key not in os.environ:
            os.environ[key] = val


@dataclass
class SttConfig:
    language: str = "ru"
    # Модель Whisper. large-v3 — максимум качества (RTX 3060 12 ГБ).
    model: str = "large-v3"
    # float16 — GPU с 12 ГБ; int8_float16 — экономия VRAM; int8 — CPU.
    compute_type: str = "float16"
    batch_size: int = 16
    device: str = "auto"  # auto | cuda | cpu
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    @classmethod
    def low_vram(cls, **kw) -> "SttConfig":
        """Профиль для GPU с 8 ГБ (CON-03): модель поменьше, int8, меньший батч."""
        base = dict(model="large-v3-turbo", compute_type="int8_float16", batch_size=8)
        base.update(kw)
        return cls(**base)
