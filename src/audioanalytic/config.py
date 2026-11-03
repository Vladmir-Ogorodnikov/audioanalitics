"""Настройки конвейера распознавания речи."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


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
