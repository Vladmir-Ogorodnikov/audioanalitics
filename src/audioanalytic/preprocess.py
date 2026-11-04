"""Предобработка аудио (FR-02): 16 кГц, моно, нормализация громкости до −16 LUFS.

Реализовано через ffmpeg/ffprobe без тяжёлых зависимостей:
  1) resample + downmix      -> tmp.wav (pcm_s16le, 16 кГц, моно)
  2) loudnorm в два прохода  -> финальный файл с целевой громкостью −16 LUFS
     (первый проход измеряет loudness, второй применяет измеренные параметры —
     так результат стабильнее однопассового варианта).
Если первый проход не дал валидных измерений (например, полностью тихий файл),
выполняется однопроходная нормализация.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

TARGET_LUFS = -16.0
TARGET_TP = -1.5   # true peak, дБ
TARGET_LRA = 11.0  # loudness range

SAMPLE_RATE = 16_000


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg завершился с кодом {proc.returncode}:\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


def normalize_loudness(src: Path, dst: Path, *, ffmpeg: str = "ffmpeg") -> Path:
    """Конвертирует src в 16 кГц моно и нормализует громкость dst до TARGET_LUFS."""
    base_cmd = [ffmpeg, "-y", "-i", str(src)]
    filters = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        ":print_format=json"
    )

    # Проход 1: измерение (результат пишется в stderr у loudnorm)
    probe_cmd = base_cmd + ["-af", filters, "-vn", "-f", "null", "-"]
    try:
        proc = subprocess.run(probe_cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        stderr = proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(stderr[-2000:])
    except Exception:
        # однопроходная нормализация как запасной вариант
        _run(base_cmd + ["-af", filters.replace(":print_format=json", ""),
                         "-ar", str(SAMPLE_RATE), "-ac", "1",
                         "-c:a", "pcm_s16le", str(dst)])
        return dst

    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
    if not m:
        _run(base_cmd + ["-af", filters.replace(":print_format=json", ""),
                         "-ar", str(SAMPLE_RATE), "-ac", "1",
                         "-c:a", "pcm_s16le", str(dst)])
        return dst
    meas = json.loads(m.group(0))
    if float(meas.get("input_i", 0)) <= -70.0:  # практически тишина
        meas["target_low"] = None  # не мешаем loudnorm работать «в лоб»

    second = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_input_i={meas['input_i']}"
        f":measured_input_tp={meas['input_tp']}"
        f":measured_input_lra={meas['input_lra']}"
        f":measured_input_thresh={meas['input_thresh']}"
        f":linear=true:measured_target_lra={meas.get('target_lra', TARGET_LRA)}"
    )
    _run(base_cmd + ["-af", second, "-ar", str(SAMPLE_RATE), "-ac", "1",
                     "-c:a", "pcm_s16le", str(dst)])
    return dst


def probe_wav(path: Path, *, ffprobe: str = "ffprobe") -> dict:
    """Возвращает {sample_rate, channels, duration} WAV-файла."""
    out = _run([ffprobe, "-v", "error", "-show_entries",
                "format=duration:stream=sample_rate,channels",
                "-of", "json", str(path)])
    data = json.loads(out)
    st = data["streams"][0]
    return {
        "sample_rate": int(st["sample_rate"]),
        "channels": int(st["channels"]),
        "duration": float(data.get("format", {}).get("duration", 0)),
    }


def preprocess(src: str | Path, dst: str | Path | None = None,
               *, ffmpeg: str | None = None) -> Path:
    """Готовит файл для STT-конвейера. Возвращает путь к подготовленному WAV.

    dst по умолчанию: рядом с исходником, `<имя>.16k.wav`.
    """
    src = Path(src)
    if ffmpeg is None:
        from .winfix import find_ffmpeg
        ffmpeg = find_ffmpeg() or "ffmpeg"
    dst = Path(dst) if dst else src.with_name(src.stem + ".16k.wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    normalize_loudness(src, dst, ffmpeg=ffmpeg)
    return dst
