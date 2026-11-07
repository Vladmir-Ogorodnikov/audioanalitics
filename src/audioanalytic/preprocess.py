"""Предобработка аудио (FR-02): 16 кГц, моно, нормализация громкости до −16 LUFS.

Реализовано через ffmpeg без тяжёлых зависимостей:
  1) измерение loudness первым проходом loudnorm;
  2) конвертация в WAV (pcm_s16le, 16 кГц, моно) со вторым проходом loudnorm (−16 LUFS).
Если первый проход не дал валидных измерений (например, тихий файл),
выполняется однопроходная нормализация.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
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

    probe_cmd = base_cmd + ["-af", filters, "-vn", "-f", "null", "-"]
    try:
        proc = subprocess.run(
            probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        stderr = proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(stderr[-2000:])
    except Exception:
        _run(
            base_cmd
            + [
                "-af", filters.replace(":print_format=json", ""),
                "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", str(dst),
            ]
        )
        return dst

    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr)
    if not m:
        _run(
            base_cmd
            + [
                "-af", filters.replace(":print_format=json", ""),
                "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", str(dst),
            ]
        )
        return dst
    meas = json.loads(m.group(0))

    def _f(key: str, default: float) -> float:
        try:
            v = float(meas.get(key, default))
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v

    input_i = _f("input_i", 0.0)
    if input_i <= -70.0:
        _run(
            base_cmd
            + [
                "-af", filters.replace(":print_format=json", ""),
                "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-c:a", "pcm_s16le", str(dst),
            ]
        )
        return dst

    second = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={input_i:.2f}"
        f":measured_TP={_f('input_tp', -1.5):.2f}"
        f":measured_LRA={_f('input_lra', TARGET_LRA):.2f}"
        f":measured_thresh={_f('input_thresh', -70.0):.2f}"
        ":linear=true"
        f":offset={_f('target_offset', 0.0):.2f}"
    )
    _run(
        base_cmd
        + [
            "-af", second, "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_s16le", str(dst),
        ]
    )
    return dst


def probe_audio(path: str | Path, *, ffprobe: str | None = None, ffmpeg: str | None = None) -> dict:
    """Возвращает ``{sample_rate, channels, duration}`` аудиофайла (WAV/MP3/FLAC).

    Не требует обязательного наличия ``ffprobe``: для WAV использует встроенный
    модуль ``wave``, а для MP3/FLAC при отсутствии ``ffprobe`` разбирает вывод ``ffmpeg -i``.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                n = w.getnframes()
                return {
                    "sample_rate": int(sr),
                    "channels": int(ch),
                    "duration": float(n / sr) if sr else 0.0,
                }
        except Exception:
            pass

    try:
        import soundfile as sf
        info = sf.info(str(p))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration": float(info.duration),
        }
    except Exception:
        pass

    ffp = ffprobe or shutil.which("ffprobe")
    if ffp:
        try:
            out = _run([
                ffp, "-v", "error",
                "-show_entries", "format=duration:stream=sample_rate,channels",
                "-of", "json", str(p),
            ])
            data = json.loads(out)
            st = data["streams"][0]
            return {
                "sample_rate": int(st["sample_rate"]),
                "channels": int(st["channels"]),
                "duration": float(data.get("format", {}).get("duration", 0)),
            }
        except Exception:
            pass

    if ffmpeg is None:
        from .winfix import find_ffmpeg
        ffmpeg = find_ffmpeg() or "ffmpeg"
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(p)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    stderr = proc.stderr
    dur = 0.0
    m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if m_dur:
        dur = int(m_dur.group(1)) * 3600 + int(m_dur.group(2)) * 60 + float(m_dur.group(3))
    sr = 0
    m_sr = re.search(r"(\d{4,6})\s*Hz", stderr)
    if m_sr:
        sr = int(m_sr.group(1))
    ch = 1 if "mono" in stderr.lower() else (2 if "stereo" in stderr.lower() else 1)
    return {"sample_rate": sr, "channels": ch, "duration": dur}


def probe_wav(path: Path, *, ffprobe: str = "ffprobe") -> dict:
    """Обратная совместимость: возвращает {sample_rate, channels, duration}."""
    return probe_audio(path, ffprobe=ffprobe if shutil.which(ffprobe) else None)


def preprocess(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Готовит файл для STT-конвейера (16 кГц, моно, −16 LUFS). Возвращает путь к WAV."""
    src = Path(src)
    if ffmpeg is None:
        from .winfix import find_ffmpeg
        ffmpeg = find_ffmpeg() or "ffmpeg"
    dst = Path(dst) if dst else src.with_name(src.stem + ".16k.wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    normalize_loudness(src, dst, ffmpeg=ffmpeg)
    return dst
