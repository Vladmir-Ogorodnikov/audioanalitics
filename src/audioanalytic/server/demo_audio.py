"""Генератор тестового аудиофайла и демо-симулятор распознавания (для сред без GPU/WhisperX).

Позволяет проверить весь сквозной конвейер Спринта 1 в интерфейсе даже на машине без
установленного WhisperX/CUDA:
  1) создаёт настоящий WAV-файл с чередованием двух голосов (разные тембры/частоты) и пауз;
  2) прогоняет реальную предобработку ffmpeg (16 кГц, моно, −16 LUFS);
  3) сохраняет атомарные чекпоинты по шагам (FR-09);
  4) прогоняет реальную нормализацию меток «Спикер N» (FR-04), постобработку опечаток
     академических терминов и фильтр [НБЗ] (FR-03).
Если в системе установлен ``whisperx``, сервер по умолчанию вызывает боевой ``stt.transcribe``.
"""
from __future__ import annotations

import math
import struct
import time
import wave
from pathlib import Path
from typing import Callable

from ..checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from ..config import SttConfig
from ..lexicon import (
    apply_censorship,
    apply_terms_postprocessing,
    load_badwords,
    load_terms,
    term_coverage,
)
from ..models import Segment, Transcript, Word, normalize_speaker_labels

DEMO_LECTURE_TURNS = [
    (
        "SPEAKER_00",
        "Здравствуйте, коллеги. Сегодня мы разберём метод наименьших квадратав и задачи регрессии в машинном обучении.",
    ),
    (
        "SPEAKER_00",
        "В худшем случае при высокой размерности признаков возникает мультиколлинеарность и нарушается гомоскедастичнасть ошибок.",
    ),
    (
        "SPEAKER_01",
        "Скажите, пожалуйста, помогает ли регуляризация стабилизировать собственные значения матрицы и градиентный спуск?",
    ),
    (
        "SPEAKER_00",
        "Отличный вопрос! Да, регуляризация Тихонова уменьшает дисперсию оценок, а триганометрия и сингулярное разложение помогают при анализе.",
    ),
    (
        "SPEAKER_01",
        "Понятно, спасибо! Значит в доверительный интервал попадают все устойчивые векторные представления.",
    ),
    (
        "SPEAKER_00",
        "Именно так. На следующем семинаре проверим квантование моделей, транскрибацию и диаризацию на открытом корпусе.",
    ),
]


def whisperx_available() -> bool:
    try:
        import whisperx  # noqa: F401
        return True
    except Exception:
        return False


def generate_demo_lecture_wav(dst: str | Path, duration_sec: float = 36.0, sr: int = 22_050) -> Path:
    """Генерирует воспроизводимый WAV с речеподобной огибающей двух дикторов и паузами.

    Специально создаётся с частотой 22,05 кГц и пониженной громкостью, чтобы этап
    предобработки (FR-02) наглядно привёл его к 16 кГц, моно и −16 LUFS.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sr)
    seg_len = duration_sec / len(DEMO_LECTURE_TURNS)

    frames = bytearray()
    for i in range(n_samples):
        t = i / sr
        turn_idx = min(int(t // seg_len), len(DEMO_LECTURE_TURNS) - 1)
        local_t = t - turn_idx * seg_len
        active_dur = max(0.5, seg_len - 0.65)
        if local_t > active_dur:
            sample = 0.0
        else:
            spk, _ = DEMO_LECTURE_TURNS[turn_idx]
            f0 = 155.0 if spk == "SPEAKER_00" else 225.0
            syllable_env = 0.55 + 0.45 * math.sin(2 * math.pi * 3.8 * local_t)
            fade = min(1.0, local_t / 0.06, (active_dur - local_t) / 0.08)
            sig = (
                0.60 * math.sin(2 * math.pi * f0 * t)
                + 0.28 * math.sin(2 * math.pi * (2 * f0) * t)
                + 0.12 * math.sin(2 * math.pi * (3 * f0) * t)
            )
            sample = 0.18 * fade * syllable_env * sig
        val = max(-32767, min(32767, int(sample * 32767)))
        frames += struct.pack("<h", val)

    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return out


def simulate_transcription(
    audio_path: str | Path,
    cfg: SttConfig,
    progress: Callable[[str], None] | None = None,
) -> Transcript:
    """Выполняет полный цикл обработки с реальным ffmpeg и чекпоинтами, симулируя только нейросеть WhisperX."""
    say = progress or (lambda _m: None)
    orig_name = Path(audio_path).name
    work_path = Path(audio_path)

    ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    ckpt = load_checkpoint(ckpt_path) if ckpt_path else None
    if ckpt is None:
        ckpt = CheckpointState(
            stage="init",
            source_file=str(audio_path),
            chunk_duration=float(cfg.chunk_duration or 300.0),
        )

    def _save(stage: str, note: str) -> None:
        ckpt.record_step(stage, note=note)
        if ckpt_path is not None:
            save_checkpoint(ckpt_path, ckpt)

    # 1. Реальная предобработка аудио через ffmpeg (-16 LUFS, 16 кГц, моно)
    if cfg.preprocess:
        if ckpt.preprocessed_path and Path(ckpt.preprocessed_path).is_file():
            work_path = Path(ckpt.preprocessed_path)
            say("Предобработка подхвачена из чекпоинта")
        else:
            try:
                from ..preprocess import preprocess
                say("Предобработка аудио (16 кГц, моно, −16 LUFS)")
                work_path = preprocess(work_path, dst=cfg.preprocessed_path)
                ckpt.preprocessed_path = str(work_path)
                _save("preprocessed", "16 кГц, моно, −16 LUFS")
            except Exception as e:
                say(f"Предупреждение предобработки: {e} — использую исходный файл")

    from ..preprocess import probe_audio
    try:
        info = probe_audio(work_path)
        duration = max(6.0, float(info.get("duration") or 36.0))
    except Exception:
        duration = 36.0
    ckpt.duration_sec = duration

    step = float(cfg.chunk_duration or 300.0)
    n_chunks = max(1, int(-(-duration // step))) if step > 0 else 1
    ckpt.total_chunks = n_chunks

    say("Демо-режим ASR (WhisperX не установлен в текущем контейнере): разметка сегментов")
    time.sleep(0.15)

    n_turns = len(DEMO_LECTURE_TURNS)
    slot = duration / n_turns
    raw_segments: list[dict] = []
    for idx, (spk, text) in enumerate(DEMO_LECTURE_TURNS):
        seg_start = round(idx * slot, 2)
        seg_end = round(min(duration, (idx + 1) * slot - 0.55), 2)
        if seg_end <= seg_start:
            seg_end = round(seg_start + max(0.4, slot * 0.85), 2)
        toks = text.split()
        w_dur = (seg_end - seg_start) / max(len(toks), 1)
        words = []
        for wi, tok in enumerate(toks):
            ws = round(seg_start + wi * w_dur, 2)
            we = round(min(seg_end, ws + w_dur * 0.92), 2)
            words.append({"word": tok, "start": ws, "end": we, "score": 0.94})
        raw_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "speaker": spk if cfg.diarize else "SPEAKER_??",
            "words": words,
        })

    for ch_i in range(ckpt.completed_chunks, n_chunks):
        ckpt.completed_chunks = ch_i + 1
        ckpt.segments = list(raw_segments)
        _save("asr_partial", f"Чекпоинт части {ch_i + 1}/{n_chunks}")

    _save("asr_done", f"ASR завершён: {len(raw_segments)} сегментов")
    say("Выравнивание по словам (±0,1 с)")
    time.sleep(0.1)
    _save("aligned", "Выравнивание по словам завершено")

    if cfg.diarize:
        say("Диаризация спикеров")
        time.sleep(0.1)
        if cfg.num_speakers == 1:
            for s in raw_segments:
                s["speaker"] = "SPEAKER_00"
        _save("diarized", "Диаризация завершена")

    segments_obj = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s["speaker"],
            words=[Word(text=w["word"], start=w["start"], end=w["end"], score=w["score"]) for w in s["words"]],
        )
        for s in raw_segments
    ]
    tr = Transcript(language=cfg.language, segments=segments_obj, source=orig_name)
    tr.meta["engine"] = "demo-simulator"
    tr.meta["duration_sec"] = round(duration, 2)
    tr.meta["preprocessed"] = bool(cfg.preprocess)

    if cfg.normalize_speakers:
        normalize_speaker_labels(tr)

    terms = load_terms(cfg.terms_file) if cfg.terms_file else []
    if terms and cfg.postprocess_terms:
        corrections = apply_terms_postprocessing(tr.segments, terms)
        if corrections:
            tr.meta["term_corrections"] = corrections
            say(f"Словарь терминов: автоисправлено опечаток — {len(corrections)}")

    if cfg.censor:
        roots = load_badwords(cfg.badwords_file)
        tr.meta["censored_count"] = apply_censorship(tr.segments, roots)

    if terms:
        full_text = " ".join(s.text for s in tr.segments)
        tr.meta["term_coverage"] = term_coverage(full_text, terms)

    _save("completed", "Обработка успешно завершена")
    tr.meta["checkpoint_history"] = ckpt.history
    say("Готово")
    return tr
