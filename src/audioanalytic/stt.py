"""Распознавание речи: WhisperX (ASR) -> выравнивание по словам -> диаризация.

Этапы выполняются последовательно, модели выгружаются из памяти GPU между
этапами, чтобы пик VRAM оставался низким (см. конфликт №3 в ТЗ).
whisperx/torch импортируются лениво, чтобы остальной код работал без них.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable

from .config import SttConfig
from .models import Segment, Transcript, Word, UNKNOWN_SPEAKER

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _to_transcript(result: dict, language: str, source: str) -> Transcript:
    segments = []
    for s in result["segments"]:
        words = [
            Word(text=w["word"], start=w.get("start"), end=w.get("end"), score=w.get("score"))
            for w in s.get("words", [])
        ]
        segments.append(Segment(
            start=float(s["start"]), end=float(s["end"]), text=s["text"].strip(),
            speaker=s.get("speaker", UNKNOWN_SPEAKER), words=words,
        ))
    return Transcript(language=language, segments=segments, source=source)


def transcribe(audio_path: str | Path, cfg: SttConfig | None = None,
               progress: Progress | None = None) -> Transcript:
    cfg = cfg or SttConfig()
    say = progress or (lambda m: log.info(m))
    import whisperx

    device = _resolve_device(cfg.device)
    compute_type = cfg.compute_type if device == "cuda" else "int8"
    audio_path = str(audio_path)

    say("Загрузка аудио")
    audio = whisperx.load_audio(audio_path)  # требует ffmpeg в PATH

    say(f"Распознавание речи ({cfg.model}, {device}, {compute_type})")
    model = whisperx.load_model(cfg.model, device, compute_type=compute_type, language=cfg.language)
    result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language)
    del model
    _free_memory()

    say("Выравнивание по словам")
    align_model, meta = whisperx.load_align_model(language_code=cfg.language, device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device,
                            return_char_alignments=False)
    del align_model
    _free_memory()

    if cfg.diarize:
        if not cfg.hf_token:
            raise RuntimeError(
                "Для диаризации нужен токен Hugging Face (переменная HF_TOKEN) и принятые "
                "условия модели pyannote/speaker-diarization-community-1. "
                "Либо запустите с --no-diarize."
            )
        say("Диаризация спикеров")
        from whisperx.diarize import DiarizationPipeline
        pipe = DiarizationPipeline(token=cfg.hf_token, device=device)
        diar = pipe(audio, num_speakers=cfg.num_speakers,
                    min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers)
        result = whisperx.assign_word_speakers(diar, result)
        del pipe
        _free_memory()

    say("Готово")
    return _to_transcript(result, cfg.language, Path(audio_path).name)
