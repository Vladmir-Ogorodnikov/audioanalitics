"""Распознавание речи: WhisperX (ASR) -> выравнивание по словам -> диаризация.

Этапы выполняются последовательно, модели выгружаются из памяти GPU между
этапами, чтобы пик VRAM оставался низким (CON-03, риск №2).
Поддерживаются атомарные чекпоинты каждые 5 мин (chunk_duration=300 с, FR-09)
с автоматическим возобновлением с последнего сохранённого куска после сбоя.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Callable

from .checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from .config import SttConfig
from .models import (
    UNKNOWN_SPEAKER,
    Segment,
    Transcript,
    Word,
    normalize_speaker_labels,
)

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
            Word(
                text=str(w.get("word", w.get("text", ""))),
                start=w.get("start"),
                end=w.get("end"),
                score=w.get("score"),
            )
            for w in s.get("words", [])
        ]
        segments.append(
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=str(s["text"]).strip(),
                speaker=s.get("speaker", UNKNOWN_SPEAKER),
                words=words,
            )
        )
    return Transcript(language=language, segments=segments, source=source)


def _torchcodec_ok() -> bool:
    try:
        import torchcodec.decoders  # noqa: F401
        return True
    except Exception:
        return False


def _load_audio_fallback(audio_path: str, sr: int = 16_000):
    """Читает WAV в обход torchcodec через soundfile, numpy или stdlib wave."""
    try:
        import numpy as np
        import soundfile as sf
        data, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if data.shape[1] > 1:
            data = data.mean(axis=1, keepdims=True)
        if rate != sr:
            raise RuntimeError(f"sample rate {rate} != {sr}")
        arr = np.ascontiguousarray(data[:, 0])
        try:
            import torch  # noqa: F401
            # whisperx ожидает 1D float32 numpy array (как возвращает whisperx.load_audio)
            return arr
        except ImportError:
            return arr
    except ImportError:
        pass
    import array
    import wave
    with wave.open(audio_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        data = array.array("h", raw)
        mono = [s / 32768.0 for s in data]
        try:
            import numpy as np
            return np.asarray(mono, dtype="float32")
        except ImportError:
            import torch
            return torch.tensor(mono, dtype=torch.float32)


def _load_audio(audio_path: str, ffmpeg_file: str | None, say: Progress | None = None) -> object:
    import whisperx

    kwargs = {}
    if ffmpeg_file and "ffmpeg_file" in _load_audio_kwargs_names():
        kwargs["ffmpeg_file"] = ffmpeg_file
    try:
        return whisperx.load_audio(audio_path, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "torchcodec" not in msg and "libtorchcodec" not in msg:
            raise
        if say:
            say("torchcodec не работает — читаю подготовленный WAV напрямую")
        return _load_audio_fallback(audio_path)


def _load_audio_kwargs_names() -> set[str]:
    import inspect
    import whisperx
    try:
        return set(inspect.signature(whisperx.load_audio).parameters)
    except (TypeError, ValueError):
        return set()


_STAGE_ORDER = {
    "init": 0,
    "preprocessed": 1,
    "asr_partial": 2,
    "asr_done": 3,
    "aligned": 4,
    "diarized": 5,
    "completed": 6,
}


def transcribe(
    audio_path: str | Path,
    cfg: SttConfig | None = None,
    progress: Progress | None = None,
) -> Transcript:
    cfg = cfg or SttConfig()
    say = progress or (lambda m: log.info(m))
    import whisperx

    orig_source_name = Path(audio_path).name
    ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
    ckpt = load_checkpoint(ckpt_path) if ckpt_path else None
    if ckpt is None:
        ckpt = CheckpointState(
            stage="init",
            source_file=str(audio_path),
            chunk_duration=float(cfg.chunk_duration or 300.0),
        )
    else:
        say(
            f"Найден чекпоинт (этап: {ckpt.stage}, готово частей: "
            f"{ckpt.completed_chunks}/{ckpt.total_chunks}) — возобновляю обработку"
        )

    def _save_ckpt(stage: str, note: str = "") -> None:
        ckpt.record_step(stage, note=note)
        if ckpt_path is not None:
            save_checkpoint(ckpt_path, ckpt)

    device = _resolve_device(cfg.device)
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            say("CUDA не доступна — переключаюсь на CPU")
            device = "cpu"
    if device == "cuda":
        compute_type = cfg.compute_type
    else:
        if cfg.compute_type != "int8":
            say(f"CPU: compute_type '{cfg.compute_type}' заменён на 'int8'")
        compute_type = "int8"

    work_audio_path = str(audio_path)

    # 1. FR-02: предобработка (с подхватом из чекпоинта, если уже выполнена)
    if cfg.preprocess:
        if (
            _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["preprocessed"]
            and ckpt.preprocessed_path
            and Path(ckpt.preprocessed_path).is_file()
        ):
            work_audio_path = ckpt.preprocessed_path
            say("Предобработка пропущена — используется файл из чекпоинта")
        else:
            say("Предобработка аудио (16 кГц, моно, −16 LUFS)")
            from .preprocess import preprocess
            prepared = preprocess(work_audio_path, dst=cfg.preprocessed_path)
            work_audio_path = str(prepared)
            ckpt.preprocessed_path = work_audio_path
            _save_ckpt("preprocessed", "16 кГц, моно, −16 LUFS")

    say("Загрузка аудио")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    exe = None
    try:
        from .winfix import find_ffmpeg
        exe = find_ffmpeg()
    except Exception:
        pass
    if not _torchcodec_ok():
        say(
            "torchcodec не установлен корректно — встроенное декодирование "
            "будет заменено чтением WAV напрямую (через soundfile/ffmpeg)"
        )
    audio = _load_audio(work_audio_path, exe, say)
    if cfg.max_duration:
        n = int(float(cfg.max_duration) * 16_000)
        if hasattr(audio, "shape") and audio.shape[-1] > n:
            say(f"Обрезаю аудио до {cfg.max_duration:g} с (--max-duration)")
            audio = audio[..., :n]

    try:
        duration = float(audio.shape[-1]) / 16_000
    except Exception:
        duration = 0.0
    ckpt.duration_sec = duration
    step = float(getattr(cfg, "chunk_duration", 0) or 0)

    from .lexicon import (
        apply_censorship,
        apply_terms_postprocessing,
        build_initial_prompt,
        load_badwords,
        load_terms,
        term_coverage,
    )
    terms = load_terms(cfg.terms_file) if cfg.terms_file else []
    initial_prompt = build_initial_prompt(terms)

    # 2. ASR (с поддержкой пошаговых 5-минутных чекпоинтов FR-09)
    if _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["asr_done"] and ckpt.segments:
        say(f"ASR пропущен — загружено {len(ckpt.segments)} сегментов из чекпоинта")
        result = {"segments": list(ckpt.segments), "language": cfg.language}
    else:
        say(f"Распознавание речи ({cfg.model}, {device}, {compute_type})")
        if initial_prompt:
            say(f"Словарь терминов: {len(terms)} слов, initial prompt задан")

        model = whisperx.load_model(
            cfg.model, device, compute_type=compute_type, language=cfg.language
        )
        tr_kwargs = dict(batch_size=cfg.batch_size, language=cfg.language)
        if initial_prompt:
            tr_kwargs["initial_prompt"] = initial_prompt

        def _is_prompt_error(e: TypeError) -> bool:
            s = str(e).lower()
            return "initial_prompt" in s or "decodeoptions" in s

        def _run_asr(aud):
            try:
                return model.transcribe(aud, **tr_kwargs)
            except TypeError as e:
                if initial_prompt and _is_prompt_error(e):
                    say(
                        "Эта версия WhisperX не поддерживает initial_prompt — "
                        "словарь терминов будет применён постобработкой (замена опечаток)"
                    )
                    tr_kwargs.pop("initial_prompt", None)
                    return model.transcribe(aud, **tr_kwargs)
                raise

        if step > 0 and duration > step:
            n_chunks = int(-(-duration // step))
            ckpt.total_chunks = n_chunks
            start_chunk = (
                ckpt.completed_chunks
                if ckpt.stage == "asr_partial" and 0 < ckpt.completed_chunks < n_chunks
                else 0
            )
            all_segments = list(ckpt.segments) if start_chunk > 0 else []
            if start_chunk > 0:
                say(f"Продолжаю ASR с части {start_chunk + 1}/{n_chunks}")
            else:
                say(
                    f"Обработка частями по {step:g} с "
                    f"(всего частей: {n_chunks}, длительность ~{duration / 60:.1f} мин)"
                )
            for i in range(start_chunk, n_chunks):
                off = i * step
                piece = audio[..., int(off * 16_000): int(min(off + step, duration) * 16_000)]
                part = _run_asr(piece)
                for s in part["segments"]:
                    s = dict(s)
                    s["start"] = float(s.get("start", 0)) + off
                    s["end"] = float(s.get("end", 0)) + off
                    for w in s.get("words", []):
                        if w.get("start") is not None:
                            w["start"] = float(w["start"]) + off
                        if w.get("end") is not None:
                            w["end"] = float(w["end"]) + off
                    all_segments.append(s)
                ckpt.segments = list(all_segments)
                ckpt.completed_chunks = i + 1
                _save_ckpt("asr_partial", f"Часть {i + 1}/{n_chunks}")
                say(f"  Часть {i + 1}/{n_chunks}: сегментов {len(part['segments'])} [чекпоинт сохранён]")
                del part, piece
                _free_memory()
            result = {"segments": all_segments, "language": cfg.language}
        else:
            ckpt.total_chunks = 1
            result = _run_asr(audio)
            ckpt.completed_chunks = 1

        del model
        _free_memory()
        ckpt.segments = list(result["segments"])
        _save_ckpt("asr_done", f"ASR завершён: {len(ckpt.segments)} сегментов")

    # 3. Выравнивание по словам
    if _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["aligned"] and ckpt.segments:
        say("Выравнивание пропущено — загружено из чекпоинта")
        result = {"segments": list(ckpt.segments), "language": cfg.language}
    else:
        say("Выравнивание по словам")
        try:
            align_model, meta = whisperx.load_align_model(language_code=cfg.language, device=device)
        except Exception as e:
            if "codec" in str(e) or "unicode" in str(e).lower():
                raise RuntimeError(
                    f"Не удалось загрузить модель выравнивания: {e}\n"
                    "Решение для Windows: задайте UTF-8 перед запуском —\n"
                    "  set PYTHONUTF8=1"
                ) from e
            raise
        use_chunked_align = bool(step > 0 and duration > step)
        if use_chunked_align:
            say(f"Выравнивание по словам частями по {step:g} с")

        def _chunk_of(s):
            if not use_chunked_align:
                return 0
            mid = (float(s["start"]) + float(s["end"])) / 2.0
            return int(mid // step)

        n_align_chunks = int(-(-duration // step)) if use_chunked_align else 1
        aligned_segments = []
        for i in range(max(n_align_chunks, 1)):
            off = min(i * step, duration) if use_chunked_align else 0.0
            hi = min(off + step, duration) if use_chunked_align else duration
            if not (hi > off):
                break
            piece = audio[..., int(off * 16_000): int(hi * 16_000)]
            part_segs = [dict(s) for s in result["segments"] if _chunk_of(s) == i]
            for s in part_segs:
                s["start"] = float(s["start"]) - off
                s["end"] = max(float(s["end"]) - off, 0.0)
                for w in s.get("words", []):
                    if w.get("start") is not None:
                        w["start"] = float(w["start"]) - off
                    if w.get("end") is not None:
                        w["end"] = float(w["end"]) - off
            if part_segs:
                aligned_part = whisperx.align(
                    part_segs, align_model, meta, piece, device, return_char_alignments=False
                )
                for s in aligned_part["segments"]:
                    s = dict(s)
                    s["start"] = float(s.get("start", 0)) + off
                    s["end"] = float(s.get("end", 0)) + off
                    for w in s.get("words", []):
                        if w.get("start") is not None:
                            w["start"] = float(w["start"]) + off
                        if w.get("end") is not None:
                            w["end"] = float(w["end"]) + off
                    aligned_segments.append(s)
            del piece, part_segs
            _free_memory()
        result["segments"] = sorted(aligned_segments, key=lambda s: s["start"])
        del align_model
        _free_memory()
        ckpt.segments = list(result["segments"])
        _save_ckpt("aligned", "Выравнивание по словам завершено")

    # 4. Диаризация
    if cfg.diarize:
        if _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["diarized"] and ckpt.segments:
            say("Диаризация пропущена — загружена из чекпоинта")
            result = {"segments": list(ckpt.segments), "language": cfg.language}
        else:
            if not cfg.hf_token:
                raise RuntimeError(
                    "Для диаризации нужен токен Hugging Face (переменная HF_TOKEN) и принятые "
                    "условия модели pyannote/speaker-diarization-community-1. "
                    "Либо запустите с --no-diarize."
                )
            say("Диаризация спикеров")
            from whisperx.diarize import DiarizationPipeline
            pipe = DiarizationPipeline(token=cfg.hf_token, device=device)
            diar = pipe(
                audio,
                num_speakers=cfg.num_speakers,
                min_speakers=cfg.min_speakers,
                max_speakers=cfg.max_speakers,
            )
            result = whisperx.assign_word_speakers(diar, result)
            del pipe
            _free_memory()
            ckpt.segments = list(result["segments"])
            _save_ckpt("diarized", "Диаризация спикеров завершена")

    tr = _to_transcript(result, cfg.language, orig_source_name)
    tr.meta["duration_sec"] = round(duration, 2)
    tr.meta["preprocessed"] = bool(cfg.preprocess)

    # FR-04: человекочитаемые метки «Спикер 1», «Спикер 2»
    if cfg.normalize_speakers:
        normalize_speaker_labels(tr)

    # FR-03: постобработка терминов по словарю (исправление опечаток ASR)
    if terms and cfg.postprocess_terms:
        corrections = apply_terms_postprocessing(tr.segments, terms)
        if corrections:
            tr.meta["term_corrections"] = corrections
            say(f"Словарь терминов: исправлено опечаток — {len(corrections)}")

    # FR-03: маркировка ненормативной лексики [НБЗ] в тексте и в пословной разметке
    if cfg.censor:
        roots = load_badwords(cfg.badwords_file)
        n = apply_censorship(tr.segments, roots)
        tr.meta["censored_count"] = n
        if n:
            log.info("Маркировано [НБЗ]: %d", n)

    if terms:
        full_text = " ".join(s.text for s in tr.segments)
        tr.meta["term_coverage"] = term_coverage(full_text, terms)

    _save_ckpt("completed", "Обработка успешно завершена")
    tr.meta["checkpoint_history"] = ckpt.history
    say("Готово")
    return tr
