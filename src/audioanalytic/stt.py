"""Распознавание речи: WhisperX (ASR) -> выравнивание по словам -> диаризация.

Оптимизации скорости (в 4–8 раз на коротких файлах и CPU):
  1) Умный подбор модели и параметров декодера под устройство:
     - на CPU автоматически выбирается ``small`` + ``int8`` + ``batch_size=4`` + ``beam_size=2``
       вместо медленной ``large-v3`` (если пользователь явно не запросил ``large-v3``);
     - на GPU по умолчанию в режиме ``auto``/``balanced`` используется ``large-v3-turbo``;
     - ограничение ``temperatures=[0.0, 0.2]`` устраняет 6-кратные повторные прогоны Whisper
       на шумных паузах;
     - настройка числа потоков CPU (физические ядра) убирает конкуренцию потоков MKL/OpenMP.
  2) Опциональное кэширование моделей в памяти воркера (``keep_models_warm=True``) с защитой VRAM
     (``CON-03``) и автоочисткой при простое — повторные файлы не ждут чтения 4,5 ГБ весов с диска.
  3) Разумные границы спикеров ``min_speakers=1, max_speakers=4`` для ``pyannote`` по умолчанию.
"""
from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable

from .checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from .config import SttConfig, optimal_cpu_threads
from .model_cache import GLOBAL_MODEL_CACHE, free_torch_memory, has_ample_vram
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
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _configure_cpu_threads(threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def _resolve_execution_params(cfg: SttConfig, device: str, say: Progress) -> tuple[str, str, int, int]:
    """Возвращает оптимизированные ``(model_name, compute_type, batch_size, beam_size)``."""
    model_name = cfg.model or "auto"
    batch_size = cfg.batch_size
    beam_size = cfg.beam_size

    if device == "cpu":
        compute_type = "int8"
        if cfg.compute_type != "int8":
            say(f"CPU: compute_type '{cfg.compute_type}' заменён на 'int8'")
        if model_name in ("auto", "large-v3", "large-v3-turbo") and not cfg.explicit_model:
            if model_name != "small":
                say(
                    f"CPU без CUDA: вместо тяжёлой '{model_name}' автоматически выбрана "
                    "быстрая модель 'small' (ускорение в ~7 раз)"
                )
            model_name = "small"
        elif model_name == "auto":
            model_name = "small"
        batch_size = min(batch_size, 4)
        if beam_size is None:
            beam_size = 1 if cfg.speed_preset == "fast" else 2
    else:
        compute_type = cfg.compute_type
        if model_name == "auto":
            model_name = "large-v3-turbo"
        if beam_size is None:
            if cfg.speed_preset == "fast":
                beam_size = 1
            elif cfg.speed_preset == "accurate":
                beam_size = 5
            else:
                beam_size = 2

    return model_name, compute_type, max(1, batch_size), max(1, beam_size)


def _load_whisper_model(
    whisperx_mod: Any,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    beam_size: int,
    cpu_threads: int,
):
    """Вызывает ``whisperx.load_model`` с передачей ``asr_options`` и ``threads``, если версия их принимает."""
    try:
        params = set(inspect.signature(whisperx_mod.load_model).parameters)
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(whisperx_mod.load_model).parameters.values()
        )
    except (TypeError, ValueError):
        params, has_varkw = set(), False

    kwargs: dict[str, Any] = {}
    if "compute_type" in params or has_varkw:
        kwargs["compute_type"] = compute_type
    if "language" in params or has_varkw:
        kwargs["language"] = language

    # Ускорение декодера: убираем 6-ступенчатый temperature-fallback на шумных паузах
    asr_opts = {
        "beam_size": beam_size,
        "best_of": beam_size,
        "temperatures": [0.0] if beam_size == 1 else [0.0, 0.2],
    }
    if "asr_options" in params:
        kwargs["asr_options"] = asr_opts
    if device == "cpu" and "threads" in params:
        kwargs["threads"] = cpu_threads

    try:
        return whisperx_mod.load_model(model_name, device, **kwargs)
    except TypeError:
        # Откат для старых версий или тестовых заглушек
        return whisperx_mod.load_model(
            model_name, device, compute_type=compute_type, language=language
        )


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
        return np.ascontiguousarray(data[:, 0])
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
            say("Чтение подготовленного WAV напрямую")
        return _load_audio_fallback(audio_path)


def _load_audio_kwargs_names() -> set[str]:
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

    threads = cfg.cpu_threads or optimal_cpu_threads()
    if device == "cpu":
        _configure_cpu_threads(threads)

    model_name, compute_type, batch_size, beam_size = _resolve_execution_params(cfg, device, say)
    use_cache = bool(cfg.keep_models_warm)
    exclusive_vram = device == "cuda" and not has_ample_vram(10.0)

    work_audio_path = str(audio_path)

    # 1. FR-02: быстрая предобработка (16 кГц, моно, −16 LUFS)
    if cfg.preprocess:
        if (
            _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["preprocessed"]
            and ckpt.preprocessed_path
            and Path(ckpt.preprocessed_path).is_file()
        ):
            work_audio_path = ckpt.preprocessed_path
            say("Предобработка подхвачена из чекпоинта")
        else:
            say("Предобработка аудио (16 кГц, моно, −16 LUFS)")
            from .preprocess import preprocess
            prepared = preprocess(
                work_audio_path,
                dst=cfg.preprocessed_path,
                two_pass=cfg.two_pass_loudnorm,
            )
            work_audio_path = str(prepared)
            ckpt.preprocessed_path = work_audio_path
            _save_ckpt("preprocessed", "16 кГц, моно, −16 LUFS")

    say("Загрузка аудио в память")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    exe = None
    try:
        from .winfix import find_ffmpeg
        exe = find_ffmpeg()
    except Exception:
        pass
    # Если у нас уже чистый 16 кГц моно WAV после предобработки — читаем сразу через soundfile
    # за миллисекунды, вообще не вызывая повторный подпроцесс ffmpeg внутри whisperx.load_audio!
    if work_audio_path.lower().endswith(".wav") and cfg.preprocess:
        try:
            audio = _load_audio_fallback(work_audio_path)
        except Exception:
            audio = _load_audio(work_audio_path, exe, say)
    else:
        if not _torchcodec_ok():
            say("Используется прямое чтение WAV")
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

    # 2. ASR (с кэшированием модели и пошаговыми 5-минутными чекпоинтами FR-09)
    if _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["asr_done"] and ckpt.segments:
        say(f"ASR пропущен — загружено {len(ckpt.segments)} сегментов из чекпоинта")
        result = {"segments": list(ckpt.segments), "language": cfg.language}
    else:
        say(f"Подготовка модели распознавания ({model_name}, {device}, {compute_type})")
        asr_cache_key = ("asr", id(whisperx), model_name, device, compute_type, cfg.language, beam_size)
        if use_cache:
            model = GLOBAL_MODEL_CACHE.get_or_load(
                asr_cache_key,
                lambda: _load_whisper_model(
                    whisperx, model_name, device, compute_type, cfg.language, beam_size, threads
                ),
                exclusive=exclusive_vram,
            )
        else:
            model = _load_whisper_model(
                whisperx, model_name, device, compute_type, cfg.language, beam_size, threads
            )

        say(f"Распознавание речи ({model_name}, {device}, batch={batch_size}, beam={beam_size})")
        tr_kwargs: dict[str, Any] = dict(batch_size=batch_size, language=cfg.language)
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
                    f"(всего частей: {n_chunks}, ~{duration / 60:.1f} мин)"
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
                say(f"Часть {i + 1}/{n_chunks}: распознано сегментов — {len(part['segments'])}")
                del part, piece
                if not use_cache:
                    free_torch_memory()
            result = {"segments": all_segments, "language": cfg.language}
        else:
            ckpt.total_chunks = 1
            result = _run_asr(audio)
            ckpt.completed_chunks = 1

        if not use_cache:
            del model
            free_torch_memory()
        ckpt.segments = list(result["segments"])
        _save_ckpt("asr_done", f"ASR завершён: {len(ckpt.segments)} сегментов")

    # 3. Выравнивание по словам
    if _STAGE_ORDER.get(ckpt.stage, 0) >= _STAGE_ORDER["aligned"] and ckpt.segments:
        say("Выравнивание пропущено — загружено из чекпоинта")
        result = {"segments": list(ckpt.segments), "language": cfg.language}
    elif not result.get("segments"):
        say("Речь не обнаружена — выравнивание пропущено")
        _save_ckpt("aligned", "Пустой набор сегментов")
    else:
        say("Выравнивание слов по таймкодам")

        def _load_align():
            try:
                return whisperx.load_align_model(language_code=cfg.language, device=device)
            except Exception as e:
                if "codec" in str(e) or "unicode" in str(e).lower():
                    raise RuntimeError(
                        f"Не удалось загрузить модель выравнивания: {e}\n"
                        "Решение для Windows: задайте set PYTHONUTF8=1"
                    ) from e
                raise

        align_key = ("align", id(whisperx), cfg.language, device)
        if use_cache:
            align_model, meta = GLOBAL_MODEL_CACHE.get_or_load(
                align_key, _load_align, exclusive=exclusive_vram
            )
        else:
            align_model, meta = _load_align()

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
            if not use_cache:
                free_torch_memory()
        result["segments"] = sorted(aligned_segments, key=lambda s: s["start"])
        if not use_cache:
            del align_model
            free_torch_memory()
        ckpt.segments = list(result["segments"])
        _save_ckpt("aligned", "Выравнивание по словам завершено")

    # 4. Диаризация
    if cfg.diarize and result.get("segments"):
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

            diar_key = ("diarize", id(DiarizationPipeline), device)
            if use_cache:
                pipe = GLOBAL_MODEL_CACHE.get_or_load(
                    diar_key,
                    lambda: DiarizationPipeline(token=cfg.hf_token, device=device),
                    exclusive=exclusive_vram,
                )
            else:
                pipe = DiarizationPipeline(token=cfg.hf_token, device=device)

            # Если точное число спикеров не задано, ограничиваем кластеризацию 1..4 спикерами
            # (по ТЗ обычно 1–3 участника; это существенно ускоряет кластеризацию pyannote)
            min_spk = cfg.min_speakers
            max_spk = cfg.max_speakers
            if cfg.num_speakers is None:
                if min_spk is None:
                    min_spk = 1
                if max_spk is None:
                    max_spk = 4

            diar = pipe(
                audio,
                num_speakers=cfg.num_speakers,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            result = whisperx.assign_word_speakers(diar, result)
            if not use_cache:
                del pipe
                free_torch_memory()
            ckpt.segments = list(result["segments"])
            _save_ckpt("diarized", "Диаризация спикеров завершена")

    say("Постобработка текста и сохранение")
    tr = _to_transcript(result, cfg.language, orig_source_name)
    tr.meta["duration_sec"] = round(duration, 2)
    tr.meta["preprocessed"] = bool(cfg.preprocess)
    tr.meta["model_used"] = model_name
    tr.meta["device_used"] = device

    if cfg.normalize_speakers:
        normalize_speaker_labels(tr)

    if terms and cfg.postprocess_terms:
        corrections = apply_terms_postprocessing(tr.segments, terms)
        if corrections:
            tr.meta["term_corrections"] = corrections

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
