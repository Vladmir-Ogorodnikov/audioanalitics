"""Распознавание речи: WhisperX (ASR) -> выравнивание по словам -> диаризация.

Этапы выполняются последовательно, модели выгружаются из памяти GPU между
этапами, чтобы пик VRAM оставался низким (см. конфликт №3 в ТЗ).
whisperx/torch импортируются лениво, чтобы остальной код работал без них.
"""
from __future__ import annotations

import gc
import logging
import os
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


def _torchcodec_ok() -> bool:
    """True, если встроенный декодер whisperx (torchaudio/torchcodec) рабочий.

    На Windows torchcodec часто не устанавливается корректно (не грузятся
    libtorchcodec_core*.dll) — тогда load_audio падает уже ПОСЛЕ нашей
    предобработки, и весь прогон идёт насмарку. Проверяем это заранее.
    """
    try:
        import torchcodec.decoders  # noqa: F401
        return True
    except Exception:
        return False


def _load_audio_fallback(audio_path: str, sr: int = 16_000):
    """Читает WAV в обход torchcodec через soundfile (или stdlib wave)."""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        data, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if data.shape[1] > 1:  # даунмикс в моно
            data = data.mean(axis=1, keepdims=True)
        if rate != sr:
            # редкий случай: пересэмплируем «наивно» нельзя — просим ffmpeg
            raise RuntimeError(f"sample rate {rate} != {sr}")
        return torch.from_numpy(np.ascontiguousarray(data.T))  # (channel, time)
    except ImportError:
        pass
    # совсем лёгкий путь без зависимостей: 16-битный моно WAV читаем модулем wave
    import array
    import torch
    import wave
    with wave.open(audio_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        data = array.array("h", raw)
        mono = [s / 32768.0 for s in data]
        return torch.tensor(mono, dtype=torch.float32).unsqueeze(0)


def _load_audio(audio_path: str, ffmpeg_file: str | None,
                say: Progress | None = None) -> object:
    """Загружает аудио для WhisperX; при сломанном torchcodec читает WAV сам."""
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


def transcribe(audio_path: str | Path, cfg: SttConfig | None = None,
               progress: Progress | None = None) -> Transcript:
    cfg = cfg or SttConfig()
    say = progress or (lambda m: log.info(m))
    import whisperx

    device = _resolve_device(cfg.device)
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            say("CUDA не доступна — переключаюсь на CPU")
            device = "cpu"
    # int8_float16/float16 на CPU падают внутри ctranslate2 -> понижаем до int8
    if device == "cuda":
        compute_type = cfg.compute_type
    else:
        if cfg.compute_type != "int8":
            say(f"CPU: compute_type '{cfg.compute_type}' заменён на 'int8'")
        compute_type = "int8"
    audio_path = str(audio_path)

    # FR-02: предобработка — 16 кГц, моно, -16 LUFS (улучшает WER на тихих/громких записях)
    if cfg.preprocess:
        say("Предобработка аудио (16 кГц, моно, −16 LUFS)")
        from .preprocess import preprocess
        audio_path = str(preprocess(audio_path))

    say("Загрузка аудио")
    # HF_HUB_ENABLE_HF_TRANSFER=1 на Windows часто обрывает скачивание весов -> отключаем
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # ffmpeg может быть не в PATH, но найден в папках winget — подставим путь явно
    exe = None
    try:
        from .winfix import find_ffmpeg
        exe = find_ffmpeg()
    except Exception:
        pass
    if not _torchcodec_ok():
        say("torchcodec не установлен корректно — встроенное декодирование "
            "будет заменено чтением WAV напрямую (через soundfile/ffmpeg)")
    audio = _load_audio(audio_path, exe, say)
    # --max-duration: обработать только первые N секунд (быстрый тест всего цикла)
    if cfg.max_duration:
        n = int(float(cfg.max_duration) * 16_000)
        if hasattr(audio, "shape") and audio.shape[-1] > n:
            say(f"Обрезаю аудио до {cfg.max_duration:g} с (--max-duration)")
            audio = audio[..., :n]

    say(f"Распознавание речи ({cfg.model}, {device}, {compute_type})")
    # FR-03: словарь терминов -> initial prompt Whisper (частично компенсирует
    # отсутствие дообучения; подсказка помогает модели выбирать правильные термины)
    from .lexicon import (apply_censorship, build_initial_prompt, load_badwords,
                          load_terms, term_coverage)
    terms = load_terms(cfg.terms_file) if cfg.terms_file else []
    initial_prompt = build_initial_prompt(terms)
    if initial_prompt:
        say(f"Словарь терминов: {len(terms)} слов, initial prompt задан")

    model = whisperx.load_model(cfg.model, device, compute_type=compute_type, language=cfg.language)
    result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language,
                              initial_prompt=initial_prompt)
    del model
    _free_memory()

    say("Выравнивание по словам")
    try:
        align_model, meta = whisperx.load_align_model(language_code=cfg.language, device=device)
    except Exception as e:
        # Типичная проблема на Windows/русских сборках: locale cp1251 и кириллица
        # в пути к кэшу HF -> UnicodeDecodeError при загрузке весов.
        if "codec" in str(e) or "unicode" in str(e).lower():
            raise RuntimeError(
                f"Не удалось загрузить модель выравнивания: {e}\n"
                "Решение для Windows: задайте UTF-8 перед запуском —\n"
                "  set PYTHONUTF8=1\n"
                "или включите «Beta: Use Unicode UTF-8 for worldwide language support» "
                "в Параметры -> Язык и регион -> Административные параметры языка."
            ) from e
        raise
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
    tr = _to_transcript(result, cfg.language, Path(audio_path).name)

    # FR-03: маркировка ненормативной лексики меткой [НБЗ]
    if cfg.censor:
        roots = load_badwords(cfg.badwords_file)
        n = apply_censorship(tr.segments, roots)
        if n:
            log.info("Маркировано [НБЗ]: %d", n)
    # метрика полноты терминов (для отчётов; не влияет на текст)
    if terms:
        full_text = " ".join(s.text for s in tr.segments)
        tr.meta["term_coverage"] = term_coverage(full_text, terms)
    return tr
