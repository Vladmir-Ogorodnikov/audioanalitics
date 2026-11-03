# АудиоАналитик ИИ — этап 1: транскрибация

Локальный конвейер: аудио (MP3/WAV/FLAC) → WhisperX (распознавание + выравнивание + диаризация) → `transcript.json` / `transcript.txt` с таймкодами и спикерами → части для LLM.

## Установка (Windows 10/11, NVIDIA RTX 3060)
1. Python 3.10–3.13, [ffmpeg](https://ffmpeg.org/) в PATH.
2. CUDA Toolkit 12.8 (требование WhisperX на Windows).
3. ```
   python -m venv .venv && .venv\Scripts\activate
   pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -e .[dev]
   ```
4. Диаризация: создайте токен на huggingface.co, примите условия модели
   `pyannote/speaker-diarization-community-1`, затем `set HF_TOKEN=hf_...`.
   Веса скачиваются один раз, дальше всё работает офлайн.

## Использование
```
audioanalytic transcribe data\lecture.mp3 --out out --speakers 2
audioanalytic transcribe data\lecture.mp3 --no-diarize
audioanalytic transcribe data\lecture.mp3 --low-vram      # GPU 8 ГБ
audioanalytic chunks out\lecture\transcript.json --max-chars 12000
pytest
```
Формат строки: `[00:12:03] SPEAKER_01: текст реплики`.

## Важно
AMD Radeon RX 6600 XT не поддерживается WhisperX/CTranslate2 на Windows (нет CUDA) — на ней будет работа только на CPU. Это стоит отразить как ограничение в ТЗ.
