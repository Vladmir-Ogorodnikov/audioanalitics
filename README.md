# АудиоАналитик ИИ — этап 1: транскрибация

Локальный конвейер: аудио (MP3/WAV/FLAC) → WhisperX (распознавание + выравнивание + диаризация) → `transcript.json` / `transcript.txt` с таймкодами и спикерами → части для LLM.

## Установка (Windows 10/11, NVIDIA RTX 3060)
1. **Python 3.10–3.12** (установщик с python.org; проверьте `python --version`).
2. **ffmpeg в PATH**: `winget install Gyan.FFmpeg` (после установки перезапустите терминал) либо сборка с [ffmpeg.org](https://ffmpeg.org/) + добавление папки `bin` в PATH.
3. **CUDA-версия torch ставится ДО остального**, иначе pip тянет CPU-сборку:
   ```bat
   cd C:\путь\к\проекту
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -e .[dev]
   ```
4. **Диаризация** (по желанию): создайте токен на huggingface.co, примите условия модели `pyannote/speaker-diarization-community-1`, затем:
   ```bat
   set HF_TOKEN=hf_...
   ```
   (для постоянного значения — «Изменение переменных среды» в свойствах системы). Веса скачиваются один раз, дальше всё работает офлайн. Без токена используйте `--no-diarize`.

### Типичные проблемы Windows (обходные пути уже встроены в код)
- **`UnicodeEncodeError` при выводе кириллицы/символов** — CLI сам переключает stdout/stderr в UTF-8 (`winfix.setup_console`). Если печатаете из своего кода — запускайте с `set PYTHONUTF8=1`.
- **`UnpicklingError` при диаризации** (torch ≥ 2.6 по умолчанию блокирует веса pyannote) — CLI автоматически разрешает загрузку этих чекпойнтов (`winfix.allow_pyannote_unpickling`).
- **Ошибка загрузки align-модели со словами «codec can't decode»** — обычно кириллица в пути к кэшу HF при системной кодировке cp1251. Решения: `set PYTHONUTF8=1`; либо галочка «Бета: использовать Юникод UTF-8…» в настройках языка Windows; либо перенесите кэш в ASCII-путь: `set HF_HOME=C:\hf-cache`.
- **float16/int8_float16 не работают на CPU** (ограничение ctranslate2) — программа сама переключится на `int8` и на CPU, если CUDA не найдена. Можно задать явно: `--cpu`.
- **Скачивание весов обрывается** — для стабильности отключён `hf_transfer` (однопоточная загрузка).
- **Отсутствует ffmpeg** — CLI сообщит об этом понятной подсказкой вместо cryptic-ошибки внутри whisperx.
- **AMD Radeon RX 6600 XT** не поддерживается WhisperX/CTranslate2 на Windows (нет CUDA) — только CPU. Это стоит отразить как ограничение в ТЗ.

## Использование
```bat
audioanalytic doctor                                :: проверка окружения (ffmpeg, CUDA, whisperx, HF_TOKEN)
audioanalytic transcribe data\lecture.mp3 --out out --speakers 2
audioanalytic transcribe data\lecture.mp3 --no-diarize
audioanalytic transcribe data\lecture.mp3 --low-vram        :: GPU 8 ГБ
audioanalytic transcribe data\lecture.mp3 --model small --cpu   :: быстрая проверка без GPU
audioanalytic chunks out\lecture\transcript.json --max-chars 12000
pytest
```
Можно запускать и так: `python -m audioanalytic transcribe ...`.
Формат строки: `[00:12:03] SPEAKER_01: текст реплики`.

## Как проверить, что всё работает
1. `audioanalytic doctor` — все строки должны быть `[OK ]` (кроме HF_TOKEN, если диаризация не нужна).
2. Логика без тяжёлых зависимостей: `pytest` (таймкоды, склейка реплик, нарезка частей, сериализация).
3. Реальный прогон: короткий (1–2 мин) фрагмент лекции в `data\`, затем
   `audioanalytic transcribe data\fragment.mp3 --no-diarize --model small`.
   Дождитесь `Сохранено в out\fragment` и откройте `out\fragment\transcript.txt` — строки вида `[00:00:12] текст`; `transcript.json` содержит сегменты и слова с таймкодами.
4. Диаризация: повторите без `--no-diarize` с заданным `HF_TOKEN` — появятся `SPEAKER_00/SPEAKER_01`.
5. Нарезка для LLM: `audioanalytic chunks out\fragment\transcript.json` → `out\fragment\llm_chunks\chunk_001.txt…`.
