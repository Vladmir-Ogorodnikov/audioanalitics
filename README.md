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
   :: убедитесь, что на диске проекта (обычно C:) есть ~20 ГБ свободно — см. раздел ниже
   pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   pip cache purge        :: удалить скачанный wheel (~3.5 ГБ) после успешной установки
   pip install -e .[dev]
   ```
   Если места мало или хочется ставить на другой диск — создайте venv там:
   `python -m venv D:\venvs\audioanalytic` и активируйте `D:\venvs\audioanalytic\Scripts\activate`.
4. **Диаризация** (по желанию): создайте токен на huggingface.co, примите условия модели `pyannote/speaker-diarization-community-1`, затем:
   ```bat
   set HF_TOKEN=hf_...
   ```
   (для постоянного значения — «Изменение переменных среды» в свойствах системы). Веса скачиваются один раз, дальше всё работает офлайн. Без токена используйте `--no-diarize`.

### Как получить токен Hugging Face (нужен только для диаризации)
1. Зарегистрируйтесь на https://huggingface.co (или войдите).
2. Откройте страницу модели, которую использует WhisperX для диаризации:
   **https://huggingface.co/pyannote/speaker-diarization-community-1**
   и примите её условия (кнопка **«Agree to terms»** / «Accept license» под описанием модели). Пока условия не приняты, запрос весов вернёт ошибку 401/403 даже с валидным токеном.
3. Создайте token: https://huggingface.co/settings/tokens → **«New token»** → тип **Read** (Fine-grained тоже подойдёт, главное — Read-доступ к репозиториям pyannote) → скопируйте строку вида `hf_abcd...`. Токен показывается один раз.
4. Вставьте токен в проект — три способа на выбор:
   **Способ A (рекомендуемый) — файл `.env` в папке проекта:**
   ```bat
   copy .env.example .env
   notepad .env      :: заменить hf_ваш_токен_здесь на реальный токен, сохранить
   ```
   Программа сама читает `.env` при каждом запуске. Файл игнорируется git'ом — токен не попадёт в репозиторий.
   **Способ B — переменная окружения:**
   ```bat
   set HF_TOKEN=hf_abcd...        :: временно, до закрытия окна cmd
   ```
   или постоянно: Win+R → `sysdm.cpl` → «Дополнительно» → «Переменные среды» → создать переменную `HF_TOKEN`, затем перезапустить терминал.
   (Если заданы и `.env`, и `set` — приоритет у `set`.)
   **Способ C — через CLI hf** (токен сохранится в `C:\Users\<you>\.cache\huggingface\token`, подхватится автоматически):
   ```bat
   pip install -U "huggingface_hub[cli]"
   hf auth login        :: вставить токен при запросе
   ```
5. Проверьте: `audioanalytic doctor` — строка `HF_TOKEN` должна быть `[OK ]`. Первый прогон с диаризацией скачает веса pyannote и wav2vec2 (align-модель для русского — тоже gated, если 403 — примите условия и на https://huggingface.co/facebook/wav2vec2-large-xlsr-53-russian).

Быстрая проверка валидности токена из PowerShell:
```powershell
Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -Headers @{Authorization="Bearer $env:HF_TOKEN"}
```
(в cmd можно так: `curl -H "Authorization: Bearer %HF_TOKEN%" https://huggingface.co/api/whoami-v2`)

### Не хватает места на диске (`[Errno 28] No space left on device`)
CUDA-сборка torch — это wheel ~3.5 ГБ, который pip сначала качает в кэш, а потом
распаковывает в venv: итого нужно **~10–12 ГБ только на torch** и ещё несколько ГБ
на веса моделей (WhisperX скачивает их при первом прогоне). Всё это по умолчанию
пишется на диск `C:`. Что делать:

1. Освободите кэш pip (это безопасно, он только для повторных установок):
   ```bat
   pip cache purge
   :: или вручную удалите папку: %LOCALAPPDATA%\pip\Cache
   ```
2. Перенесите кэш pip на другой диск (например, D:) — задайте переменную ОДИН раз:
   ```bat
   setx PIP_CACHE_DIR "D:\pip-cache"
   ```
   затем **перезапустите терминал** (setx действует только в новых окнах) и повторите установку.
3. Перенесите кэш моделей Hugging Face (он тоже разрастается на гигабайты):
   ```bat
   setx HF_HOME "D:\hf-cache"
   ```
4. Альтернатива/дополнение — создайте venv не на C:, а на свободном диске:
   ```bat
   python -m venv D:\venvs\audioanalytic
   D:\venvs\audioanalytic\Scripts\activate
   ```
5. После успешной установки torch можно сделать `pip cache purge` — wheel больше не нужен.
6. Проверьте, где сколько места и сколько заняли кэши: `audioanalytic doctor`
   покажет строки «Диск», «Кэш pip», «Кэш huggingface» с размером в ГБ.

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
