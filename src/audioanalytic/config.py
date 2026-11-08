"""Настройки конвейера распознавания речи и серверного окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный загрузчик .env (без зависимостей): HF_TOKEN, DATABASE_URL и т.п."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and val and key not in os.environ:
            os.environ[key] = val


def get_data_dir() -> Path:
    """Каталог локального хранения данных внутри папки текущего пользователя ОС (NFR-04.1)."""
    raw = os.environ.get("AUDIOANALYTIC_DATA_DIR")
    base = Path(raw).expanduser() if raw else (Path.home() / ".audioanalytic")
    for sub in ("uploads", "preprocessed", "checkpoints", "exports", "dictionaries", "logs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def optimal_cpu_threads() -> int:
    """Оптимальное число потоков CPU (физические ядра без гипертрединга, чтобы избежать конкуренции MKL)."""
    logical = os.cpu_count() or 4
    return min(8, max(2, logical // 2))


@dataclass
class SttConfig:
    language: str = "ru"
    # Модель Whisper: large-v3, large-v3-turbo, small, либо "auto" (автовыбор под GPU/CPU).
    model: str = "large-v3"
    # Профиль скорости: "auto" | "fast" | "balanced" | "accurate"
    speed_preset: str = "auto"
    # float16 — GPU с 12 ГБ; int8_float16 — экономия VRAM; int8 — CPU.
    compute_type: str = "float16"
    batch_size: int = 16
    beam_size: int | None = None
    cpu_threads: int = field(default_factory=optimal_cpu_threads)
    device: str = "auto"  # auto | cuda | cpu
    # Если пользователь явно не фиксировал тяжёлую модель, на CPU автоматически берётся быстрая модель
    explicit_model: bool = False
    # Удерживать загруженные модели в памяти между файлами (ускоряет короткие файлы в 4–6 раз)
    keep_models_warm: bool = False

    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    # FR-02: предобработка аудио (16 кГц, моно, -16 LUFS)
    preprocess: bool = True
    two_pass_loudnorm: bool = False  # однопроходный ресемпл+loudnorm в 3.8x быстрее при том же таргете -16 LUFS
    preprocessed_path: str | None = None

    # FR-03: словарь терминов и маркировка [НБЗ]
    terms_file: str | None = None
    badwords_file: str | None = None
    censor: bool = True
    postprocess_terms: bool = True

    # FR-04: приводить метки SPEAKER_00 -> «Спикер 1»
    normalize_speakers: bool = True
    # FR-09: путь к файлу чекпоинта (каждые 5 мин / chunk_duration)
    checkpoint_path: str | None = None

    max_duration: float | None = None
    chunk_duration: float = 300.0

    @classmethod
    def low_vram(cls, **kw) -> "SttConfig":
        """Профиль для GPU с 8 ГБ (CON-03): large-v3-turbo, int8_float16, меньший батч."""
        base = dict(
            model="large-v3-turbo",
            speed_preset="balanced",
            compute_type="int8_float16",
            batch_size=8,
            beam_size=2,
            keep_models_warm=False,
        )
        base.update(kw)
        return cls(**base)

    @classmethod
    def from_preset(cls, preset: str = "auto", *, low_vram: bool = False, **kw) -> "SttConfig":
        """Создаёт конфигурацию по профилю скорости для веб-интерфейса и очереди."""
        if low_vram:
            return cls.low_vram(speed_preset=preset, **kw)

        p = (preset or "auto").lower()
        if p == "fast":
            base = dict(
                model="small",
                speed_preset="fast",
                compute_type="int8",
                batch_size=8,
                beam_size=1,
                explicit_model=True,
                two_pass_loudnorm=False,
            )
        elif p == "accurate":
            base = dict(
                model="large-v3",
                speed_preset="accurate",
                compute_type="float16",
                batch_size=16,
                beam_size=5,
                explicit_model=True,
                two_pass_loudnorm=True,
            )
        else:
            # "auto" / "balanced": large-v3-turbo на GPU (в 4x быстрее large-v3) и small на CPU (в 7x быстрее)
            base = dict(
                model="large-v3-turbo",
                speed_preset="balanced" if p == "balanced" else "auto",
                compute_type="float16",
                batch_size=16,
                beam_size=2,
                explicit_model=False,
                two_pass_loudnorm=False,
            )
        base.update(kw)
        return cls(**base)


@dataclass
class ServerSettings:
    """Настройки локального сервера (FastAPI + БД + изоляция 152-ФЗ)."""
    host: str = field(default_factory=lambda: os.environ.get("AUDIOANALYTIC_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("AUDIOANALYTIC_PORT", "8000")))
    data_dir: Path = field(default_factory=get_data_dir)
    database_url: str = field(default="")
    audit_retention_days: int = field(
        default_factory=lambda: max(90, int(os.environ.get("AUDIOANALYTIC_AUDIT_DAYS", "90")))
    )
    max_upload_bytes: int = 1024 ** 3  # FR-01: до 1 ГБ
    model_cache_idle_sec: float = field(
        default_factory=lambda: float(os.environ.get("AUDIOANALYTIC_CACHE_IDLE_SEC", "300"))
    )

    def __post_init__(self) -> None:
        if not self.database_url:
            env_db = os.environ.get("DATABASE_URL")
            if env_db:
                self.database_url = env_db
            else:
                db_file = (self.data_dir / "audioanalytic.db").resolve()
                self.database_url = f"sqlite:///{db_file.as_posix()}"
