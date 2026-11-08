"""Подключение к СУБД (PostgreSQL 15+ и локальный SQLite fallback), индексы FTS и начальное наполнение."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import ServerSettings
from .db_models import Base, ModelRegistryEntry

DEFAULT_ACADEMIC_TERMS = [
    "# Академический словарь терминов (редактируется пользователем, п. 4.10 ТЗ)",
    "транскрибация",
    "диаризация",
    "квантование",
    "тригонометрия",
    "гомоскедастичность",
    "гетероскедастичность",
    "метод наименьших квадратов",
    "градиентный спуск",
    "регуляризация",
    "трансформер",
    "косинусное сходство",
    "векторные представления",
    "перекрёстная энтропия",
    "обратное распространение ошибки",
    "собственные значения",
    "сингулярное разложение",
    "нормальное распределение",
    "доверительный интервал",
    "математическое ожидание",
    "дисперсия",
]

DEFAULT_MODELS_SEED = [
    {
        "component": "stt",
        "model_name": "Systran/faster-whisper-large-v3",
        "version": "large-v3 (CTranslate2)",
        "source_url": "https://huggingface.co/Systran/faster-whisper-large-v3",
        "license": "MIT",
        "quantization": "float16 / int8",
        "vram_mb": 4200,
        "metrics_json": json.dumps(
            {"target_wer_le": 0.12, "target_cer_le": 0.05, "profile": "GPU 12 ГБ (RTX 3060)"},
            ensure_ascii=False,
        ),
        "is_active": True,
    },
    {
        "component": "stt_low_vram",
        "model_name": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "version": "large-v3-turbo",
        "source_url": "https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2",
        "license": "MIT",
        "quantization": "int8_float16",
        "vram_mb": 2400,
        "metrics_json": json.dumps(
            {"target_wer_le": 0.12, "profile": "GPU 8 ГБ (CON-03)"},
            ensure_ascii=False,
        ),
        "is_active": True,
    },
    {
        "component": "alignment",
        "model_name": "facebook/wav2vec2-large-xlsr-53-russian",
        "version": "1.0",
        "source_url": "https://huggingface.co/facebook/wav2vec2-large-xlsr-53-russian",
        "license": "Apache-2.0",
        "quantization": "fp16 / fp32",
        "vram_mb": 1300,
        "metrics_json": json.dumps(
            {"word_timestamp_precision_sec": 0.15, "requirement": "FR-07 (±0,5 с)"},
            ensure_ascii=False,
        ),
        "is_active": True,
    },
    {
        "component": "diarization",
        "model_name": "pyannote/speaker-diarization-community-1",
        "version": "community-1 (pyannote 3.x)",
        "source_url": "https://huggingface.co/pyannote/speaker-diarization-community-1",
        "license": "CC-BY-4.0 / MIT (pyannote.audio)",
        "quantization": "fp32",
        "vram_mb": 1600,
        "metrics_json": json.dumps(
            {"target_der_le": 0.15, "collar_sec": 0.25, "max_eval_speakers": 3},
            ensure_ascii=False,
        ),
        "is_active": True,
    },
    {
        "component": "embeddings_rag",
        "model_name": "intfloat/multilingual-e5-small",
        "version": "2.0",
        "source_url": "https://huggingface.co/intfloat/multilingual-e5-small",
        "license": "MIT",
        "quantization": "fp16 / int8",
        "vram_mb": 520,
        "metrics_json": json.dumps(
            {"stage": "Этап 3 (Спринт 2 — RAG)", "dim": 384},
            ensure_ascii=False,
        ),
        "is_active": False,
    },
    {
        "component": "llm",
        "model_name": "Meta-Llama-3.1-8B-Instruct-GGUF",
        "version": "3.1-8B",
        "source_url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "license": "Llama 3.1 Community License",
        "quantization": "Q4_K_M / Q6_K (4–8 бит)",
        "vram_mb": 5600,
        "metrics_json": json.dumps(
            {"target_hallucination_rate_le": 0.02, "stage": "Этап 3 (Спринт 2 — Конспект)"},
            ensure_ascii=False,
        ),
        "is_active": False,
    },
]


def _ru_lower(val: str | None) -> str:
    return val.lower() if isinstance(val, str) else ""


def build_engine(database_url: str) -> Engine:
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
    engine = create_engine(database_url, connect_args=connect_args, future=True, pool_pre_ping=True)

    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
            # Регистронезависимый поиск по кириллице в SQLite
            dbapi_connection.create_function("ru_lower", 1, _ru_lower)

    return engine


class DatabaseManager:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.engine = build_engine(settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def init_db(self) -> None:
        Base.metadata.create_all(bind=self.engine)
        # Автоматическая миграция для уже существующих локальных баз пользователей
        try:
            cols = {c["name"] for c in inspect(self.engine).get_columns("recordings")}
            if "speed_preset" not in cols:
                with self.engine.begin() as conn:
                    conn.execute(
                        text("ALTER TABLE recordings ADD COLUMN speed_preset VARCHAR(32) NOT NULL DEFAULT 'auto'")
                    )
        except Exception:
            pass
        if self.is_postgres:
            # Штатный полнотекстовый поиск СУБД PostgreSQL 15+ для русского языка (табл. 18, FR-08)
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_transcripts_fts_ru "
                        "ON transcripts USING GIN (to_tsvector('russian', coalesce(full_text, '')))"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_summaries_fts_ru "
                        "ON summaries USING GIN (to_tsvector('russian', coalesce(search_text, '')))"
                    )
                )
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        terms_path: Path = self.settings.data_dir / "dictionaries" / "terms.txt"
        if not terms_path.is_file():
            terms_path.parent.mkdir(parents=True, exist_ok=True)
            terms_path.write_text("\n".join(DEFAULT_ACADEMIC_TERMS) + "\n", encoding="utf-8")

        badwords_path: Path = self.settings.data_dir / "dictionaries" / "badwords.txt"
        if not badwords_path.is_file():
            badwords_path.write_text(
                "# Дополнительные корни маркируемой лексики [НБЗ] (по одному в строке)\n",
                encoding="utf-8",
            )

        with self.SessionLocal() as db:
            existing = {
                row[0] for row in db.execute(select(ModelRegistryEntry.model_name)).all()
            }
            added = False
            for item in DEFAULT_MODELS_SEED:
                if item["model_name"] not in existing:
                    db.add(ModelRegistryEntry(**item))
                    added = True
            if added:
                db.commit()

    def session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
