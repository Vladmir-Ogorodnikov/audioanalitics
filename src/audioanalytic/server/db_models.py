"""ORM-модели базы данных по п. 4.10, 4.9 и 4.4 ТЗ (SQLAlchemy 2.0).

Сущности ТЗ:
  1. Recording          — «Запись»
  2. TranscriptRecord   — «Транскрипт»
  3. Utterance          — «Реплика» (с пословными таймкодами для плеера ±0,5 с)
  4. Summary            — «Конспект» (5 разделов + NER + полнотекстовый индекс)
  5. Thesis             — «Тезис» (с активной ссылкой на таймкод и confidence)
  6. AuditLog           — «Журнал» (152-ФЗ, хранение ≥ 90 дней)
  7. ModelRegistryEntry — «Реестр моделей» (версия, лицензия, контрольные метрики)
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Recording(Base):
    """Сущность «Запись» (исходное аудио, параметры очереди и чекпоинтов FR-09/FR-10)."""

    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_format: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False, default=16000)
    channels: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessed_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Очередь и отказоустойчивость (FR-09, FR-10)
    # queued | preprocessing | transcribing | completed | failed
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=2, index=True)  # 1=низкий, 2=обычный, 3=высокий
    progress_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage_message: Mapped[str] = mapped_column(String(512), nullable=False, default="Ожидает в очереди")
    checkpoint_stage: Mapped[str] = mapped_column(String(64), nullable=False, default="init")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Параметры распознавания
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="ru")
    requested_speakers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diarize_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    low_vram: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    pipeline_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")
    speed_preset: Mapped[str] = mapped_column(String(32), nullable=False, default="auto")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)

    transcript: Mapped["TranscriptRecord | None"] = relationship(
        back_populates="recording", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )
    summary: Mapped["Summary | None"] = relationship(
        back_populates="recording", cascade="all, delete-orphan", passive_deletes=True, uselist=False
    )

    __table_args__ = (
        Index("ix_recordings_queue_order", "status", "priority", "created_at"),
    )


class TranscriptRecord(Base):
    """Сущность «Транскрипт»."""

    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recordings.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="ru")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="large-v3")
    engine_used: Mapped[str] = mapped_column(String(64), nullable=False, default="whisperx")
    full_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    censored_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    term_coverage_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    speaker_map_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="transcript")
    utterances: Mapped[list["Utterance"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Utterance.seq_num",
    )


class Utterance(Base):
    """Сущность «Реплика» (FR-04, FR-07)."""

    __tablename__ = "utterances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcript_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq_num: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    speaker: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    original_speaker: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    is_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    words_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    transcript: Mapped["TranscriptRecord"] = relationship(back_populates="utterances")


class Summary(Base):
    """Сущность «Конспект» из 5 разделов + NER + полнотекстовый поиск (FR-05, FR-08)."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recordings.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="llama-3.1-8b-instruct")
    goal: Mapped[str] = mapped_column(Text, nullable=False, default="не выявлено")
    key_concepts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    tasks_decisions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    self_check_questions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    entities_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    search_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    recording: Mapped["Recording"] = relationship(back_populates="summary")
    theses: Mapped[list["Thesis"]] = relationship(
        back_populates="summary",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Thesis.seq_num",
    )


class Thesis(Base):
    """Сущность «Тезис» со ссылкой на таймкод и оценкой уверенности (FR-06, п. 4.5.3)."""

    __tablename__ = "theses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    summary_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq_num: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    section: Mapped[str] = mapped_column(String(64), nullable=False, default="key_theses")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    timecode_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    timecode_formatted: Mapped[str] = mapped_column(String(16), nullable=False, default="[00:00:00]")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    summary: Mapped["Summary"] = relationship(back_populates="theses")


class AuditLog(Base):
    """Сущность «Журнал» операций безопасности (NFR-04.3, хранение ≥ 90 дней)."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    client_ip: Mapped[str] = mapped_column(String(64), nullable=False, default="127.0.0.1")


class ModelRegistryEntry(Base):
    """Реестр открытых ИИ-моделей с лицензиями и контрольными метриками (п. 4.9, 4.11, CON-02)."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    component: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_url: Mapped[str] = mapped_column(String(512), nullable=False)
    license: Mapped[str] = mapped_column(String(128), nullable=False)
    quantization: Mapped[str] = mapped_column(String(64), nullable=False)
    vram_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
