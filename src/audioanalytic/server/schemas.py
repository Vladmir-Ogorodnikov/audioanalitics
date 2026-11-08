"""Pydantic-схемы REST API (OpenAPI 3.0)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WordOut(BaseModel):
    text: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


class UtteranceOut(BaseModel):
    id: int
    seq_num: int
    start_sec: float
    end_sec: float
    timecode: str
    speaker: str
    original_speaker: str
    text: str
    is_edited: bool
    words: list[WordOut] = Field(default_factory=list)


class TranscriptOut(BaseModel):
    id: int
    language: str
    model_name: str
    engine_used: str
    word_count: int
    censored_count: int
    term_coverage: dict[str, Any] = Field(default_factory=dict)
    speaker_map: dict[str, str] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    speakers: list[str] = Field(default_factory=list)
    utterances: list[UtteranceOut] = Field(default_factory=list)


class RecordingSummaryItem(BaseModel):
    id: int
    title: str
    original_filename: str
    file_format: str
    file_size_bytes: int
    duration_sec: float
    sample_rate: int
    channels: int
    status: str
    priority: int
    progress_pct: float
    stage_message: str
    checkpoint_stage: str
    retry_count: int
    error_message: str | None = None
    estimated_wait_sec: float | None = None
    created_at: datetime
    completed_at: datetime | None = None
    processing_time_sec: float | None = None
    word_count: int = 0
    search_snippet: str | None = None
    matched_timecode_sec: float | None = None


class RecordingDetailOut(RecordingSummaryItem):
    language: str
    requested_speakers: int | None = None
    diarize_enabled: bool
    low_vram: bool
    pipeline_mode: str
    speed_preset: str = "auto"
    checkpoint_history: list[dict[str, Any]] = Field(default_factory=list)
    transcript: TranscriptOut | None = None


class SpeakerRenameRequest(BaseModel):
    old_name: str = Field(..., min_length=1, max_length=128)
    new_name: str = Field(..., min_length=1, max_length=128)


class UtteranceUpdateRequest(BaseModel):
    text: str | None = None
    speaker: str | None = Field(default=None, max_length=128)
    apply_censor: bool = True


class LexiconPayload(BaseModel):
    terms_text: str
    badwords_text: str


class MetricsEvaluateRequest(BaseModel):
    reference: str
    hypothesis: str
    terms: list[str] | None = None
    apply_postprocessing_preview: bool = True
    case_sensitive: bool = False
