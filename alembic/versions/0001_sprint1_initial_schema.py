"""Начальная схема БД Спринта 1 (6 сущностей п. 4.10 + реестр моделей п. 4.9 + GIN FTS).

Revision ID: 0001_sprint1
Revises: None
Create Date: 2026-11-07
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0001_sprint1"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recordings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("file_format", sa.String(16), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_sec", sa.Float(), nullable=False, server_default="0"),
        sa.Column("sample_rate", sa.Integer(), nullable=False, server_default="16000"),
        sa.Column("channels", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("original_path", sa.Text(), nullable=False),
        sa.Column("preprocessed_path", sa.Text(), nullable=True),
        sa.Column("checkpoint_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("progress_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stage_message", sa.String(512), nullable=False, server_default="Ожидает в очереди"),
        sa.Column("checkpoint_stage", sa.String(64), nullable=False, server_default="init"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("language", sa.String(16), nullable=False, server_default="ru"),
        sa.Column("requested_speakers", sa.Integer(), nullable=True),
        sa.Column("diarize_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("low_vram", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("pipeline_mode", sa.String(32), nullable=False, server_default="auto"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_sec", sa.Float(), nullable=True),
    )
    op.create_index("ix_recordings_status", "recordings", ["status"])
    op.create_index("ix_recordings_priority", "recordings", ["priority"])
    op.create_index("ix_recordings_queue_order", "recordings", ["status", "priority", "created_at"])

    op.create_table(
        "transcripts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("recording_id", sa.Integer(), sa.ForeignKey("recordings.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("language", sa.String(16), nullable=False, server_default="ru"),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("engine_used", sa.String(64), nullable=False),
        sa.Column("full_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("word_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("censored_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("term_coverage_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("speaker_map_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("meta_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "utterances",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("transcript_id", sa.Integer(), sa.ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq_num", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("speaker", sa.String(128), nullable=False),
        sa.Column("original_speaker", sa.String(128), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("is_edited", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("words_json", sa.Text(), nullable=False, server_default="[]"),
    )
    op.create_index("ix_utterances_transcript_id", "utterances", ["transcript_id"])

    op.create_table(
        "summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("recording_id", sa.Integer(), sa.ForeignKey("recordings.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("key_concepts_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("tasks_decisions_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("self_check_questions_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("entities_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("search_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "theses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("summary_id", sa.Integer(), sa.ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq_num", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("section", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("timecode_sec", sa.Float(), nullable=False, server_default="0"),
        sa.Column("timecode_formatted", sa.String(16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("needs_review", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64), nullable=True),
        sa.Column("entity_id", sa.Integer(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("client_ip", sa.String(64), nullable=False, server_default="127.0.0.1"),
    )
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"])

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("component", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False, unique=True),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("source_url", sa.String(512), nullable=False),
        sa.Column("license", sa.String(128), nullable=False),
        sa.Column("quantization", sa.String(64), nullable=False),
        sa.Column("vram_mb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metrics_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_transcripts_fts_ru "
            "ON transcripts USING GIN (to_tsvector('russian', coalesce(full_text, '')))"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_summaries_fts_ru "
            "ON summaries USING GIN (to_tsvector('russian', coalesce(search_text, '')))"
        )


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("audit_logs")
    op.drop_table("theses")
    op.drop_table("summaries")
    op.drop_table("utterances")
    op.drop_table("transcripts")
    op.drop_table("recordings")
