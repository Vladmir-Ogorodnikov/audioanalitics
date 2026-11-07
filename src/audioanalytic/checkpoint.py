"""Чекпоинты обработки каждые 5 минут и возобновление после сбоя (FR-09).

Состояние записывается атомарно (через временный файл ``.tmp`` и ``os.replace``),
поэтому внезапное отключение питания или сбой драйвера GPU не повреждают чекпоинт.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CheckpointState:
    stage: str = "init"  # init | preprocessed | asr_partial | asr_done | aligned | diarized | completed
    source_file: str = ""
    preprocessed_path: str | None = None
    duration_sec: float = 0.0
    chunk_duration: float = 300.0  # 5 минут по ТЗ (FR-09)
    total_chunks: int = 0
    completed_chunks: int = 0
    segments: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    updated_at: str = ""

    def record_step(self, stage: str, note: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.stage = stage
        self.updated_at = now
        self.history.append({
            "stage": stage,
            "completed_chunks": self.completed_chunks,
            "total_chunks": self.total_chunks,
            "note": note,
            "timestamp": now,
        })


def save_checkpoint(path: str | Path, state: CheckpointState) -> Path:
    """Атомарно сохраняет чекпоинт в JSON-файл."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not state.updated_at:
        state.updated_at = datetime.now(timezone.utc).isoformat()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def load_checkpoint(path: str | Path | None) -> CheckpointState | None:
    """Загружает чекпоинт с диска или возвращает None, если файла нет / он повреждён."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return CheckpointState(
            stage=raw.get("stage", "init"),
            source_file=raw.get("source_file", ""),
            preprocessed_path=raw.get("preprocessed_path"),
            duration_sec=float(raw.get("duration_sec", 0.0)),
            chunk_duration=float(raw.get("chunk_duration", 300.0)),
            total_chunks=int(raw.get("total_chunks", 0)),
            completed_chunks=int(raw.get("completed_chunks", 0)),
            segments=list(raw.get("segments", [])),
            history=list(raw.get("history", [])),
            updated_at=raw.get("updated_at", ""),
        )
    except Exception:
        return None
