"""Форматирование транскрипта: чтение человеком и подача в LLM с таймкодами."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Transcript


@dataclass
class Turn:
    """Реплика: подряд идущие сегменты одного спикера."""
    start: float
    end: float
    speaker: str
    text: str


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def build_turns(t: Transcript, max_gap: float = 2.0, max_turn_sec: float = 120.0) -> list[Turn]:
    """Склеивает соседние сегменты одного спикера, если пауза небольшая."""
    turns: list[Turn] = []
    for seg in t.segments:
        if not seg.text:
            continue
        last = turns[-1] if turns else None
        if (last and last.speaker == seg.speaker
                and seg.start - last.end <= max_gap
                and seg.end - last.start <= max_turn_sec):
            last.text += " " + seg.text
            last.end = seg.end
        else:
            turns.append(Turn(seg.start, seg.end, seg.speaker, seg.text))
    return turns


def turn_line(turn: Turn, with_speakers: bool = True) -> str:
    who = f"{turn.speaker}: " if with_speakers else ""
    return f"[{fmt_ts(turn.start)}] {who}{turn.text}"


def to_text(t: Transcript, with_speakers: bool = True) -> str:
    return "\n".join(turn_line(x, with_speakers) for x in build_turns(t)) + "\n"


def chunk_for_llm(t: Transcript, max_chars: int = 12000, with_speakers: bool = True) -> list[str]:
    """Режет транскрипт на части для контекста LLM по границам реплик.

    Одна реплика не разрывается; слишком длинная реплика остаётся целой частью.
    """
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for turn in build_turns(t):
        line = turn_line(turn, with_speakers)
        if cur and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks
