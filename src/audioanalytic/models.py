"""Структуры данных транскрипта (не зависят от WhisperX и torch)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

UNKNOWN_SPEAKER = "SPEAKER_??"


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    score: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = UNKNOWN_SPEAKER
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]
    source: str = ""
    meta: dict = field(default_factory=dict)  # term_coverage, флаги предобработки и т.п.

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segs = [
            Segment(
                start=s["start"], end=s["end"], text=s["text"],
                speaker=s.get("speaker", UNKNOWN_SPEAKER),
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(language=d["language"], segments=segs, source=d.get("source", ""),
                   meta=d.get("meta", {}))
