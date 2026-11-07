"""Структуры данных транскрипта (не зависят от WhisperX и torch)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

UNKNOWN_SPEAKER = "SPEAKER_??"
_RAW_SPEAKER_RE = re.compile(r"^speaker[_\-\s]*(\d+)$", re.IGNORECASE)


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
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"],
                speaker=s.get("speaker", UNKNOWN_SPEAKER),
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return cls(
            language=d["language"],
            segments=segs,
            source=d.get("source", ""),
            meta=d.get("meta", {}),
        )


def normalize_speaker_labels(transcript: Transcript) -> dict[str, str]:
    """Приводит технические метки диаризации (SPEAKER_00) к виду «Спикер N» (FR-04).

    Сохраняет порядок появления спикеров в записи и записывает карту соответствия
    в ``transcript.meta["speaker_map"]``. Уже переименованные метки («Спикер 1»,
    «Лектор» и т.п.) и ``UNKNOWN_SPEAKER`` не перезаписываются.
    """
    mapping: dict[str, str] = {}
    used_numbers: set[int] = set()

    for seg in transcript.segments:
        m_ru = re.match(r"^Спикер\s+(\d+)$", seg.speaker.strip())
        if m_ru:
            used_numbers.add(int(m_ru.group(1)))

    next_idx = 1
    for seg in transcript.segments:
        raw = (seg.speaker or UNKNOWN_SPEAKER).strip()
        if not raw or raw == UNKNOWN_SPEAKER:
            seg.speaker = UNKNOWN_SPEAKER
            continue
        if raw in mapping:
            seg.speaker = mapping[raw]
            continue
        if _RAW_SPEAKER_RE.match(raw):
            while next_idx in used_numbers:
                next_idx += 1
            human = f"Спикер {next_idx}"
            used_numbers.add(next_idx)
            next_idx += 1
            mapping[raw] = human
            seg.speaker = human

    if mapping:
        transcript.meta["speaker_map"] = mapping
    return mapping
