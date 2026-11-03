from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import SttConfig
from .formatting import chunk_for_llm, to_text
from .models import Transcript

AUDIO_EXT = {".mp3", ".wav", ".flac"}
MAX_BYTES = 1024 ** 3  # ограничение ТЗ: до 1 ГБ на файл


def _cmd_transcribe(a: argparse.Namespace) -> int:
    src = Path(a.audio)
    if src.suffix.lower() not in AUDIO_EXT:
        print(f"Формат {src.suffix} не поддерживается (MP3, WAV, FLAC)", file=sys.stderr)
        return 2
    if not src.is_file() or src.stat().st_size > MAX_BYTES:
        print("Файл не найден или больше 1 ГБ", file=sys.stderr)
        return 2

    make = SttConfig.low_vram if a.low_vram else SttConfig
    cfg = make(language=a.language, diarize=not a.no_diarize,
               num_speakers=a.speakers, min_speakers=a.min_speakers, max_speakers=a.max_speakers)
    if a.model:
        cfg.model = a.model

    from .stt import transcribe  # тяжёлые импорты только здесь
    tr = transcribe(src, cfg, progress=lambda m: print(f"• {m}", flush=True))

    out = Path(a.out) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(
        json.dumps(tr.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "transcript.txt").write_text(to_text(tr, not a.no_diarize), encoding="utf-8")
    print(f"Сохранено в {out}")
    return 0


def _cmd_chunks(a: argparse.Namespace) -> int:
    tr = Transcript.from_dict(json.loads(Path(a.transcript).read_text(encoding="utf-8")))
    parts = chunk_for_llm(tr, max_chars=a.max_chars)
    out = Path(a.transcript).parent / "llm_chunks"
    out.mkdir(exist_ok=True)
    for i, p in enumerate(parts, 1):
        (out / f"chunk_{i:03d}.txt").write_text(p, encoding="utf-8")
    print(f"{len(parts)} частей -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="audioanalytic")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="аудио -> transcript.json/.txt")
    t.add_argument("audio")
    t.add_argument("--out", default="out")
    t.add_argument("--language", default="ru")
    t.add_argument("--model", help="например large-v3, large-v3-turbo")
    t.add_argument("--low-vram", action="store_true", help="профиль для GPU 8 ГБ")
    t.add_argument("--no-diarize", action="store_true")
    t.add_argument("--speakers", type=int, help="точное число спикеров")
    t.add_argument("--min-speakers", type=int)
    t.add_argument("--max-speakers", type=int)
    t.set_defaults(fn=_cmd_transcribe)

    c = sub.add_parser("chunks", help="transcript.json -> части для LLM")
    c.add_argument("transcript")
    c.add_argument("--max-chars", type=int, default=12000)
    c.set_defaults(fn=_cmd_chunks)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
