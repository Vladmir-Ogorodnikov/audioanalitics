from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import LlmConfig, ServerSettings, SttConfig, load_dotenv
from .exporters import build_markdown
from .formatting import build_turns, chunk_for_llm, to_text
from .models import UNKNOWN_SPEAKER, Transcript
from .summarizer import generate_summary_for_turns, probe_llm_service
from .winfix import (
    allow_pyannote_unpickling,
    check_ffmpeg,
    disk_report_lines,
    ensure_disk_space,
    find_ffmpeg,
    setup_console,
)

AUDIO_EXT = {".mp3", ".wav", ".flac"}
MAX_BYTES = 1024 ** 3  # ограничение ТЗ: до 1 ГБ на файл


def _cmd_transcribe(a: argparse.Namespace) -> int:
    src = Path(a.audio)
    if src.suffix.lower() not in AUDIO_EXT:
        print(f"Формат {src.suffix} не поддерживается (MP3, WAV, FLAC)", file=sys.stderr)
        return 2
    if not src.is_file():
        print(f"Файл не найден: {src}", file=sys.stderr)
        return 2
    if src.stat().st_size > MAX_BYTES:
        print("Файл больше 1 ГБ", file=sys.stderr)
        return 2
    try:
        check_ffmpeg()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 3

    make = SttConfig.low_vram if a.low_vram else SttConfig
    cfg = make(
        language=a.language,
        diarize=not a.no_diarize,
        num_speakers=a.speakers,
        min_speakers=a.min_speakers,
        max_speakers=a.max_speakers,
    )
    if a.model:
        cfg.model = a.model
        cfg.explicit_model = True
    if getattr(a, "fast", False):
        cfg.speed_preset = "fast"
        if not a.model:
            cfg.model = "small"
            cfg.explicit_model = True
        cfg.beam_size = 1
    if getattr(a, "cpu", False):
        cfg.device = "cpu"
    if getattr(a, "max_duration", None) is not None:
        cfg.max_duration = float(a.max_duration)
    if getattr(a, "chunk_duration", None) is not None:
        cfg.chunk_duration = float(a.chunk_duration)
    cfg.preprocess = not a.no_preprocess
    cfg.terms_file = a.terms
    cfg.badwords_file = a.badwords
    cfg.censor = not a.no_censor
    if getattr(a, "checkpoint", None):
        cfg.checkpoint_path = a.checkpoint

    allow_pyannote_unpickling()
    try:
        ensure_disk_space(3, "скачивания весов моделей")
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 4
    from .stt import transcribe
    tr = transcribe(src, cfg, progress=lambda m: print(f"• {m}", flush=True))

    out = Path(a.out) / src.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.json").write_text(
        json.dumps(tr.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
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


def _cmd_evaluate(a: argparse.Namespace) -> int:
    """Метрики WER/CER, полноты и F1 по терминам транскрипта против эталона (NFR-01)."""
    from .lexicon import load_terms, term_coverage
    from .metrics import cer, term_f1, wer

    tr = Transcript.from_dict(json.loads(Path(a.transcript).read_text(encoding="utf-8")))
    hyp = " ".join(s.text for s in tr.segments)
    ref = Path(a.reference).read_text(encoding="utf-8")
    w, c = wer(ref, hyp), cer(ref, hyp)
    print(
        f"WER: {w.rate:.2%}  (ошибок {w.errors} из {w.total}; "
        f"замены {w.substitutions}, удаления {w.deletions}, вставки {w.insertions})"
    )
    print(f"CER: {c.rate:.2%}  (ошибок {c.errors} из {c.total})")
    if a.terms:
        terms = load_terms(a.terms)
        cov = term_coverage(hyp, terms)
        tf = term_f1(ref, hyp, terms)
        print(
            f"Полнота терминов (по словарю): {cov['terms_found']}/{cov['terms_total']} "
            f"= {cov['recall']:.1%}"
        )
        print(
            f"Термины против эталона (NFR-01): Recall={tf.recall:.1%}, "
            f"Precision={tf.precision:.1%}, F1={tf.f1:.3f} "
            f"(совпало {tf.true_positives} из {tf.ref_occurrences} эталонных вхождений)"
        )
    return 0


def _hf_get_json(url: str, token: str | None = None) -> tuple[int | None, dict | str]:
    import json as _json
    import urllib.error
    import urllib.request

    headers = {"User-Agent": "audioanalytic"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=15
        ) as resp:
            return getattr(resp, "status", 200), _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except Exception as e:
        return None, str(e)


def _check_diarize(report) -> None:
    tok = SttConfig().hf_token
    if not tok:
        report(
            "диаризация",
            False,
            "HF_TOKEN не задан — создайте .env из .env.example (или set HF_TOKEN=hf_...), "
            "либо запускайте с --no-diarize",
        )
        return
    try:
        import pyannote.audio as pa  # noqa: F401
        ver = getattr(pa, "__version__", None) or "?"
        report("pyannote.audio", True, f"установлен {ver}")
    except ImportError:
        report(
            "pyannote.audio",
            False,
            "pip install -U pyannote.audio (диаризация через WhisperX требует его)",
        )
        return
    except Exception as e:
        report("pyannote.audio", False, f"импорт падает: {e}")
        return

    status, info = _hf_get_json("https://huggingface.co/api/whoami-v2", tok)
    if status == 200 and isinstance(info, dict):
        report("HF_TOKEN", True, f"валиден (аккаунт: {info.get('name', '?')})")
        t_status, t_info = _hf_get_json(
            "https://huggingface.co/api/models/pyannote/"
            "speaker-diarization-community-1/tree/main",
            tok,
        )
        if t_status == 200:
            report("доступ к pyannote", True, "условия модели приняты — диаризация готова")
        elif t_status in (401, 403, 404):
            report(
                "доступ к pyannote",
                False,
                f"HTTP {t_status}: не приняты условия на "
                "https://huggingface.co/pyannote/speaker-diarization-community-1",
            )
        else:
            report(
                "доступ к pyannote",
                True,
                f"сеть недоступна ({t_info}) — пропускаем",
            )
    elif status in (401, 403):
        report(
            "HF_TOKEN",
            False,
            "токен отклонён (HTTP 401/403) — проверьте значение HF_TOKEN",
        )
    else:
        report(
            "HF_TOKEN",
            True,
            f"не удалось проверить онлайн ({info}) — пропускаем",
        )


def _cmd_doctor(_a: argparse.Namespace) -> int:
    import shutil

    ok = True

    def report(name: str, good: bool, note: str = "") -> None:
        nonlocal ok
        mark = "OK " if good else "!! "
        if not good:
            ok = False
        print(f"[{mark}] {name}" + (f" — {note}" if note else ""))

    report("Python", sys.version_info >= (3, 10), sys.version.split()[0])
    ff = shutil.which("ffmpeg") or find_ffmpeg()
    if ff and not shutil.which("ffmpeg"):
        report("ffmpeg", True, f"найден вне PATH: {ff} (подключается автоматически)")
    else:
        report(
            "ffmpeg в PATH",
            ff is not None,
            ff or "winget install Gyan.FFmpeg или pip install imageio-ffmpeg",
        )

    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "нет CUDA, будет CPU (медленнее)"
        report("torch / CUDA", True, f"torch {torch.__version__}, {name}")
    except ImportError:
        report(
            "torch",
            False,
            "pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
        )

    try:
        import whisperx
        report("whisperx", True, getattr(whisperx, "__version__", "установлен"))
    except ImportError:
        report("whisperx", False, "pip install whisperx")

    _check_diarize(report)

    try:
        import torchcodec.decoders  # noqa: F401
        report("torchcodec", True, "установлен корректно")
    except Exception as e:
        report(
            "torchcodec",
            True,
            f"не грузится ({type(e).__name__}) — не страшно: подготовленный WAV читается напрямую",
        )

    for name, good, note in disk_report_lines():
        report(name, good, note)

    llm_info = probe_llm_service(LlmConfig(), timeout=1.5)
    if llm_info["reachable"]:
        report(
            "локальная LLM (конспект)",
            True,
            f"{llm_info['provider']} на {llm_info['base_url']} — активна модель {llm_info['selected_model']}",
        )
    else:
        report(
            "локальная LLM (конспект)",
            True,
            f"Ollama не запущена ({llm_info['base_url']}) — будет использован встроенный резервный конспект",
        )

    print("Готов к работе" if ok else "Есть проблемы — см. строки с [!!]")
    return 0 if ok else 1


def _cmd_summarize(a: argparse.Namespace) -> int:
    """Формирует структурированный конспект (summary.md + summary.json) по transcript.json (FR-05)."""
    from .lexicon import load_terms

    tr_file = Path(a.transcript)
    if not tr_file.is_file():
        print(f"Файл не найден: {tr_file}", file=sys.stderr)
        return 2
    tr = Transcript.from_dict(json.loads(tr_file.read_text(encoding="utf-8")))
    turns_objs = build_turns(tr)
    turns = [
        {"start_sec": t.start, "end_sec": t.end, "speaker": t.speaker, "text": t.text}
        for t in turns_objs
    ]
    llm_cfg = LlmConfig()
    if a.model:
        llm_cfg.model = a.model
    known_terms = load_terms(a.terms) if a.terms else []
    summary = generate_summary_for_turns(
        turns,
        title=a.title or tr_file.parent.name,
        cfg=llm_cfg,
        known_terms=known_terms,
    )
    summary_dict = {
        "model_name": summary.model_name,
        "goal": summary.goal,
        "key_concepts": summary.key_concepts,
        "theses": [
            {
                "seq_num": th.seq_num,
                "text": th.text,
                "timecode_sec": th.timecode_sec,
                "timecode_formatted": th.timecode_formatted,
                "confidence": th.confidence,
                "needs_review": th.needs_review,
            }
            for th in summary.theses
        ],
        "tasks_decisions": summary.tasks_decisions,
        "self_check_questions": summary.self_check_questions,
        "entities": summary.entities,
    }
    out_dir = tr_file.parent
    (out_dir / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_text = build_markdown(
        title=a.title or tr_file.parent.name,
        duration_sec=tr.duration,
        word_count=sum(len(t["text"].split()) for t in turns),
        utterances=turns,
        summary=summary_dict,
    )
    (out_dir / "summary.md").write_text(md_text, encoding="utf-8")
    print(f"Конспект сохранён ({summary.model_name}) -> {out_dir / 'summary.md'}")
    return 0


def _cmd_check(a: argparse.Namespace) -> int:
    src = Path(a.audio)
    if not src.is_file():
        print(f"Файл не найден: {src}", file=sys.stderr)
        return 2

    print("=== Шаг 1/3: проверка окружения ===")
    _cmd_doctor(argparse.Namespace())
    print()

    print("=== Шаг 2/3: пробная транскрипция (полный цикл) ===")
    targs = argparse.Namespace(
        audio=str(src),
        out=a.out,
        language="ru",
        model=a.model,
        fast=False,
        low_vram=False,
        cpu=a.cpu,
        max_duration=a.max_duration,
        chunk_duration=a.chunk_duration,
        no_diarize=a.no_diarize,
        speakers=a.speakers,
        min_speakers=None,
        max_speakers=None,
        no_preprocess=False,
        terms=None,
        badwords=None,
        no_censor=False,
        checkpoint=None,
    )
    rc_t = _cmd_transcribe(targs)
    print()

    tr_path = Path(a.out) / src.stem / "transcript.json"
    txt_path = tr_path.with_name("transcript.txt")
    if rc_t != 0 or not tr_path.is_file():
        print(
            "[!!] Транскрипция не завершилась — полный цикл НЕ проходит "
            f"(код возврата {rc_t}). См. сообщения выше."
        )
        return 1

    print("=== Шаг 3/3: проверка результатов ===")
    tr = Transcript.from_dict(json.loads(tr_path.read_text(encoding="utf-8")))
    segs = tr.segments
    ok = True

    def report(name: str, good: bool, note: str = "") -> None:
        nonlocal ok
        if not good:
            ok = False
        print(f"[{'OK ' if good else '!! '}] {name}" + (f" — {note}" if note else ""))

    dur = segs[-1].end if segs else 0.0
    report("Сегменты", len(segs) > 0, f"{len(segs)} сегм., длительность ~{dur:.0f} с")
    n_words = sum(len(s.words) for s in segs)
    timed = all(w.start is not None and w.end is not None for s in segs for w in s.words)
    report("Выравнивание по словам", n_words > 0 and timed, f"{n_words} слов с таймкодами")

    turns = build_turns(tr)
    report("Реплики (склеивание сегментов)", len(turns) >= 1, f"{len(turns)} реплик из {len(segs)} сегментов")
    gaps = [
        round(t.start - prev.end, 2)
        for prev, t in zip(turns, turns[1:])
        if t.start - prev.end > 0.5
    ]
    print(
        f"[-- ] Паузы между репликами — {len(gaps)} пауз >0.5 c"
        + (f" (например {gaps[:5]})" if gaps else "")
    )

    if a.no_diarize:
        print("[-- ] Диаризация пропущена (--no-diarize)")
    else:
        spk = sorted({s.speaker for s in segs if s.speaker and s.speaker != UNKNOWN_SPEAKER})
        unknown = sum(1 for s in segs if s.speaker == UNKNOWN_SPEAKER)
        report(
            "Диаризация",
            len(spk) >= 1,
            f"спикеры: {', '.join(spk)}; сегментов без спикера: {unknown}/{len(segs)}",
        )

    parts = chunk_for_llm(tr, max_chars=a.max_chars)
    total = sum(len(p) for p in parts)
    joined = "\n".join(parts)
    intact = all(seg.text.strip() in joined for seg in segs if seg.text.strip())
    report(
        "Чанки для LLM",
        len(parts) >= 1 and intact,
        f"{len(parts)} част.(макс {max((len(p) for p in parts), default=0)} симв., "
        f"всего {total}); тексты сегментов целые: {'да' if intact else 'НЕТ'}",
    )

    print()
    print(f"Файлы: {tr_path}, {txt_path}")
    if ok:
        print("ПОЛНЫЙ ЦИКЛ РАБОТАЕТ ✓ (предобработка → ASR → выравнивание → диаризация → чанки)")
    else:
        print("Есть проблемы — см. строки с [!!]")
    return 0 if ok else 1


def _cmd_serve(a: argparse.Namespace) -> int:
    """Запускает локальный веб-интерфейс и REST API FastAPI (по умолчанию 127.0.0.1:8000, NFR-04.2)."""
    import uvicorn

    os.environ["AUDIOANALYTIC_HOST"] = a.host
    os.environ["AUDIOANALYTIC_PORT"] = str(a.port)
    os.environ["AUDIOANALYTIC_SKIP_DEFAULT_APP"] = "1"
    settings = ServerSettings(host=a.host, port=a.port)
    from .server.app import create_app

    print(f"АС «АудиоАналитик ИИ» запускается на http://{a.host}:{a.port}")
    print(f"Локальный каталог данных (NFR-04.1): {settings.data_dir}")
    app = create_app(settings=settings, start_worker=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_console()
    load_dotenv()
    p = argparse.ArgumentParser(prog="audioanalytic")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="аудио -> transcript.json/.txt")
    t.add_argument("audio")
    t.add_argument("--out", default="out")
    t.add_argument("--language", default="ru")
    t.add_argument("--model", help="например large-v3, large-v3-turbo, small")
    t.add_argument("--fast", action="store_true", help="ускоренный режим (модель small, greedy декодер)")
    t.add_argument("--low-vram", action="store_true", help="профиль для GPU 8 ГБ")
    t.add_argument("--cpu", action="store_true", help="принудительно CPU (без CUDA)")
    t.add_argument("--max-duration", type=float, metavar="SEC", help="обработать только первые SEC секунд")
    t.add_argument("--chunk-duration", type=float, metavar="SEC", help="резать длинные записи на куски по SEC с")
    t.add_argument("--checkpoint", help="путь к файлу чекпоинта (FR-09)")
    t.add_argument("--no-diarize", action="store_true")
    t.add_argument("--speakers", type=int, help="точное число спикеров")
    t.add_argument("--min-speakers", type=int)
    t.add_argument("--max-speakers", type=int)
    t.add_argument("--no-preprocess", action="store_true", help="пропустить предобработку FR-02")
    t.add_argument("--terms", help="файл словаря терминов (FR-03)")
    t.add_argument("--badwords", help="файл списка НБЗ-слов")
    t.add_argument("--no-censor", action="store_true", help="не маркировать [НБЗ]")
    t.set_defaults(fn=_cmd_transcribe)

    c = sub.add_parser("chunks", help="transcript.json -> части для LLM")
    c.add_argument("transcript")
    c.add_argument("--max-chars", type=int, default=12000)
    c.set_defaults(fn=_cmd_chunks)

    sm = sub.add_parser("summarize", help="transcript.json -> конспект summary.md и summary.json (FR-05)")
    sm.add_argument("transcript", help="путь к transcript.json")
    sm.add_argument("--model", help="имя модели в Ollama / API (по умолчанию qwen2.5:3b)")
    sm.add_argument("--title", help="название занятия для заголовка конспекта")
    sm.add_argument("--terms", help="словарь академических терминов")
    sm.set_defaults(fn=_cmd_summarize)

    d = sub.add_parser("doctor", help="проверка окружения (ffmpeg, CUDA, whisperx, HF_TOKEN, диаризация)")
    d.set_defaults(fn=_cmd_doctor)

    c2 = sub.add_parser("check", help="проверка ПОЛНОГО ЦИКЛА на коротком фрагменте")
    c2.add_argument("audio")
    c2.add_argument("--out", default="out_check")
    c2.add_argument("--model", default="small")
    c2.add_argument("--cpu", action="store_true")
    c2.add_argument("--max-duration", type=float, default=60.0, metavar="SEC")
    c2.add_argument("--chunk-duration", type=float, default=None, metavar="SEC")
    c2.add_argument("--no-diarize", action="store_true")
    c2.add_argument("--speakers", type=int)
    c2.add_argument("--max-chars", type=int, default=12000)
    c2.set_defaults(fn=_cmd_check)

    e = sub.add_parser("evaluate", help="WER/CER и F1 по терминам транскрипта против эталона (NFR-01)")
    e.add_argument("transcript", help="out/.../transcript.json")
    e.add_argument("reference", help="эталонная ручная расшифровка (.txt)")
    e.add_argument("--terms", help="словарь терминов для оценки полноты и F1")
    e.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("serve", help="запуск локального веб-интерфейса и REST API (FastAPI)")
    s.add_argument("--host", default=os.environ.get("AUDIOANALYTIC_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("AUDIOANALYTIC_PORT", "8000")))
    s.set_defaults(fn=_cmd_serve)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
