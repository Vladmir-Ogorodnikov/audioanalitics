"""FastAPI-приложение АС «АудиоАналитик ИИ» (OpenAPI 3.0, Спринт 1)."""
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from ..checkpoint import load_checkpoint
from ..config import ServerSettings
from ..formatting import fmt_ts
from ..lexicon import (
    DEFAULT_PROFANITY_ROOTS,
    SAFE_WORD_PREFIXES,
    censor_text,
    correct_terms_in_text,
    load_badwords,
    load_terms,
)
from ..metrics import cer, term_f1, wer
from ..models import Segment, Word
from ..preprocess import probe_audio
from ..winfix import find_ffmpeg, free_gb
from .audit import (
    cleanup_expired_audit_logs,
    export_audit_csv,
    export_audit_json,
    fetch_audit_logs,
    log_audit_event,
)
from .database import DatabaseManager
from .db_models import (
    ModelRegistryEntry,
    Recording,
    Summary,
    TranscriptRecord,
    Utterance,
)
from .demo_audio import generate_demo_lecture_wav, whisperx_available
from .schemas import (
    LexiconPayload,
    MetricsEvaluateRequest,
    RecordingDetailOut,
    RecordingSummaryItem,
    SpeakerRenameRequest,
    TranscriptOut,
    UtteranceOut,
    UtteranceUpdateRequest,
    WordOut,
)
from .worker import QueueWorker

ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".flac"}
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 МБ потоковая запись (FR-01)


def _gpu_status() -> dict[str, Any]:
    """Информация о GPU и VRAM для страницы мониторинга (п. 4.6, CON-03)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mb = round(props.total_memory / (1024 ** 2))
            alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2))
            reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2))
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "vram_total_mb": total_mb,
                "vram_allocated_mb": alloc_mb,
                "vram_reserved_mb": reserved_mb,
                "profile_recommendation": "standard (12 ГБ)" if total_mb >= 10000 else "low_vram (8 ГБ)",
            }
    except Exception:
        pass
    return {
        "available": False,
        "device_name": "CUDA GPU не обнаружен (активен CPU / демо-конвейер)",
        "vram_total_mb": 0,
        "vram_allocated_mb": 0,
        "vram_reserved_mb": 0,
        "profile_recommendation": "cpu / int8",
    }


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "127.0.0.1"


def _build_utterance_out(u: Utterance) -> UtteranceOut:
    try:
        raw_words = json.loads(u.words_json or "[]")
    except Exception:
        raw_words = []
    return UtteranceOut(
        id=u.id,
        seq_num=u.seq_num,
        start_sec=u.start_sec,
        end_sec=u.end_sec,
        timecode=f"[{fmt_ts(u.start_sec)}]",
        speaker=u.speaker,
        original_speaker=u.original_speaker,
        text=u.text,
        is_edited=u.is_edited,
        words=[WordOut(**w) for w in raw_words if isinstance(w, dict) and "text" in w],
    )


def _interpolate_words(text_val: str, start_sec: float, end_sec: float) -> list[dict]:
    """Пропорционально пересчитывает таймкоды слов после ручной правки реплики пользователем."""
    tokens = [t for t in text_val.split() if t]
    if not tokens:
        return []
    span = max(0.2, end_sec - start_sec)
    step = span / len(tokens)
    out = []
    for idx, tok in enumerate(tokens):
        ws = round(start_sec + idx * step, 2)
        we = round(min(end_sec, ws + step * 0.9), 2)
        out.append({"text": tok, "start": ws, "end": we, "score": 1.0})
    return out


def create_app(
    settings: ServerSettings | None = None,
    *,
    start_worker: bool = True,
) -> FastAPI:
    cfg = settings or ServerSettings()
    db_manager = DatabaseManager(cfg)
    worker = QueueWorker(db_manager, cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db_manager.init_db()
        with db_manager.SessionLocal() as db:
            cleanup_expired_audit_logs(db, cfg.audit_retention_days)
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(
        title="АС «АудиоАналитик ИИ»",
        description=(
            "Локальная автоматизированная система транскрибации, диаризации и анализа "
            "академических аудиозаписей (ГОСТ 34.602-2020, шифр темы ААИИ-2026-01)."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    app.openapi_version = "3.0.3"

    app.state.settings = cfg
    app.state.db_manager = db_manager
    app.state.worker = worker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Disposition"],
    )

    def get_db():
        yield from db_manager.session()

    # ── Мониторинг и реестр моделей ──────────────────────────────────────────

    @app.get("/api/v1/health", tags=["Мониторинг"])
    def healthcheck() -> dict[str, str]:
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/api/v1/system/status", tags=["Мониторинг"])
    def system_status(db: Session = Depends(get_db)) -> dict[str, Any]:
        queue_info = worker.estimate_queue_eta_sec(db)
        total_recordings = int(db.scalar(select(func.count(Recording.id))) or 0)
        completed_recordings = int(
            db.scalar(select(func.count(Recording.id)).where(Recording.status == "completed")) or 0
        )
        failed_recordings = int(
            db.scalar(select(func.count(Recording.id)).where(Recording.status == "failed")) or 0
        )
        ff = find_ffmpeg()
        return {
            "version": "0.2.0",
            "database_dialect": db_manager.engine.dialect.name,
            "data_dir": str(cfg.data_dir),
            "disk_free_gb": round(free_gb(cfg.data_dir) or 0.0, 2),
            "network_isolation": {
                "configured_host": cfg.host,
                "localhost_only": cfg.host in ("127.0.0.1", "localhost", "::1"),
                "external_telemetry": False,
                "offline_ready": True,
                "audit_retention_days": cfg.audit_retention_days,
            },
            "ffmpeg": {
                "available": ff is not None,
                "path": ff,
            },
            "whisperx_installed": whisperx_available(),
            "gpu": _gpu_status(),
            "queue": queue_info,
            "stats": {
                "total_recordings": total_recordings,
                "completed_recordings": completed_recordings,
                "failed_recordings": failed_recordings,
            },
        }

    @app.get("/api/v1/models", tags=["Реестр моделей"])
    def list_models(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
        rows = list(db.scalars(select(ModelRegistryEntry).order_by(ModelRegistryEntry.id.asc())).all())
        return [
            {
                "id": m.id,
                "component": m.component,
                "model_name": m.model_name,
                "version": m.version,
                "source_url": m.source_url,
                "license": m.license,
                "quantization": m.quantization,
                "vram_mb": m.vram_mb,
                "metrics": json.loads(m.metrics_json or "{}"),
                "is_active": m.is_active,
                "registered_at": m.registered_at.isoformat() if m.registered_at else None,
            }
            for m in rows
        ]

    # ── Приём аудио, библиотека и поиск (FR-01, FR-08, FR-10) ────────────────

    @app.post("/api/v1/recordings/upload", response_model=RecordingSummaryItem, tags=["Записи"])
    async def upload_recording(
        request: Request,
        file: UploadFile = File(...),
        title: str = Form(default=""),
        priority: int = Form(default=2),
        num_speakers: int | None = Form(default=None),
        diarize: bool = Form(default=True),
        low_vram: bool = Form(default=False),
        pipeline_mode: str = Form(default="auto"),
        db: Session = Depends(get_db),
    ) -> RecordingSummaryItem:
        orig_name = Path(file.filename or "recording.wav").name
        ext = Path(orig_name).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"Формат «{ext}» не поддерживается. Допустимы только MP3, WAV, FLAC (FR-01).",
            )
        prio = max(1, min(3, int(priority)))
        uid = uuid.uuid4().hex[:12]
        dst = cfg.data_dir / "uploads" / f"{uid}_{orig_name}"
        dst.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        try:
            with dst.open("wb") as out:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > cfg.max_upload_bytes:
                        out.close()
                        dst.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail="Размер файла превышает максимально допустимый предел 1 ГБ (FR-01).",
                        )
                    out.write(chunk)
        finally:
            await file.close()

        if written == 0:
            dst.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Загружен пустой аудиофайл.")

        info = {"sample_rate": 16000, "channels": 1, "duration": 0.0}
        try:
            info = probe_audio(dst)
        except Exception:
            pass

        clean_title = title.strip() or Path(orig_name).stem
        rec = Recording(
            title=clean_title,
            original_filename=orig_name,
            file_format=ext.lstrip("."),
            file_size_bytes=written,
            duration_sec=round(float(info.get("duration") or 0.0), 2),
            sample_rate=int(info.get("sample_rate") or 16000),
            channels=int(info.get("channels") or 1),
            original_path=str(dst),
            status="queued",
            priority=prio,
            progress_pct=0.0,
            stage_message="Поставлено в очередь обработки",
            checkpoint_stage="init",
            language="ru",
            requested_speakers=num_speakers if (num_speakers and num_speakers > 0) else None,
            diarize_enabled=diarize,
            low_vram=low_vram,
            pipeline_mode=pipeline_mode if pipeline_mode in ("auto", "whisperx", "demo") else "auto",
        )
        db.add(rec)
        db.flush()
        log_audit_event(
            db,
            "upload",
            f"Загружен аудиофайл «{orig_name}» ({written / (1024 * 1024):.2f} МБ) -> запись #{rec.id}",
            entity_type="recording",
            entity_id=rec.id,
            details={"size_bytes": written, "format": rec.file_format, "priority": prio},
            client_ip=_client_ip(request),
            commit=False,
        )
        db.commit()
        db.refresh(rec)

        if start_worker:
            worker.notify()

        eta_map = worker.estimate_queue_eta_sec(db)["eta_by_recording_id"]
        return _to_summary_item(rec, eta_map.get(rec.id))

    @app.post("/api/v1/recordings/demo", response_model=RecordingSummaryItem, tags=["Записи"])
    def create_demo_recording(
        request: Request,
        title: str = Query(default="Лекция №4: Метод наименьших квадратов и регуляризация"),
        priority: int = Query(default=3, ge=1, le=3),
        db: Session = Depends(get_db),
    ) -> RecordingSummaryItem:
        """Создаёт синтетическую учебную WAV-запись (22 кГц) и ставит её в очередь для мгновенной проверки UI."""
        uid = uuid.uuid4().hex[:10]
        dst = cfg.data_dir / "uploads" / f"demo_{uid}_lecture.wav"
        generate_demo_lecture_wav(dst, duration_sec=36.0)
        size = dst.stat().st_size
        info = probe_audio(dst)

        rec = Recording(
            title=title,
            original_filename="demo_lecture_ml.wav",
            file_format="wav",
            file_size_bytes=size,
            duration_sec=round(float(info.get("duration") or 36.0), 2),
            sample_rate=int(info.get("sample_rate") or 22050),
            channels=int(info.get("channels") or 1),
            original_path=str(dst),
            status="queued",
            priority=priority,
            progress_pct=0.0,
            stage_message="Поставлено в очередь обработки",
            checkpoint_stage="init",
            language="ru",
            requested_speakers=2,
            diarize_enabled=True,
            low_vram=False,
            pipeline_mode="demo",
        )
        db.add(rec)
        db.flush()
        log_audit_event(
            db,
            "upload",
            f"Создана демонстрационная запись #{rec.id} «{title}» для проверки конвейера",
            entity_type="recording",
            entity_id=rec.id,
            details={"demo": True, "size_bytes": size},
            client_ip=_client_ip(request),
            commit=False,
        )
        db.commit()
        db.refresh(rec)

        if start_worker:
            worker.notify()

        eta_map = worker.estimate_queue_eta_sec(db)["eta_by_recording_id"]
        return _to_summary_item(rec, eta_map.get(rec.id))

    @app.get("/api/v1/recordings", response_model=list[RecordingSummaryItem], tags=["Записи"])
    def list_recordings(
        q: str | None = Query(default=None, description="Полнотекстовый поиск по базе знаний (FR-08)"),
        status: str | None = Query(default=None),
        db: Session = Depends(get_db),
    ) -> list[RecordingSummaryItem]:
        eta_map = worker.estimate_queue_eta_sec(db)["eta_by_recording_id"]
        query_str = (q or "").strip()

        stmt = select(Recording).outerjoin(TranscriptRecord).outerjoin(Summary)
        if status:
            stmt = stmt.where(Recording.status == status)

        if query_str:
            q_low = f"%{query_str.lower()}%"
            if db_manager.is_postgres:
                stmt = stmt.where(
                    or_(
                        Recording.title.ilike(q_low),
                        text(
                            "to_tsvector('russian', coalesce(transcripts.full_text, '')) "
                            "@@ plainto_tsquery('russian', :fts_q)"
                        ).bindparams(fts_q=query_str),
                        text(
                            "to_tsvector('russian', coalesce(summaries.search_text, '')) "
                            "@@ plainto_tsquery('russian', :fts_q)"
                        ).bindparams(fts_q=query_str),
                        TranscriptRecord.full_text.ilike(q_low),
                    )
                )
            else:
                stmt = stmt.where(
                    or_(
                        func.ru_lower(Recording.title).like(q_low),
                        func.ru_lower(TranscriptRecord.full_text).like(q_low),
                        func.ru_lower(Summary.search_text).like(q_low),
                    )
                )

        stmt = stmt.order_by(Recording.created_at.desc())
        rows = list(db.scalars(stmt).unique().all())

        out: list[RecordingSummaryItem] = []
        for r in rows:
            snippet: str | None = None
            matched_tc: float | None = None
            if query_str and r.transcript is not None:
                q_norm = query_str.lower()
                for u in r.transcript.utterances:
                    if q_norm in u.text.lower() or any(
                        tok in u.text.lower() for tok in q_norm.split() if len(tok) >= 4
                    ):
                        snippet = f"[{fmt_ts(u.start_sec)}] {u.speaker}: {u.text}"
                        matched_tc = u.start_sec
                        break
            out.append(
                _to_summary_item(
                    r,
                    eta_map.get(r.id),
                    snippet=snippet,
                    matched_timecode_sec=matched_tc,
                )
            )
        return out

    @app.get("/api/v1/recordings/{recording_id}", response_model=RecordingDetailOut, tags=["Записи"])
    def get_recording(recording_id: int, db: Session = Depends(get_db)) -> RecordingDetailOut:
        rec = db.get(Recording, recording_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Запись не найдена.")

        eta_map = worker.estimate_queue_eta_sec(db)["eta_by_recording_id"]
        ckpt = load_checkpoint(rec.checkpoint_path)
        ckpt_history = ckpt.history if ckpt else []

        tr_out: TranscriptOut | None = None
        if rec.transcript is not None:
            tr = rec.transcript
            utt_out = [_build_utterance_out(u) for u in tr.utterances]
            seen_spk: list[str] = []
            for u in utt_out:
                if u.speaker not in seen_spk:
                    seen_spk.append(u.speaker)
            tr_out = TranscriptOut(
                id=tr.id,
                language=tr.language,
                model_name=tr.model_name,
                engine_used=tr.engine_used,
                word_count=tr.word_count,
                censored_count=tr.censored_count,
                term_coverage=json.loads(tr.term_coverage_json or "{}"),
                speaker_map=json.loads(tr.speaker_map_json or "{}"),
                meta=json.loads(tr.meta_json or "{}"),
                speakers=seen_spk,
                utterances=utt_out,
            )

        base = _to_summary_item(rec, eta_map.get(rec.id))
        return RecordingDetailOut(
            **base.model_dump(),
            language=rec.language,
            requested_speakers=rec.requested_speakers,
            diarize_enabled=rec.diarize_enabled,
            low_vram=rec.low_vram,
            pipeline_mode=rec.pipeline_mode,
            checkpoint_history=ckpt_history,
            transcript=tr_out,
        )

    @app.get("/api/v1/recordings/{recording_id}/audio", tags=["Записи"])
    def stream_recording_audio(recording_id: int, db: Session = Depends(get_db)) -> FileResponse:
        """Отдаёт аудиофайл с поддержкой HTTP Range (206 Partial Content) для перемотки плеера ±0,5 с (FR-07)."""
        rec = db.get(Recording, recording_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Запись не найдена.")

        prep = Path(rec.preprocessed_path) if rec.preprocessed_path else None
        orig = Path(rec.original_path)
        target = prep if (prep and prep.is_file()) else orig
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Аудиофайл на диске не найден.")

        media_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
        }
        media_type = media_map.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(path=str(target), media_type=media_type, filename=target.name)

    @app.post("/api/v1/recordings/{recording_id}/retry", response_model=RecordingSummaryItem, tags=["Записи"])
    def retry_from_checkpoint(
        recording_id: int,
        request: Request,
        db: Session = Depends(get_db),
    ) -> RecordingSummaryItem:
        """Возобновляет обработку с последнего сохранённого 5-минутного чекпоинта (FR-09)."""
        rec = db.get(Recording, recording_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Запись не найдена.")
        ckpt = load_checkpoint(rec.checkpoint_path)
        stage = ckpt.stage if ckpt else rec.checkpoint_stage
        rec.status = "queued"
        rec.retry_count = (rec.retry_count or 0) + 1
        rec.error_message = None
        rec.stage_message = f"Повторный запуск с чекпоинта ({stage})"
        log_audit_event(
            db,
            "checkpoint_resumed",
            f"Пользователь запустил возобновление записи #{rec.id} с чекпоинта {stage}",
            entity_type="recording",
            entity_id=rec.id,
            details={"checkpoint_stage": stage, "retry_count": rec.retry_count},
            client_ip=_client_ip(request),
            commit=False,
        )
        db.commit()
        db.refresh(rec)
        if start_worker:
            worker.notify()
        eta_map = worker.estimate_queue_eta_sec(db)["eta_by_recording_id"]
        return _to_summary_item(rec, eta_map.get(rec.id))

    @app.delete("/api/v1/recordings/{recording_id}", tags=["Записи"])
    def delete_recording(
        recording_id: int,
        request: Request,
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        """Каскадно удаляет запись из СУБД и все производные файлы с SSD за ≤ 20 с (FR-08, CON-04)."""
        t0 = time.monotonic()
        rec = db.get(Recording, recording_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Запись не найдена.")
        if worker.active_recording_id == recording_id:
            raise HTTPException(
                status_code=409,
                detail="Запись прямо сейчас обрабатывается воркером. Дождитесь завершения текущего шага.",
            )

        title = rec.title
        paths_to_remove = [
            rec.original_path,
            rec.preprocessed_path,
            rec.checkpoint_path,
        ]
        removed_files: list[str] = []
        for raw_p in paths_to_remove:
            if raw_p:
                p = Path(raw_p)
                if p.is_file():
                    p.unlink(missing_ok=True)
                    removed_files.append(p.name)
                tmp_p = p.with_suffix(p.suffix + ".tmp")
                if tmp_p.is_file():
                    tmp_p.unlink(missing_ok=True)

        export_dir = cfg.data_dir / "exports" / f"rec_{recording_id}"
        if export_dir.is_dir():
            shutil.rmtree(export_dir, ignore_errors=True)

        db.delete(rec)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        log_audit_event(
            db,
            "delete",
            f"Удалена запись #{recording_id} «{title}» и все производные файлы ({len(removed_files)} шт.) за {elapsed_ms} мс",
            entity_type="recording",
            entity_id=recording_id,
            details={"removed_files": removed_files, "deleted_in_ms": elapsed_ms},
            client_ip=_client_ip(request),
            commit=False,
        )
        db.commit()
        return {
            "deleted": True,
            "recording_id": recording_id,
            "removed_files": removed_files,
            "deleted_in_ms": elapsed_ms,
        }

    # ── Правка транскрипта, спикеров и экспорт (FR-04, FR-07) ────────────────

    @app.patch("/api/v1/recordings/{recording_id}/speakers", tags=["Транскрипт и спикеры"])
    def rename_speaker(
        recording_id: int,
        payload: SpeakerRenameRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        rec = db.get(Recording, recording_id)
        if rec is None or rec.transcript is None:
            raise HTTPException(status_code=404, detail="Транскрипт записи не найден.")

        old_n = payload.old_name.strip()
        new_n = payload.new_name.strip()
        updated = 0
        for u in rec.transcript.utterances:
            if u.speaker == old_n:
                u.speaker = new_n
                updated += 1

        if updated == 0:
            raise HTTPException(status_code=404, detail=f"Спикер «{old_n}» не найден в транскрипте.")

        spk_map = json.loads(rec.transcript.speaker_map_json or "{}")
        spk_map[old_n] = new_n
        rec.transcript.speaker_map_json = json.dumps(spk_map, ensure_ascii=False)
        rec.transcript.full_text = "\n".join(f"{u.speaker}: {u.text}" for u in rec.transcript.utterances)

        log_audit_event(
            db,
            "speaker_rename",
            f"В записи #{recording_id} спикер «{old_n}» переименован в «{new_n}» ({updated} реплик)",
            entity_type="recording",
            entity_id=recording_id,
            details={"old_name": old_n, "new_name": new_n, "updated_utterances": updated},
            client_ip=_client_ip(request),
            commit=False,
        )
        db.commit()
        return {"updated_utterances": updated, "old_name": old_n, "new_name": new_n}

    @app.patch("/api/v1/utterances/{utterance_id}", response_model=UtteranceOut, tags=["Транскрипт и спикеры"])
    def update_utterance(
        utterance_id: int,
        payload: UtteranceUpdateRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> UtteranceOut:
        u = db.get(Utterance, utterance_id)
        if u is None:
            raise HTTPException(status_code=404, detail="Реплика не найдена.")

        changed = False
        if payload.speaker is not None and payload.speaker.strip():
            u.speaker = payload.speaker.strip()
            changed = True

        if payload.text is not None:
            new_text = payload.text.strip()
            if payload.apply_censor:
                badwords_path = cfg.data_dir / "dictionaries" / "badwords.txt"
                roots = load_badwords(badwords_path)
                new_text, _ = censor_text(new_text, roots)
            if new_text != u.text:
                u.text = new_text
                u.is_edited = True
                u.words_json = json.dumps(
                    _interpolate_words(new_text, u.start_sec, u.end_sec), ensure_ascii=False
                )
                changed = True

        if changed and u.transcript is not None:
            tr = u.transcript
            tr.full_text = "\n".join(f"{x.speaker}: {x.text}" for x in tr.utterances)
            tr.word_count = sum(len(x.text.split()) for x in tr.utterances)
            log_audit_event(
                db,
                "utterance_edit",
                f"Отредактирована реплика #{u.id} ([{fmt_ts(u.start_sec)}]) в записи #{tr.recording_id}",
                entity_type="utterance",
                entity_id=u.id,
                details={"recording_id": tr.recording_id, "speaker": u.speaker},
                client_ip=_client_ip(request),
                commit=False,
            )
            db.commit()
            db.refresh(u)

        return _build_utterance_out(u)

    @app.get("/api/v1/recordings/{recording_id}/export", tags=["Транскрипт и спикеры"])
    def export_recording(
        recording_id: int,
        request: Request,
        format: str = Query(default="txt", pattern="^(txt|json)$"),
        db: Session = Depends(get_db),
    ) -> Response:
        rec = db.get(Recording, recording_id)
        if rec is None or rec.transcript is None:
            raise HTTPException(status_code=404, detail="Транскрипт для экспорта не найден.")

        tr = rec.transcript
        safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in rec.title).strip() or f"recording_{rec.id}"
        encoded_name = urllib.parse.quote(f"{safe_stem}.{format}")

        log_audit_event(
            db,
            "export",
            f"Экспорт транскрипта записи #{rec.id} в формат {format.upper()}",
            entity_type="recording",
            entity_id=rec.id,
            details={"format": format},
            client_ip=_client_ip(request),
            commit=True,
        )

        if format == "json":
            payload = {
                "recording_id": rec.id,
                "title": rec.title,
                "duration_sec": rec.duration_sec,
                "language": tr.language,
                "utterances": [_build_utterance_out(u).model_dump() for u in tr.utterances],
            }
            return Response(
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
            )

        lines = [
            f"АС «АудиоАналитик ИИ» — Расшифровка записи: {rec.title}",
            f"Длительность: {fmt_ts(rec.duration_sec)} | Слов: {tr.word_count}",
            "=" * 72,
            "",
        ]
        for u in tr.utterances:
            lines.append(f"[{fmt_ts(u.start_sec)}] {u.speaker}: {u.text}")
        txt_body = "\n".join(lines) + "\n"
        return PlainTextResponse(
            content=txt_body,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}"},
        )

    # ── Словари и интерактивная проверка метрик Спринта 1 (FR-03, NFR-01) ────

    @app.get("/api/v1/lexicon", tags=["Лингвистика и метрики"])
    def get_lexicon() -> dict[str, Any]:
        terms_path = cfg.data_dir / "dictionaries" / "terms.txt"
        badwords_path = cfg.data_dir / "dictionaries" / "badwords.txt"
        terms_raw = terms_path.read_text(encoding="utf-8") if terms_path.is_file() else ""
        bad_raw = badwords_path.read_text(encoding="utf-8") if badwords_path.is_file() else ""
        parsed_terms = load_terms(terms_path)
        all_roots = sorted(load_badwords(badwords_path))
        return {
            "terms_text": terms_raw,
            "badwords_text": bad_raw,
            "terms_count": len(parsed_terms),
            "active_profanity_roots_count": len(all_roots),
            "default_profanity_roots": sorted(DEFAULT_PROFANITY_ROOTS),
            "safe_word_prefixes": sorted(SAFE_WORD_PREFIXES),
        }

    @app.put("/api/v1/lexicon", tags=["Лингвистика и метрики"])
    def update_lexicon(
        payload: LexiconPayload,
        request: Request,
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        dict_dir = cfg.data_dir / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)
        terms_path = dict_dir / "terms.txt"
        badwords_path = dict_dir / "badwords.txt"
        terms_path.write_text(payload.terms_text, encoding="utf-8")
        badwords_path.write_text(payload.badwords_text, encoding="utf-8")

        parsed_terms = load_terms(terms_path)
        all_roots = sorted(load_badwords(badwords_path))
        log_audit_event(
            db,
            "lexicon_update",
            f"Обновлены пользовательские словари (терминов: {len(parsed_terms)}, корней [НБЗ]: {len(all_roots)})",
            entity_type="lexicon",
            details={"terms_count": len(parsed_terms), "roots_count": len(all_roots)},
            client_ip=_client_ip(request),
            commit=True,
        )
        return {
            "saved": True,
            "terms_count": len(parsed_terms),
            "active_profanity_roots_count": len(all_roots),
        }

    @app.post("/api/v1/metrics/evaluate", tags=["Лингвистика и метрики"])
    def evaluate_metrics_api(payload: MetricsEvaluateRequest) -> dict[str, Any]:
        """Стенд расчёта WER, CER, Term F1 и проверки фильтра [НБЗ] / постобработки терминов (NFR-01)."""
        terms_path = cfg.data_dir / "dictionaries" / "terms.txt"
        badwords_path = cfg.data_dir / "dictionaries" / "badwords.txt"
        terms = payload.terms if payload.terms is not None else load_terms(terms_path)
        roots = load_badwords(badwords_path)

        hyp_input = payload.hypothesis
        corrections: list[dict[str, str]] = []
        if payload.apply_postprocessing_preview and terms:
            hyp_input, corrections = correct_terms_in_text(hyp_input, terms)

        censored_hyp, censored_count = censor_text(hyp_input, roots)
        # Проверка пословной цензуры (как в Segment.words)
        sample_seg = Segment(
            start=0.0,
            end=5.0,
            text=payload.hypothesis,
            words=[Word(text=tok, start=i * 0.4, end=(i + 1) * 0.4) for i, tok in enumerate(payload.hypothesis.split())],
        )
        from ..lexicon import apply_censorship
        apply_censorship([sample_seg], roots)

        w = wer(payload.reference, censored_hyp, case_sensitive=payload.case_sensitive)
        c = cer(payload.reference, censored_hyp, case_sensitive=payload.case_sensitive)
        tf = term_f1(payload.reference, censored_hyp, terms)

        return {
            "processed_hypothesis": censored_hyp,
            "word_level_preview": [w_item.text for w_item in sample_seg.words],
            "censored_count": censored_count,
            "term_corrections": corrections,
            "wer": {
                "rate": round(w.rate, 4),
                "percent": round(w.rate * 100, 2),
                "passes_threshold_12pct": w.rate <= 0.12,
                "errors": w.errors,
                "total_words": w.total,
                "substitutions": w.substitutions,
                "deletions": w.deletions,
                "insertions": w.insertions,
            },
            "cer": {
                "rate": round(c.rate, 4),
                "percent": round(c.rate * 100, 2),
                "passes_threshold_5pct": c.rate <= 0.05,
                "errors": c.errors,
                "total_chars": c.total,
            },
            "terms": {
                "precision": round(tf.precision, 4),
                "recall": round(tf.recall, 4),
                "f1": round(tf.f1, 4),
                "passes_recall_85pct": tf.recall >= 0.85,
                "passes_f1_088": tf.f1 >= 0.88,
                "ref_occurrences": tf.ref_occurrences,
                "hyp_occurrences": tf.hyp_occurrences,
                "true_positives": tf.true_positives,
                "per_term": tf.per_term,
            },
        }

    # ── Журнал операций безопасности 152-ФЗ (NFR-04.3) ───────────────────────

    @app.get("/api/v1/audit-logs", tags=["Безопасность и аудит (152-ФЗ)"])
    def get_audit_logs(
        event_type: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=2000),
        db: Session = Depends(get_db),
    ) -> list[dict[str, Any]]:
        logs = fetch_audit_logs(db, event_type=event_type, limit=limit)
        return [
            {
                "id": item.id,
                "timestamp": item.timestamp.isoformat() if item.timestamp else None,
                "event_type": item.event_type,
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "client_ip": item.client_ip,
                "message": item.message,
                "details": json.loads(item.details_json or "{}"),
            }
            for item in logs
        ]

    @app.get("/api/v1/audit-logs/export", tags=["Безопасность и аудит (152-ФЗ)"])
    def export_audit_logs(
        format: str = Query(default="csv", pattern="^(csv|json)$"),
        db: Session = Depends(get_db),
    ) -> Response:
        logs = fetch_audit_logs(db, limit=5000)
        if format == "json":
            return Response(
                content=export_audit_json(logs),
                media_type="application/json; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=audit_logs_90d.json"},
            )
        return Response(
            content="\ufeff" + export_audit_csv(logs),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=audit_logs_90d.csv"},
        )

    # ── Статическая раздача собранного веб-интерфейса React 18 ───────────────

    static_dir = Path(__file__).resolve().parent.parent / "web" / "static"
    if (static_dir / "index.html").is_file():
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static-assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            candidate = (static_dir / full_path).resolve()
            if full_path and str(candidate).startswith(str(static_dir.resolve())) and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(static_dir / "index.html"))

    return app


def _to_summary_item(
    rec: Recording,
    eta_sec: float | None,
    *,
    snippet: str | None = None,
    matched_timecode_sec: float | None = None,
) -> RecordingSummaryItem:
    return RecordingSummaryItem(
        id=rec.id,
        title=rec.title,
        original_filename=rec.original_filename,
        file_format=rec.file_format,
        file_size_bytes=rec.file_size_bytes,
        duration_sec=rec.duration_sec,
        sample_rate=rec.sample_rate,
        channels=rec.channels,
        status=rec.status,
        priority=rec.priority,
        progress_pct=round(rec.progress_pct or 0.0, 1),
        stage_message=rec.stage_message,
        checkpoint_stage=rec.checkpoint_stage,
        retry_count=rec.retry_count or 0,
        error_message=rec.error_message,
        estimated_wait_sec=eta_sec,
        created_at=rec.created_at,
        completed_at=rec.completed_at,
        processing_time_sec=rec.processing_time_sec,
        word_count=rec.transcript.word_count if rec.transcript is not None else 0,
        search_snippet=snippet,
        matched_timecode_sec=matched_timecode_sec,
    )


# Для запуска через uvicorn factory: `uvicorn audioanalytic.server.app:create_app --factory`
def create_default_app() -> FastAPI:
    return create_app()
