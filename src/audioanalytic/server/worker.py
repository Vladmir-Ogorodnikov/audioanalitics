"""Фоновый воркер очереди с приоритетами, расчётом ETA и автовосстановлением с чекпоинтов (FR-09, FR-10)."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..checkpoint import load_checkpoint
from ..config import ServerSettings, SttConfig
from ..model_cache import GLOBAL_MODEL_CACHE
from ..models import Segment, Transcript
from .audit import log_audit_event
from .database import DatabaseManager
from .db_models import Recording, TranscriptRecord, Utterance
from .demo_audio import simulate_transcription, whisperx_available

log = logging.getLogger(__name__)

# По п. 4.3 ТЗ: на 1 ч аудио требуется ~15 мин распознавания (коэффициент 0.25 от длительности)
DEFAULT_RT_FACTOR = 0.25


def merge_segments_to_utterances(
    segments: list[Segment],
    *,
    max_gap: float = 2.0,
    max_turn_sec: float = 120.0,
) -> list[dict]:
    """Склеивает соседние сегменты одного спикера в реплики с сохранением пословных таймкодов."""
    turns: list[dict] = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        seg_words = [
            {"text": w.text, "start": w.start, "end": w.end, "score": w.score}
            for w in (seg.words or [])
        ]
        last = turns[-1] if turns else None
        if (
            last is not None
            and last["speaker"] == seg.speaker
            and (seg.start - last["end_sec"]) <= max_gap
            and (seg.end - last["start_sec"]) <= max_turn_sec
        ):
            last["text"] = f"{last['text']} {txt}"
            last["end_sec"] = float(seg.end)
            last["words"].extend(seg_words)
        else:
            turns.append({
                "start_sec": float(seg.start),
                "end_sec": float(seg.end),
                "speaker": seg.speaker,
                "original_speaker": seg.speaker,
                "text": txt,
                "words": seg_words,
            })
    return turns


class QueueWorker:
    """Потоковый обработчик очереди задач (последовательная работа на одном GPU по CON-03)."""

    def __init__(self, db_manager: DatabaseManager, settings: ServerSettings) -> None:
        self.db_manager = db_manager
        self.settings = settings
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_recording_id: int | None = None

    @property
    def active_recording_id(self) -> int | None:
        return self._active_recording_id

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.recover_interrupted_tasks()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="audioanalytic-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        GLOBAL_MODEL_CACHE.clear()

    def notify(self) -> None:
        """Будит воркер при поступлении новой задачи в очередь."""
        self._wake_event.set()

    def recover_interrupted_tasks(self) -> int:
        """FR-09: При запуске переводит прерванные сбоем задачи обратно в очередь с подхватом чекпоинта."""
        recovered = 0
        with self.db_manager.SessionLocal() as db:
            stuck = list(
                db.scalars(
                    select(Recording).where(Recording.status.in_(["preprocessing", "transcribing", "analyzing"]))
                ).all()
            )
            for rec in stuck:
                ckpt = load_checkpoint(rec.checkpoint_path)
                ckpt_stage = ckpt.stage if ckpt else rec.checkpoint_stage
                rec.status = "queued"
                rec.retry_count = (rec.retry_count or 0) + 1
                rec.checkpoint_stage = ckpt_stage
                rec.stage_message = f"Восстановлено после сбоя (чекпоинт: {ckpt_stage})"
                log_audit_event(
                    db,
                    "checkpoint_resumed",
                    f"Автоматическое восстановление записи #{rec.id} «{rec.title}» с чекпоинта {ckpt_stage}",
                    entity_type="recording",
                    entity_id=rec.id,
                    details={"checkpoint_stage": ckpt_stage, "retry_count": rec.retry_count},
                    commit=False,
                )
                recovered += 1
            if recovered:
                db.commit()
        return recovered

    def estimate_queue_eta_sec(self, db: Session) -> dict:
        """Расчётное время ожидания очереди (FR-10) по длительности записей и скорости обработки."""
        done_rows = db.execute(
            select(
                func.sum(Recording.processing_time_sec),
                func.sum(Recording.duration_sec),
            ).where(
                Recording.status == "completed",
                Recording.duration_sec > 5.0,
                Recording.processing_time_sec.is_not(None),
            )
        ).first()
        rt_factor = DEFAULT_RT_FACTOR
        if done_rows and done_rows[0] and done_rows[1] and done_rows[1] >= 60.0:
            measured = float(done_rows[0]) / float(done_rows[1])
            rt_factor = max(0.02, min(1.5, measured))

        pending = list(
            db.scalars(
                select(Recording)
                .where(Recording.status.in_(["queued", "preprocessing", "transcribing"]))
                .order_by(Recording.priority.desc(), Recording.created_at.asc())
            ).all()
        )
        total_audio_sec = 0.0
        eta_by_id: dict[int, float] = {}
        cumulative_wait = 0.0
        for r in pending:
            dur = max(10.0, float(r.duration_sec or 60.0))
            rem_ratio = max(0.05, 1.0 - float(r.progress_pct or 0.0) / 100.0)
            est_task = dur * rt_factor * rem_ratio
            cumulative_wait += est_task
            eta_by_id[r.id] = round(cumulative_wait, 1)
            total_audio_sec += dur

        return {
            "queued_count": sum(1 for r in pending if r.status == "queued"),
            "active_count": sum(1 for r in pending if r.status != "queued"),
            "queued_audio_hours": round(total_audio_sec / 3600.0, 3),
            "rt_factor": round(rt_factor, 3),
            "estimated_total_wait_sec": round(cumulative_wait, 1),
            "eta_by_recording_id": eta_by_id,
        }

    def process_one(self) -> bool:
        """Берёт одну задачу с наивысшим приоритетом и обрабатывает её. Возвращает True, если задача была."""
        with self.db_manager.SessionLocal() as db:
            stmt = (
                select(Recording)
                .where(Recording.status == "queued")
                .order_by(Recording.priority.desc(), Recording.created_at.asc())
                .limit(1)
            )
            if self.db_manager.is_postgres:
                stmt = stmt.with_for_update(skip_locked=True)
            rec = db.scalars(stmt).first()
            if rec is None:
                return False

            rec_id = rec.id
            rec.status = "preprocessing"
            rec.started_at = rec.started_at or datetime.now(timezone.utc)
            rec.progress_pct = max(rec.progress_pct or 0.0, 5.0)
            rec.stage_message = "Запуск конвейера предобработки и распознавания"
            rec.error_message = None
            log_audit_event(
                db,
                "queue_start",
                f"Начата обработка записи #{rec.id} «{rec.title}» (приоритет {rec.priority})",
                entity_type="recording",
                entity_id=rec.id,
                commit=False,
            )
            db.commit()

        self._active_recording_id = rec_id
        t0 = time.monotonic()
        try:
            self._execute_recording(rec_id, t0)
        except Exception as exc:
            log.exception("Ошибка обработки записи #%s", rec_id)
            with self.db_manager.SessionLocal() as db:
                rec = db.get(Recording, rec_id)
                if rec is not None:
                    ckpt = load_checkpoint(rec.checkpoint_path)
                    if ckpt:
                        rec.checkpoint_stage = ckpt.stage
                    rec.status = "failed"
                    rec.error_message = str(exc)
                    rec.stage_message = f"Ошибка: {str(exc)[:240]}"
                    log_audit_event(
                        db,
                        "fail",
                        f"Сбой обработки записи #{rec.id}: {str(exc)[:300]}",
                        entity_type="recording",
                        entity_id=rec.id,
                        details={"checkpoint_stage": rec.checkpoint_stage},
                        commit=False,
                    )
                    db.commit()
        finally:
            self._active_recording_id = None
        return True

    def _execute_recording(self, rec_id: int, t0: float) -> None:
        with self.db_manager.SessionLocal() as db:
            rec = db.get(Recording, rec_id)
            if rec is None:
                return
            orig_path = Path(rec.original_path)
            prep_path = Path(
                rec.preprocessed_path
                or (self.settings.data_dir / "preprocessed" / f"rec_{rec.id}.16k.wav")
            )
            ckpt_path = Path(
                rec.checkpoint_path
                or (self.settings.data_dir / "checkpoints" / f"rec_{rec.id}.ckpt.json")
            )
            rec.preprocessed_path = str(prep_path)
            rec.checkpoint_path = str(ckpt_path)
            db.commit()

            terms_path = self.settings.data_dir / "dictionaries" / "terms.txt"
            badwords_path = self.settings.data_dir / "dictionaries" / "badwords.txt"
            preset = getattr(rec, "speed_preset", None) or "auto"
            cfg = SttConfig.from_preset(
                preset,
                low_vram=bool(rec.low_vram),
                keep_models_warm=not bool(rec.low_vram),
                language=rec.language or "ru",
                diarize=bool(rec.diarize_enabled),
                num_speakers=rec.requested_speakers,
                preprocess=True,
                preprocessed_path=str(prep_path),
                checkpoint_path=str(ckpt_path),
                terms_file=str(terms_path) if terms_path.is_file() else None,
                badwords_file=str(badwords_path) if badwords_path.is_file() else None,
                censor=True,
                postprocess_terms=True,
                normalize_speakers=True,
            )
            mode = rec.pipeline_mode or "auto"

        def _on_progress(msg: str) -> None:
            low = msg.lower()
            pct = 18.0
            status = "transcribing"
            if "предобработк" in low:
                pct, status = 12.0, "preprocessing"
            elif "загрузка аудио" in low or "чтение" in low:
                pct = 20.0
            elif "подготовка модели" in low:
                pct = 28.0
            elif "распознавание" in low or "демо-режим asr" in low:
                pct = 40.0
            elif "часть " in low and "/" in low:
                pct = 58.0
            elif "выравниван" in low:
                pct = 74.0
            elif "диаризац" in low:
                pct = 88.0
            elif "постобработк" in low:
                pct = 95.0
            elif "готово" in low:
                pct = 99.0
            with self.db_manager.SessionLocal() as s_db:
                r = s_db.get(Recording, rec_id)
                if r is not None:
                    ckpt = load_checkpoint(ckpt_path)
                    if ckpt:
                        r.checkpoint_stage = ckpt.stage
                    r.status = status
                    r.progress_pct = max(r.progress_pct or 0.0, pct)
                    r.stage_message = msg[:500]
                    s_db.commit()

        use_real_wx = mode == "whisperx" or (mode == "auto" and whisperx_available())
        if use_real_wx:
            if cfg.diarize and not cfg.hf_token:
                cfg.diarize = False
            from ..stt import transcribe as run_whisperx
            from ..winfix import allow_pyannote_unpickling
            allow_pyannote_unpickling()
            transcript: Transcript = run_whisperx(orig_path, cfg, progress=_on_progress)
            actual_model = str(transcript.meta.get("model_used") or cfg.model)
            engine_used = f"whisperx ({actual_model})"
        else:
            transcript = simulate_transcription(orig_path, cfg, progress=_on_progress)
            actual_model = cfg.model
            engine_used = "demo-simulator (ffmpeg + чекпоинты + словари)"

        turns = merge_segments_to_utterances(transcript.segments)
        full_text = "\n".join(f"{u['speaker']}: {u['text']}" for u in turns)
        word_count = sum(len(u["text"].split()) for u in turns)
        elapsed = round(time.monotonic() - t0, 2)

        # Обновляем параметры подготовленного файла (16 кГц, моно)
        prep_sr, prep_ch, prep_dur = 16000, 1, float(transcript.meta.get("duration_sec") or transcript.duration)
        if prep_path.is_file():
            try:
                from ..preprocess import probe_audio
                p_info = probe_audio(prep_path)
                prep_sr = int(p_info.get("sample_rate") or 16000)
                prep_ch = int(p_info.get("channels") or 1)
                if p_info.get("duration"):
                    prep_dur = float(p_info["duration"])
            except Exception:
                pass

        with self.db_manager.SessionLocal() as db:
            rec = db.get(Recording, rec_id)
            if rec is None:
                return
            if rec.transcript is not None:
                db.delete(rec.transcript)
                db.flush()

            tr_row = TranscriptRecord(
                recording_id=rec.id,
                language=transcript.language,
                model_name=actual_model,
                engine_used=engine_used,
                full_text=full_text,
                word_count=word_count,
                censored_count=int(transcript.meta.get("censored_count", 0)),
                term_coverage_json=json.dumps(transcript.meta.get("term_coverage", {}), ensure_ascii=False),
                speaker_map_json=json.dumps(transcript.meta.get("speaker_map", {}), ensure_ascii=False),
                meta_json=json.dumps(transcript.meta, ensure_ascii=False),
            )
            db.add(tr_row)
            db.flush()

            for idx, u in enumerate(turns, start=1):
                db.add(
                    Utterance(
                        transcript_id=tr_row.id,
                        seq_num=idx,
                        start_sec=u["start_sec"],
                        end_sec=u["end_sec"],
                        speaker=u["speaker"],
                        original_speaker=u["original_speaker"],
                        text=u["text"],
                        is_edited=False,
                        words_json=json.dumps(u["words"], ensure_ascii=False),
                    )
                )

            rec.duration_sec = round(prep_dur, 2)
            rec.sample_rate = prep_sr
            rec.channels = prep_ch
            rec.status = "completed"
            rec.progress_pct = 100.0
            rec.checkpoint_stage = "completed"
            rec.stage_message = "Транскрибация и диаризация завершены"
            rec.completed_at = datetime.now(timezone.utc)
            rec.processing_time_sec = elapsed

            log_audit_event(
                db,
                "complete",
                f"Запись #{rec.id} «{rec.title}» успешно обработана за {elapsed:.1f} с ({len(turns)} реплик)",
                entity_type="recording",
                entity_id=rec.id,
                details={
                    "utterances": len(turns),
                    "word_count": word_count,
                    "censored_count": tr_row.censored_count,
                    "engine": engine_used,
                    "processing_time_sec": elapsed,
                },
                commit=False,
            )
            db.commit()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            processed = False
            try:
                processed = self.process_one()
            except Exception:
                log.exception("Непредвиденная ошибка цикла воркера")
            if not processed:
                GLOBAL_MODEL_CACHE.evict_if_idle(self.settings.model_cache_idle_sec)
                self._wake_event.wait(timeout=1.5)
                self._wake_event.clear()
