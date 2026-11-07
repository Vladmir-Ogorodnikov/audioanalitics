"""Интеграционные тесты серверной части FastAPI, БД, очереди, FTS и аудита 152-ФЗ (Спринт 1)."""
from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient

os.environ["AUDIOANALYTIC_SKIP_DEFAULT_APP"] = "1"

from audioanalytic.config import ServerSettings
from audioanalytic.server.app import create_app
from audioanalytic.server.db_models import Recording
from audioanalytic.server.demo_audio import generate_demo_lecture_wav


def make_test_app(tmp_path: Path):
    settings = ServerSettings(
        host="127.0.0.1",
        port=8000,
        data_dir=tmp_path / "user_data",
    )
    # Обеспечиваем создание подпапок внутри папки пользователя
    for sub in ("uploads", "preprocessed", "checkpoints", "exports", "dictionaries", "logs"):
        (settings.data_dir / sub).mkdir(parents=True, exist_ok=True)
    app = create_app(settings=settings, start_worker=False)
    return app, settings


def test_alembic_migration_upgrade(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "alembic_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("AUDIOANALYTIC_DATA_DIR", str(tmp_path / "data"))
    alembic_cfg = AlembicConfig("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    assert db_file.is_file()


def test_end_to_end_sprint1_api_workflow(tmp_path: Path):
    app, settings = make_test_app(tmp_path)

    with TestClient(app) as client:
        # 1. Мониторинг и реестр открытых моделей (п. 4.6, 4.9, NFR-04.2)
        st = client.get("/api/v1/system/status")
        assert st.status_code == 200
        st_json = st.json()
        assert st_json["network_isolation"]["localhost_only"] is True
        assert st_json["ffmpeg"]["available"] is True

        models = client.get("/api/v1/models")
        assert models.status_code == 200
        assert len(models.json()) >= 6

        # 2. Валидация форматов FR-01
        bad = client.post(
            "/api/v1/recordings/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert bad.status_code == 400

        # 3. Загрузка реального WAV и создание демо-записи с высоким приоритетом (FR-01, FR-10)
        wav_path = generate_demo_lecture_wav(tmp_path / "sample.wav", duration_sec=8.0, sr=22050)
        with wav_path.open("rb") as fh:
            up_low = client.post(
                "/api/v1/recordings/upload",
                data={"title": "Фоновый семинар", "priority": "1", "pipeline_mode": "demo"},
                files={"file": ("sample.wav", fh, "audio/wav")},
            )
        assert up_low.status_code == 200
        low_id = up_low.json()["id"]

        up_high = client.post("/api/v1/recordings/demo?priority=3&title=Срочная лекция МНК")
        assert up_high.status_code == 200
        high_id = up_high.json()["id"]

        # 4. Проверка соблюдения приоритетов очереди: первой должна обработаться задача priority=3!
        worker = app.state.worker
        assert worker.process_one() is True

        rec_high = client.get(f"/api/v1/recordings/{high_id}").json()
        rec_low = client.get(f"/api/v1/recordings/{low_id}").json()
        assert rec_high["status"] == "completed"
        assert rec_low["status"] == "queued"
        assert rec_high["sample_rate"] == 16000
        assert rec_high["channels"] == 1

        # Проверяем метки «Спикер 1» / «Спикер 2», пословные таймкоды и автоисправление терминов!
        tr = rec_high["transcript"]
        assert tr is not None
        assert "Спикер 1" in tr["speakers"]
        assert len(tr["utterances"]) >= 4
        assert len(tr["utterances"][0]["words"]) > 3
        assert len(tr["meta"].get("term_corrections", [])) >= 2

        # 5. Полнотекстовый поиск на русском языке в любом регистре (FR-08)
        search_res = client.get("/api/v1/recordings", params={"q": "РЕГУЛЯРИЗАЦИЯ"})
        assert search_res.status_code == 200
        hits = search_res.json()
        assert len(hits) == 1
        assert hits[0]["id"] == high_id
        assert hits[0]["search_snippet"] is not None
        assert hits[0]["matched_timecode_sec"] is not None

        # 6. HTTP Range стриминг аудио для перемотки плеера ±0,5 с (FR-07)
        audio_range = client.get(
            f"/api/v1/recordings/{high_id}/audio",
            headers={"Range": "bytes=0-511"},
        )
        assert audio_range.status_code == 206
        assert len(audio_range.content) == 512

        # 7. Ручное переименование спикера (FR-04) и редактирование реплики с цензурой [НБЗ] (FR-07)
        ren = client.patch(
            f"/api/v1/recordings/{high_id}/speakers",
            json={"old_name": "Спикер 1", "new_name": "Доцент Долженкова М. Л."},
        )
        assert ren.status_code == 200
        assert ren.json()["updated_utterances"] >= 1

        first_utt_id = tr["utterances"][0]["id"]
        edit_res = client.patch(
            f"/api/v1/utterances/{first_utt_id}",
            json={"text": "В худшем случае это просто пиздец, коллеги."},
        )
        assert edit_res.status_code == 200
        edited_utt = edit_res.json()
        assert edited_utt["is_edited"] is True
        assert "худшем" in edited_utt["text"]
        assert "[НБЗ]" in edited_utt["text"]

        # 8. Экспорт TXT и JSON (FR-07)
        exp = client.get(f"/api/v1/recordings/{high_id}/export?format=txt")
        assert exp.status_code == 200
        assert "Доцент Долженкова М. Л." in exp.text
        exp_json = client.get(f"/api/v1/recordings/{high_id}/export?format=json")
        assert exp_json.status_code == 200
        assert exp_json.json()["recording_id"] == high_id

        # 8b. Редактор словарей (GET / PUT /api/v1/lexicon) и раздача SPA главной страницы
        lex_get = client.get("/api/v1/lexicon")
        assert lex_get.status_code == 200
        lex_put = client.put(
            "/api/v1/lexicon",
            json={
                "terms_text": lex_get.json()["terms_text"] + "\nнейросетевой классификатор\n",
                "badwords_text": "# user roots\n",
            },
        )
        assert lex_put.status_code == 200
        assert lex_put.json()["saved"] is True

        spa_root = client.get("/")
        assert spa_root.status_code == 200
        assert "АудиоАналитик" in spa_root.text

        # 9. Стенд оценки метрик и словари (NFR-01)
        ev = client.post(
            "/api/v1/metrics/evaluate",
            json={
                "reference": "В худшем случае тригонометрия точна!",
                "hypothesis": "в худшем случае триганометрия точна",
                "apply_postprocessing_preview": True,
                "case_sensitive": False,
            },
        )
        assert ev.status_code == 200
        ev_json = ev.json()
        assert ev_json["wer"]["rate"] == 0.0
        assert ev_json["cer"]["rate"] == 0.0
        assert ev_json["terms"]["f1"] == 1.0

        # 10. Восстановление прерванных задач после сбоя (FR-09)
        with app.state.db_manager.SessionLocal() as db:
            r_low_row = db.get(Recording, low_id)
            r_low_row.status = "transcribing"
            r_low_row.checkpoint_stage = "asr_partial"
            db.commit()
        recovered = worker.recover_interrupted_tasks()
        assert recovered == 1
        assert client.get(f"/api/v1/recordings/{low_id}").json()["status"] == "queued"

        # 11. Полное удаление записи и файлов с SSD за ≤ 20 с (FR-08, CON-04)
        del_res = client.delete(f"/api/v1/recordings/{high_id}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True
        assert del_res.json()["deleted_in_ms"] < 20_000
        assert client.get(f"/api/v1/recordings/{high_id}").status_code == 404

        # 12. Журнал безопасности 152-ФЗ и выгрузка CSV (NFR-04.3)
        logs = client.get("/api/v1/audit-logs").json()
        event_types = {row["event_type"] for row in logs}
        assert {"upload", "queue_start", "complete", "speaker_rename", "utterance_edit", "export", "delete"} <= event_types

        csv_exp = client.get("/api/v1/audit-logs/export?format=csv")
        assert csv_exp.status_code == 200
        assert "speaker_rename" in csv_exp.text
        json_exp = client.get("/api/v1/audit-logs/export?format=json")
        assert json_exp.status_code == 200
        assert len(json_exp.json()) >= 5

        retry_res = client.post(f"/api/v1/recordings/{low_id}/retry")
        assert retry_res.status_code == 200
