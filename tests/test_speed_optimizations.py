"""Тесты ускорения обработки: кэш моделей, автоподбор под CPU/GPU, быстрый loudnorm и миграция БД."""
from __future__ import annotations

import sqlite3
import sys
import time
import types
from pathlib import Path

from fastapi.testclient import TestClient

from audioanalytic.config import ServerSettings, SttConfig
from audioanalytic.model_cache import GLOBAL_MODEL_CACHE, ModelCache
from audioanalytic.preprocess import preprocess, probe_audio
from audioanalytic.server.app import create_app
from audioanalytic.server.database import DatabaseManager
from audioanalytic.server.demo_audio import generate_demo_lecture_wav
from audioanalytic.stt import transcribe


def test_model_cache_reuse_exclusive_oom_and_idle_eviction():
    cache = ModelCache()
    calls = {"a": 0, "b": 0}

    obj1 = cache.get_or_load(("asr", "small"), lambda: (calls.__setitem__("a", calls["a"] + 1) or "M1"))
    obj2 = cache.get_or_load(("asr", "small"), lambda: (calls.__setitem__("a", calls["a"] + 1) or "M1_dup"))
    assert obj1 == "M1" and obj2 == "M1"
    assert calls["a"] == 1
    assert cache.size == 1

    # Режим exclusive (для GPU 8 ГБ CON-03) выгружает предыдущую модель перед загрузкой следующей
    obj_b = cache.get_or_load(
        ("align", "ru"),
        lambda: (calls.__setitem__("b", calls["b"] + 1) or "ALIGN"),
        exclusive=True,
    )
    assert obj_b == "ALIGN"
    assert cache.size == 1

    # При CUDA OOM кэш автоматически очищается и загрузка повторяется успешно
    oom_attempt = {"n": 0}

    def flaky_loader():
        oom_attempt["n"] += 1
        if oom_attempt["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "RECOVERED"

    res = cache.get_or_load(("diar", "cuda"), flaky_loader)
    assert res == "RECOVERED"
    assert oom_attempt["n"] == 2

    # Выгрузка при простое очереди
    assert cache.evict_if_idle(max_idle_sec=60.0) == 0
    time.sleep(0.03)
    assert cache.evict_if_idle(max_idle_sec=0.01) >= 1
    assert cache.size == 0


def test_warm_cache_and_cpu_auto_small_across_consecutive_files(tmp_path: Path, monkeypatch):
    """Три короткие записи подряд загружают веса ASR, выравнивания и диаризации ровно 1 раз."""
    GLOBAL_MODEL_CACHE.clear()
    wav1 = generate_demo_lecture_wav(tmp_path / "clip1.wav", duration_sec=3.0, sr=16000)
    wav2 = generate_demo_lecture_wav(tmp_path / "clip2.wav", duration_sec=3.0, sr=16000)
    wav3 = generate_demo_lecture_wav(tmp_path / "clip3.wav", duration_sec=3.0, sr=16000)

    loads = {"asr": 0, "align": 0, "diarize": 0, "models_requested": [], "asr_options": []}

    wx = types.ModuleType("whisperx")

    class FakeModel:
        def transcribe(self, aud, **kw):
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Проверка кэша моделей",
                        "words": [{"word": "Проверка", "start": 0.0, "end": 0.8, "score": 0.95}],
                    }
                ]
            }

    def load_model(model_name, device, compute_type="int8", language="ru", asr_options=None, threads=4):
        loads["asr"] += 1
        loads["models_requested"].append((model_name, device, compute_type))
        loads["asr_options"].append(asr_options)
        return FakeModel()

    def load_align_model(language_code="ru", device="cpu"):
        loads["align"] += 1
        return (object(), {})

    wx.load_audio = lambda p, **kw: __import__("soundfile").read(p, dtype="float32")[0]
    wx.load_model = load_model
    wx.load_align_model = load_align_model
    wx.align = lambda segs, *a, **k: {"segments": segs}
    wx.assign_word_speakers = lambda diar, res: {
        **res,
        "segments": [{**s, "speaker": "SPEAKER_00"} for s in res["segments"]],
    }

    diar_mod = types.ModuleType("whisperx.diarize")

    def make_diar_pipe(**kw):
        loads["diarize"] += 1
        return lambda aud, **d_kw: object()

    diar_mod.DiarizationPipeline = make_diar_pipe
    wx.diarize = diar_mod

    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diar_mod)

    cfg = SttConfig.from_preset(
        "auto",
        device="cpu",
        keep_models_warm=True,
        diarize=True,
        hf_token="hf_dummy",
        preprocess=False,
    )

    for wav in (wav1, wav2, wav3):
        tr = transcribe(wav, cfg)
        assert len(tr.segments) == 1
        assert tr.segments[0].speaker == "Спикер 1"
        assert tr.meta["model_used"] == "small"

    # Все три тяжёлые модели загрузились в память ровно 1 раз на 3 файла!
    assert loads["asr"] == 1
    assert loads["align"] == 1
    assert loads["diarize"] == 1
    assert loads["models_requested"][0] == ("small", "cpu", "int8")
    assert loads["asr_options"][0]["beam_size"] == 2
    assert loads["asr_options"][0]["temperatures"] == [0.0, 0.2]

    # Если же выбран профиль "accurate", даже на CPU сохраняется явно запрошенная large-v3
    cfg_acc = SttConfig.from_preset(
        "accurate",
        device="cpu",
        keep_models_warm=False,
        diarize=False,
        preprocess=False,
    )
    tr_acc = transcribe(wav1, cfg_acc)
    assert tr_acc.meta["model_used"] == "large-v3"
    GLOBAL_MODEL_CACHE.clear()


def test_single_pass_and_two_pass_preprocess_both_output_16k_mono(tmp_path: Path):
    src = generate_demo_lecture_wav(tmp_path / "raw_44k.wav", duration_sec=3.0, sr=44100)
    fast_dst = preprocess(src, dst=tmp_path / "fast.16k.wav", two_pass=False)
    twopass_dst = preprocess(src, dst=tmp_path / "twopass.16k.wav", two_pass=True)

    for p in (fast_dst, twopass_dst):
        info = probe_audio(p)
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert abs(info["duration"] - 3.0) < 0.25


def test_auto_migration_and_speed_preset_upload(tmp_path: Path):
    """Старая локальная БД без колонки speed_preset автоматически мигрируется при запуске."""
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title VARCHAR(255) NOT NULL,
            original_filename VARCHAR(255) NOT NULL,
            file_format VARCHAR(16) NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            duration_sec FLOAT NOT NULL DEFAULT 0.0,
            sample_rate INTEGER NOT NULL DEFAULT 16000,
            channels INTEGER NOT NULL DEFAULT 1,
            original_path TEXT NOT NULL,
            preprocessed_path TEXT,
            checkpoint_path TEXT,
            status VARCHAR(32) NOT NULL DEFAULT 'queued',
            priority INTEGER NOT NULL DEFAULT 2,
            progress_pct FLOAT NOT NULL DEFAULT 0.0,
            stage_message VARCHAR(512) NOT NULL DEFAULT '',
            checkpoint_stage VARCHAR(64) NOT NULL DEFAULT 'init',
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            language VARCHAR(16) NOT NULL DEFAULT 'ru',
            requested_speakers INTEGER,
            diarize_enabled BOOLEAN NOT NULL DEFAULT 1,
            low_vram BOOLEAN NOT NULL DEFAULT 0,
            pipeline_mode VARCHAR(32) NOT NULL DEFAULT 'auto',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            completed_at DATETIME,
            processing_time_sec FLOAT
        )
        """
    )
    conn.commit()
    conn.close()

    settings = ServerSettings(
        host="127.0.0.1",
        port=8099,
        data_dir=tmp_path,
        database_url=f"sqlite:///{db_file.as_posix()}",
    )
    dbm = DatabaseManager(settings)
    dbm.init_db()

    app = create_app(settings=settings, start_worker=False)
    wav_path = generate_demo_lecture_wav(tmp_path / "short.wav", duration_sec=2.0)
    with TestClient(app) as client:
        with wav_path.open("rb") as fh:
            up = client.post(
                "/api/v1/recordings/upload",
                files={"file": ("short.wav", fh, "audio/wav")},
                data={"title": "Быстрый тест", "speed_preset": "fast", "pipeline_mode": "demo"},
            )
        assert up.status_code == 200
        rec_id = up.json()["id"]
        det = client.get(f"/api/v1/recordings/{rec_id}")
        assert det.status_code == 200
        assert det.json()["speed_preset"] == "fast"
