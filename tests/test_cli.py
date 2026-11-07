"""Смоук-тесты CLI без тяжёлых зависимостей (whisperx/torch не нужны)."""
import json
from pathlib import Path

from audioanalytic.cli import main
from audioanalytic.models import Segment, Transcript


def test_cli_help(capsys):
    import pytest
    with pytest.raises(SystemExit):  # argparse завершает --help через SystemExit
        main(["--help"])
    out = capsys.readouterr().out
    assert "transcribe" in out and "chunks" in out and "doctor" in out


def test_transcribe_rejects_bad_format(tmp_path, capsys):
    f = tmp_path / "notes.txt"
    f.write_text("текст", encoding="utf-8")
    assert main(["transcribe", str(f)]) == 2
    assert "не поддерживается" in capsys.readouterr().err


def test_transcribe_missing_file(tmp_path, capsys):
    assert main(["transcribe", str(tmp_path / "nope.mp3")]) == 2
    assert "не найден" in capsys.readouterr().err


def test_chunks_command(tmp_path):
    tr = Transcript("ru", [
        Segment(0, 5, "Первая реплика.", "SPEAKER_00"),
        Segment(10, 15, "Вторая реплика.", "SPEAKER_01"),
    ], "lec.mp3")
    p = tmp_path / "transcript.json"
    p.write_text(json.dumps(tr.to_dict(), ensure_ascii=False), encoding="utf-8")

    assert main(["chunks", str(p), "--max-chars", "40"]) == 0
    chunks = sorted((tmp_path / "llm_chunks").glob("chunk_*.txt"))
    assert chunks, "файлы частей не созданы"
    text = "\n".join(c.read_text(encoding="utf-8") for c in chunks)
    assert "Первая реплика." in text and "[00:00:10] SPEAKER_01" in text


def test_doctor_reports_disk(tmp_path, capsys, monkeypatch):
    # doctor не должен падать и обязан показать строки про диск/кэши
    assert main(["doctor"]) in (0, 1)
    out = capsys.readouterr().out
    assert "Диск" in out and "Кэш pip" in out


def test_ensure_disk_space_message(monkeypatch, capsys):
    import pytest
    from audioanalytic import winfix

    monkeypatch.setattr(winfix, "free_gb", lambda path=None: 1.0)
    with pytest.raises(RuntimeError) as e:
        winfix.ensure_disk_space(5, "тест")
    assert "PIP_CACHE_DIR" in str(e.value)


def test_load_dotenv_and_low_vram(tmp_path, monkeypatch):
    from audioanalytic.config import SttConfig, load_dotenv

    env_f = tmp_path / ".env"
    env_f.write_text("# comment\nCUSTOM_TEST_KEY='hello_val'\n", encoding="utf-8")
    monkeypatch.delenv("CUSTOM_TEST_KEY", raising=False)
    load_dotenv(env_f)
    import os
    assert os.environ.get("CUSTOM_TEST_KEY") == "hello_val"

    cfg = SttConfig.low_vram()
    assert cfg.model == "large-v3-turbo"
    assert cfg.compute_type == "int8_float16"


def test_cli_check_full_cycle_with_mock_whisperx(tmp_path, monkeypatch):
    import sys
    from audioanalytic.server.demo_audio import generate_demo_lecture_wav
    from test_full_cycle import make_fake_whisperx

    wav = generate_demo_lecture_wav(tmp_path / "lec.wav", duration_sec=6.0, sr=16000)
    fake = make_fake_whisperx()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake.diarize)
    monkeypatch.setenv("HF_TOKEN", "hf_fake")

    out_dir = tmp_path / "out_check"
    rc = main(["check", str(wav), "--cpu", "--max-duration", "5", "--out", str(out_dir), "--model", "tiny"])
    assert rc == 0
    assert (out_dir / "lec" / "transcript.json").is_file()
