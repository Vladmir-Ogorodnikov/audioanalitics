"""Тесты чекпоинтов FR-09, меток «Спикер N» FR-04 и реальной предобработки FR-02."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from audioanalytic.checkpoint import load_checkpoint
from audioanalytic.config import SttConfig
from audioanalytic.models import UNKNOWN_SPEAKER, Segment, Transcript, normalize_speaker_labels
from audioanalytic.preprocess import preprocess, probe_audio
from audioanalytic.server.demo_audio import generate_demo_lecture_wav


def test_normalize_speaker_labels_to_russian():
    tr = Transcript(
        language="ru",
        segments=[
            Segment(0, 2, "Привет", "SPEAKER_01"),
            Segment(2, 4, "Вопрос", "SPEAKER_00"),
            Segment(4, 6, "Ответ", "SPEAKER_01"),
            Segment(6, 8, "Шум", UNKNOWN_SPEAKER),
        ],
    )
    mapping = normalize_speaker_labels(tr)
    assert mapping == {"SPEAKER_01": "Спикер 1", "SPEAKER_00": "Спикер 2"}
    assert [s.speaker for s in tr.segments] == ["Спикер 1", "Спикер 2", "Спикер 1", UNKNOWN_SPEAKER]


def test_real_ffmpeg_preprocess_16khz_mono(tmp_path: Path):
    src = generate_demo_lecture_wav(tmp_path / "raw_22k.wav", duration_sec=4.0, sr=22050)
    raw_info = probe_audio(src)
    assert raw_info["sample_rate"] == 22050

    dst = preprocess(src, dst=tmp_path / "norm.16k.wav")
    assert dst.is_file()
    info = probe_audio(dst)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert abs(info["duration"] - 4.0) < 0.25


def test_stt_resumes_from_5min_checkpoint_after_crash(tmp_path: Path, monkeypatch):
    """FR-09: При падении на середине записи повторный запуск продолжает с последнего чекпоинта."""
    wav_path = generate_demo_lecture_wav(tmp_path / "lecture.wav", duration_sec=15.0, sr=16000)
    ckpt_file = tmp_path / "lecture.ckpt.json"
    transcribed_offsets: list[int] = []
    should_crash = {"flag": True}

    wx = types.ModuleType("whisperx")

    def load_audio(path, **kwargs):
        import soundfile as sf
        data, _ = sf.read(path, dtype="float32")
        return data

    class FakeAsrModel:
        def transcribe(self, audio_piece, **kwargs):
            idx = len(transcribed_offsets)
            if should_crash["flag"] and idx == 1:
                raise RuntimeError("Simulated GPU OOM / driver reset on chunk 2")
            transcribed_offsets.append(idx)
            dur = float(audio_piece.shape[-1]) / 16000.0
            return {
                "segments": [
                    {
                        "start": 0.2,
                        "end": min(2.0, dur),
                        "text": f"Кусок лекции {idx + 1} триганометрия",
                        "words": [{"word": "триганометрия", "start": 0.2, "end": 1.0, "score": 0.9}],
                    }
                ]
            }

    wx.load_audio = load_audio
    wx.load_model = lambda *a, **k: FakeAsrModel()
    wx.load_align_model = lambda **k: (object(), {})
    wx.align = lambda segs, *a, **k: {"segments": segs}
    wx.assign_word_speakers = lambda diar, res: {
        **res,
        "segments": [{**s, "speaker": f"SPEAKER_{i % 2:02d}"} for i, s in enumerate(res["segments"])],
    }
    diar_mod = types.ModuleType("whisperx.diarize")
    diar_mod.DiarizationPipeline = lambda **k: (lambda aud, **kw: object())
    wx.diarize = diar_mod

    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diar_mod)

    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("тригонометрия\n", encoding="utf-8")

    cfg = SttConfig(
        device="cpu",
        compute_type="int8",
        preprocess=False,
        diarize=True,
        hf_token="hf_test",
        chunk_duration=5.0,  # 3 куска по 5 с на 15-секундном файле
        checkpoint_path=str(ckpt_file),
        terms_file=str(terms_file),
    )

    from audioanalytic.stt import transcribe

    # 1-й запуск падает на 2-м куске, но чекпоинт 1-го куска уже сохранён на диск!
    with pytest.raises(RuntimeError, match="Simulated GPU"):
        transcribe(wav_path, cfg)

    saved = load_checkpoint(ckpt_file)
    assert saved is not None
    assert saved.stage == "asr_partial"
    assert saved.completed_chunks == 1
    assert len(transcribed_offsets) == 1

    # 2-й запуск подхватывает чекпоинт и доделывает только оставшиеся 2 куска (итого 1 + 2 = 3 вызова ASR, а не 4!)
    should_crash["flag"] = False
    tr = transcribe(wav_path, cfg)
    assert len(transcribed_offsets) == 3
    assert len(tr.segments) == 3
    assert tr.segments[0].speaker == "Спикер 1"
    assert tr.segments[1].speaker == "Спикер 2"
    assert "тригонометрия" in tr.segments[0].text
    assert load_checkpoint(ckpt_file).stage == "completed"
