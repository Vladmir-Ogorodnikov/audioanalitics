# -*- coding: utf-8 -*-
"""Проверка ПОЛНОГО ЦИКЛА без GPU/моделей: подменяем whisperx заглушкой.

Запуск из корня проекта:  python test_full_cycle.py [путь_к.mp3]
Без аргумента — берётся data/lecture1.mp3.

Что проверяется на реальном коде конвейера (cli -> preprocess -> stt -> formatting):
  1. предобработка ffmpeg (16 кГц, моно, -16 LUFS, два прохода loudnorm);
  2. torchcodec-fallback: стт читает WAV напрямую, даже если load_audio «сломан»;
  3. --max-duration: обрезка аудио (деление на чанки для ASR);
  4. диаризация: спикеры проставляются по сегментам;
  5. выравнивание по словам (таймкоды слов);
  6. нарезка транскрипта на части для LLM (audioanalytic chunks).
"""
import sys
import types
from pathlib import Path


def make_fake_whisperx() -> types.ModuleType:
    wx = types.ModuleType("whisperx")

    def load_audio(path, *args, **kwargs):
        # Симулируем сломанный torchcodec на Windows — код должен сам прочитать WAV
        raise RuntimeError(
            "Could not load libtorchcodec. FFmpeg version 7: Could not find module "
            "'...torchcodec\\\\libtorchcodec_core7.dll' (or one of its dependencies)")

    class _Model:
        def transcribe(self, audio, batch_size=16, language=None, initial_prompt=None):
            dur = audio.shape[-1] / 16000.0
            segs = []
            t = 0.0
            i = 0
            while t < min(dur, 30.0):  # до 30 с достаточно для проверки
                end = min(t + 5.0, dur)
                words = [{"word": w, "start": t + j * 0.5, "end": t + j * 0.5 + 0.4}
                         for j, w in enumerate(["тест", "речь", "три", "четыре"])]
                segs.append({"start": t, "end": end, "text": f"Сегмент номер {i}. тест речь",
                             "words": words})
                t = end + 0.7  # пауза между сегментами
                i += 1
            return {"segments": segs}

    def load_model(model, device, compute_type=None, language=None):
        assert compute_type == "int8", f"ожидали int8 на CPU, получили {compute_type}"
        return _Model()

    def load_align_model(language_code=None, device=None):
        return object(), {"language": language_code}

    def align(segments, align_model, metadata, audio, device, return_char_alignments=False):
        for s in segments:
            if "words" not in s:
                n = max(1, int((s["end"] - s["start"]) * 2))
                s["words"] = [{"word": "слово", "start": s["start"] + j * 0.5,
                               "end": s["start"] + j * 0.5 + 0.4, "score": 0.9}
                              for j in range(n)]
        return {"segments": segments}

    def assign_word_speakers(diar, result):
        for i, s in enumerate(result["segments"]):
            s["speaker"] = f"SPEAKER_{i % 2:02d}"
        return result

    class DiarizationPipeline:
        def __init__(self, token=None, device=None):
            assert token, "нет HF_TOKEN"

        def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
            return object()  # объект-«диаризация», дальше использует assign_word_speakers

    diarize_mod = types.ModuleType("whisperx.diarize")
    diarize_mod.DiarizationPipeline = DiarizationPipeline
    wx.load_audio = load_audio
    wx.load_model = load_model
    wx.load_align_model = load_align_model
    wx.align = align
    wx.assign_word_speakers = assign_word_speakers
    wx.diarize = diarize_mod
    return wx


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/lecture1.mp3")
    if not src.is_file():
        print(f"Файл не найден: {src}")
        return 2

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    fake = make_fake_whisperx()
    sys.modules["whisperx"] = fake
    sys.modules["whisperx.diarize"] = fake.diarize

    import os
    os.environ.setdefault("HF_TOKEN", "hf_fake_token_for_test")

    from audioanalytic.cli import main as cli_main

    print("=" * 60)
    print("ТЕСТ 1: полный цикл ЧЕРЕЗ CLI: audioanalytic check (диаризация ВКЛ)")
    print("=" * 60)
    rc = cli_main(["check", str(src), "--cpu", "--max-duration", "20",
                   "--out", "out_check", "--model", "tiny"])
    print(f"\nrc(check) = {rc}")
    assert rc == 0, "check с диаризацией должен проходить"

    print("\n" + "=" * 60)
    print("ТЕСТ 2: cycle без диаризации (--no-diarize)")
    print("=" * 60)
    rc2 = cli_main(["check", str(src), "--cpu", "--max-duration", "10",
                    "--no-diarize", "--out", "out_check_nd", "--model", "tiny"])
    print(f"\nrc(check --no-diarize) = {rc2}")
    assert rc2 == 0

    print("\n" + "=" * 60)
    print("ТЕСТ 3: команда chunks по результату")
    print("=" * 60)
    tr_json = Path("out_check") / src.stem / "transcript.json"
    rc3 = cli_main(["chunks", str(tr_json), "--max-chars", "200"])
    print(f"rc(chunks) = {rc3}")
    assert rc3 == 0
    n = len(list((tr_json.parent / "llm_chunks").glob("chunk_*.txt")))
    assert n >= 1, "чанки не созданы"

    print("\nВСЕ ПРОВЕРКИ ПОЛНОГО ЦИКЛА ПРОЙДЕНЫ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
