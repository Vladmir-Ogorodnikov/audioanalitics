from audioanalytic.formatting import build_turns, chunk_for_llm, fmt_ts, to_text
from audioanalytic.models import Segment, Transcript


def make():
    return Transcript("ru", [
        Segment(0, 5, "Здравствуйте.", "SPEAKER_00"),
        Segment(5.5, 9, "Начнём лекцию.", "SPEAKER_00"),
        Segment(12, 15, "Можно вопрос?", "SPEAKER_01"),
        Segment(3700, 3705, "Спасибо.", "SPEAKER_00"),
    ], "lec.mp3")


def test_fmt_ts():
    assert fmt_ts(3725.9) == "01:02:05"


def test_merge_same_speaker():
    turns = build_turns(make())
    assert len(turns) == 3
    assert turns[0].text == "Здравствуйте. Начнём лекцию."
    assert turns[0].end == 9


def test_text_has_timecodes_and_speakers():
    txt = to_text(make())
    assert txt.splitlines()[0] == "[00:00:00] SPEAKER_00: Здравствуйте. Начнём лекцию."
    assert "[01:01:40] SPEAKER_00: Спасибо." in txt


def test_chunking_keeps_turns_whole():
    chunks = chunk_for_llm(make(), max_chars=60)
    assert len(chunks) >= 2
    assert all(c.startswith("[") for c in chunks)
    assert "\n".join(chunks) == to_text(make()).rstrip("\n")


def test_roundtrip():
    t = make()
    assert Transcript.from_dict(t.to_dict()).segments[2].speaker == "SPEAKER_01"
