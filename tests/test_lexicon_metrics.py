"""Тесты лингвистического обеспечения и метрик (FR-03, NFR-01/02)."""
import json
from pathlib import Path

import pytest

from audioanalytic.lexicon import (PROFANITY_MARKER, apply_censorship,
                                   build_initial_prompt, censor_text,
                                   load_badwords, load_terms, term_coverage)
from audioanalytic.metrics import cer, der, normalize, wer
from audioanalytic.models import Segment, Transcript


# --- lexicon -------------------------------------------------------------------

def test_load_terms_skips_comments(tmp_path: Path):
    f = tmp_path / "terms.txt"
    f.write_text("# словарь\nтригонометрия\n\nматрица\n", encoding="utf-8")
    assert load_terms(f) == ["тригонометрия", "матрица"]


def test_initial_prompt_respects_max_chars():
    terms = [f"термин{i}" for i in range(100)]
    p = build_initial_prompt(terms, max_chars=50)
    assert p is not None and len(p) <= 50
    assert build_initial_prompt([]) is None


def test_censor_marks_profanity():
    roots = load_badwords(None)
    text, n = censor_text("Блин, ну ёлки-иголки, вот блин", {"блин"})
    assert n == 2 and text.count(PROFANITY_MARKER) == 2
    # падежные формы того же корня тоже маркируются
    t2, n2 = censor_text("Какой-то блядский вопрос", roots)
    assert n2 == 1 and PROFANITY_MARKER in t2


def test_apply_censorship_on_segments():
    segs = [Segment(0, 1, "Тут хуйня и пиздец"), Segment(1, 2, "Чистый текст")]
    n = apply_censorship(segs, load_badwords(None))
    assert n >= 2
    assert PROFANITY_MARKER in segs[0].text
    assert segs[1].text == "Чистый текст"


def test_term_coverage():
    cov = term_coverage("Матрица и тригонометрия. Опять матрица!", ["матрица", "теорема"])
    assert cov["terms_total"] == 2 and cov["terms_found"] == 1
    assert abs(cov["recall"] - 0.5) < 1e-9


# --- metrics ---------------------------------------------------------------------

def test_normalize_strips_marker_and_punct():
    assert normalize("[НБЗ] Привет, мир!") == "привет мир"


def test_wer_basic():
    r = wer("один два три четыре", "один два ТРИ четыре")
    assert r.errors == 1 and abs(r.rate - 0.25) < 1e-9
    assert r.substitutions == 1


def test_wer_ignores_nbz():
    r = wer("нормальный текст", "[НБЗ] нормальный текст")
    assert r.errors == 0


def test_cer_counts_chars():
    c = cer("мама", "рама")
    assert c.errors == 2


def test_der_perfect_match():
    ref = [(0.0, 1.0, "A"), (1.5, 3.0, "B")]
    hyp = [(0.0, 1.0, "SPEAKER_0"), (1.5, 3.0, "SPEAKER_1")]
    d = der(ref, hyp)
    assert d.rate == pytest.approx(0.0, abs=1e-6)


def test_der_swapped_speakers():
    ref = [(0.0, 2.0, "A"), (4.0, 6.0, "B")]
    hyp = [(0.0, 2.0, "B"), (4.0, 6.0, "A")]
    d = der(ref, hyp)
    assert d.rate > 0.5  # почти всё время — перепутанные спикеры


def test_der_missed_speech():
    ref = [(0.0, 2.0, "A"), (4.0, 6.0, "B")]
    hyp = [(0.0, 2.0, "A")]  # второй сегмент вообще не размечен
    d = der(ref, hyp)
    assert 0.2 < d.rate < 0.7


# --- CLI evaluate smoke -------------------------------------------------------

def test_cli_evaluate(tmp_path: Path, capsys):
    from audioanalytic.cli import main

    tr = Transcript(language="ru", segments=[Segment(0, 1, "привет мир")])
    tj = tmp_path / "transcript.json"
    tj.write_text(json.dumps(tr.to_dict(), ensure_ascii=False), encoding="utf-8")
    ref = tmp_path / "ref.txt"
    ref.write_text("привет мирок", encoding="utf-8")

    rc = main(["evaluate", str(tj), str(ref)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WER:" in out and "CER:" in out
