"""Тесты лингвистического обеспечения и метрик с учётом исправлений Спринта 1 (FR-03, NFR-01/02)."""
import json
from pathlib import Path

import pytest

from audioanalytic.lexicon import (
    PROFANITY_MARKER,
    apply_censorship,
    apply_terms_postprocessing,
    build_initial_prompt,
    censor_text,
    correct_terms_in_text,
    load_badwords,
    load_terms,
    term_coverage,
)
from audioanalytic.metrics import cer, der, normalize, term_f1, wer
from audioanalytic.models import UNKNOWN_SPEAKER, Segment, Transcript, Word


# --- lexicon & Sprint 1 fixes --------------------------------------------------

def test_load_terms_skips_comments_and_deduplicates(tmp_path: Path):
    f = tmp_path / "terms.txt"
    f.write_text("# словарь\nтригонометрия\nТригонометрия\n\nматрица\n", encoding="utf-8")
    assert load_terms(f) == ["тригонометрия", "матрица"]


def test_initial_prompt_respects_max_chars():
    terms = [f"термин{i}" for i in range(100)]
    p = build_initial_prompt(terms, max_chars=50)
    assert p is not None and len(p) <= 50
    assert build_initial_prompt([]) is None


def test_censor_marks_profanity_and_preserves_punctuation():
    roots = load_badwords(None)
    text, n = censor_text("Блин, ну ёлки-иголки, вот блин!", {"блин"})
    assert n == 2 and text.count(PROFANITY_MARKER) == 2
    assert text == "[НБЗ], ну ёлки-иголки, вот [НБЗ]!"

    t2, n2 = censor_text("Какой-то блядский вопрос и нахуй это", roots)
    assert n2 == 2 and t2.count(PROFANITY_MARKER) == 2


def test_censor_does_not_hit_safe_academic_words():
    """Исправление ошибки №2: слова на «ху-»/«сук-» («худший», «художник», «сукно») не маркируются [НБЗ]."""
    roots = load_badwords(None)
    academic = "В худшем случае художник рисует художественный эскиз на сукне в хуторе."
    cleaned, count = censor_text(academic, roots)
    assert count == 0
    assert cleaned == academic


def test_apply_censorship_censors_both_segment_text_and_words():
    """Исправление ошибки №3: нецензурная лексика скрывается и в Segment.words для плеера FR-07."""
    segs = [
        Segment(
            0,
            2,
            "Тут хуйня и пиздец!",
            words=[
                Word("Тут", 0.0, 0.4),
                Word("хуйня", 0.4, 1.0),
                Word("и", 1.0, 1.2),
                Word("пиздец!", 1.2, 2.0),
            ],
        ),
        Segment(2, 4, "В худшем случае чистый текст", words=[Word("худшем", 2.2, 2.8)]),
    ]
    n = apply_censorship(segs, load_badwords(None))
    assert n == 2
    assert PROFANITY_MARKER in segs[0].text
    assert segs[0].words[1].text == PROFANITY_MARKER
    assert segs[0].words[3].text == f"{PROFANITY_MARKER}!"
    assert segs[1].words[0].text == "худшем"


def test_terms_postprocessing_fixes_asr_typos():
    """Исправление ошибки №4: постобработка исправляет близкие опечатки ASR по словарю терминов."""
    terms = ["тригонометрия", "гомоскедастичность", "метод наименьших квадратов"]
    segs = [
        Segment(
            0,
            5,
            "Триганометрия, гомоскедастичнасть и метод наименьших квадратав важны.",
            words=[Word("Триганометрия,", 0.0, 1.0), Word("гомоскедастичнасть", 1.1, 2.5)],
        )
    ]
    corr = apply_terms_postprocessing(segs, terms)
    assert len(corr) == 3
    assert "Тригонометрия," in segs[0].text
    assert "гомоскедастичность" in segs[0].text
    assert "метод наименьших квадратов" in segs[0].text
    assert segs[0].words[0].text == "Тригонометрия,"
    assert segs[0].words[1].text == "гомоскедастичность"


def test_term_coverage_handles_inflections():
    cov = term_coverage("Матрицы и тригонометрии. Опять матрицу!", ["матрица", "теорема"])
    assert cov["terms_total"] == 2 and cov["terms_found"] == 1
    assert cov["hits"]["матрица"] == 2
    assert abs(cov["recall"] - 0.5) < 1e-9


# --- metrics & Sprint 1 fixes --------------------------------------------------

def test_normalize_strips_marker_and_punct():
    assert normalize("[НБЗ] Привет, мир!") == "привет мир"


def test_wer_and_cer_ignore_case_and_punctuation_by_default():
    """Исправление ошибки №1: регистр и пунктуация не завышают WER и CER."""
    ref = "Здравствуйте, коллеги! Начинаем лекцию: часть первая."
    hyp = "здравствуйте коллеги начинаем лекцию часть первая"
    assert wer(ref, hyp).rate == 0.0
    assert cer(ref, hyp).rate == 0.0
    # При явном флаге case_sensitive=True регистр учитывается
    assert wer(ref, hyp, case_sensitive=True).errors > 0


def test_wer_counts_real_word_errors():
    r = wer("один два три четыре", "один два пять четыре")
    assert r.errors == 1 and abs(r.rate - 0.25) < 1e-9
    assert r.substitutions == 1


def test_wer_ignores_nbz():
    r = wer("нормальный текст", "[НБЗ] нормальный текст")
    assert r.errors == 0


def test_cer_counts_chars():
    c = cer("мама, привет!", "рама привет")
    assert c.errors == 1 and abs(c.rate - (1 / 10)) < 1e-9


def test_term_f1_calculation():
    terms = ["тригонометрия", "регуляризация", "дисперсия"]
    ref = "Тригонометрия и регуляризация уменьшают дисперсию оценок."
    hyp = "тригонометрия и регуляризация уменьшают ошибку оценок"
    m = term_f1(ref, hyp, terms)
    # В эталоне 3 термина, в гипотезе верно распознаны 2, ложных срабатываний 0:
    # Precision = 1.0, Recall = 2/3, F1 = 0.8
    assert m.ref_occurrences == 3
    assert m.true_positives == 2
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(0.8)


def test_der_perfect_match():
    ref = [(0.0, 1.0, "A"), (1.5, 3.0, "B")]
    hyp = [(0.0, 1.0, UNKNOWN_SPEAKER), (1.5, 3.0, UNKNOWN_SPEAKER)]
    d = der(ref, hyp)
    assert d.rate == pytest.approx(0.0, abs=1e-6)


def test_der_swapped_speakers():
    ref = [(0.0, 2.0, "A"), (4.0, 6.0, "B")]
    hyp = [(0.0, 2.0, "B"), (4.0, 6.0, "A")]
    d = der(ref, hyp)
    assert d.rate > 0.5


def test_der_missed_speech():
    ref = [(0.0, 2.0, "A"), (4.0, 6.0, "B")]
    hyp = [(0.0, 2.0, "A")]
    d = der(ref, hyp)
    assert 0.2 < d.rate < 0.7


def test_cli_evaluate_with_term_f1(tmp_path: Path, capsys):
    from audioanalytic.cli import main

    tr = Transcript(language="ru", segments=[Segment(0, 1, "тригонометрия и матрица")])
    tj = tmp_path / "transcript.json"
    tj.write_text(json.dumps(tr.to_dict(), ensure_ascii=False), encoding="utf-8")
    ref = tmp_path / "ref.txt"
    ref.write_text("Тригонометрия и матрица!", encoding="utf-8")
    terms_f = tmp_path / "terms.txt"
    terms_f.write_text("тригонометрия\nматрица\n", encoding="utf-8")

    rc = main(["evaluate", str(tj), str(ref), "--terms", str(terms_f)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WER: 0.00%" in out
    assert "CER: 0.00%" in out
    assert "F1=1.000" in out
