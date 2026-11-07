"""Метрики качества STT/диаризации (NFR-01, NFR-02) для приёмочных испытаний.

WER/CER считаются классическим расстоянием редактирования (Левенштейна) между
эталонной и машинной расшифровками после нормализации: приведение к нижнему
регистру, удаление пунктуации, разбивка на слова/символы. Метка [НБЗ] из
расчёта исключается с обеих сторон (иначе маркировка искусственно завысит WER).

Для терминов предметной области (табл. 17 ТЗ: полнота ≥ 85 %, F1 ≥ 0,88)
вычисляются Precision, Recall и F1 относительно эталонной расшифровки.

DER считается по временным интервалам: доля времени с ошибкой (пропуск речи,
ложное срабатывание, перепутанный спикер) относительно общего времени речи
эталона с защитной зоной collar = 0.25 с.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import count_term_occurrences
from .models import UNKNOWN_SPEAKER

_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]«»…\-—–]")
_NBZ_RE = re.compile(r"\[нбз\]", flags=re.IGNORECASE)


def normalize(text: str, *, lowercase: bool = True) -> str:
    """Удаление метки [НБЗ], пунктуации и нормализация регистра/пробелов."""
    t = _NBZ_RE.sub(" ", text)
    if lowercase:
        t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def words(text: str, *, lowercase: bool = True) -> list[str]:
    return normalize(text, lowercase=lowercase).split()


def chars(text: str, *, lowercase: bool = True) -> list[str]:
    return [c for c in normalize(text, lowercase=lowercase) if not c.isspace()]


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Возвращает (расстояние, замены, удаления, вставки) по алгоритму Левенштейна."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,          # удаление
                d[i][j - 1] + 1,          # вставка
                d[i - 1][j - 1] + cost,   # замена/совпадение
            )
    i, j = n, m
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return d[n][m], sub, dele, ins


@dataclass
class ErrorRate:
    rate: float
    errors: float
    total: float
    substitutions: float
    deletions: float
    insertions: float


@dataclass
class TermMetrics:
    """Метрики качества распознавания терминов предметной области (табл. 17 ТЗ)."""
    precision: float
    recall: float
    f1: float
    ref_occurrences: int
    hyp_occurrences: int
    true_positives: int
    per_term: dict[str, dict[str, int]]


def wer(reference: str, hypothesis: str, *, case_sensitive: bool = False) -> ErrorRate:
    """Word Error Rate после нормализации регистра, пунктуации и меток [НБЗ]."""
    r = words(reference, lowercase=not case_sensitive)
    h = words(hypothesis, lowercase=not case_sensitive)
    dist, s, d_, i = edit_distance(r, h)
    total = len(r)
    return ErrorRate(dist / total if total else (1.0 if h else 0.0), dist, total, s, d_, i)


def cer(reference: str, hypothesis: str, *, case_sensitive: bool = False) -> ErrorRate:
    """Character Error Rate после нормализации регистра, пунктуации и меток [НБЗ]."""
    r = chars(reference, lowercase=not case_sensitive)
    h = chars(hypothesis, lowercase=not case_sensitive)
    dist, s, d_, i = edit_distance(r, h)
    total = len(r)
    return ErrorRate(dist / total if total else (1.0 if h else 0.0), dist, total, s, d_, i)


def term_f1(reference: str, hypothesis: str, terms: list[str]) -> TermMetrics:
    """Вычисляет полноту (Recall), точность (Precision) и F1-меру по списку терминов (NFR-01).

    Сопоставляет число вхождений каждого термина в эталонной и машинной расшифровках:
      TP_t = min(count_ref(t), count_hyp(t))
      Recall = sum(TP_t) / sum(count_ref(t))
      Precision = sum(TP_t) / sum(count_hyp(t))
      F1 = 2 * P * R / (P + R)
    """
    ref_total = 0
    hyp_total = 0
    tp_total = 0
    per_term: dict[str, dict[str, int]] = {}

    for term in terms:
        rc = count_term_occurrences(reference, term)
        hc = count_term_occurrences(hypothesis, term)
        tp = min(rc, hc)
        ref_total += rc
        hyp_total += hc
        tp_total += tp
        if rc > 0 or hc > 0:
            per_term[term] = {"ref": rc, "hyp": hc, "tp": tp}

    if ref_total == 0 and hyp_total == 0:
        precision = recall = f1_val = 1.0
    else:
        precision = (tp_total / hyp_total) if hyp_total > 0 else 0.0
        recall = (tp_total / ref_total) if ref_total > 0 else 1.0
        f1_val = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return TermMetrics(
        precision=precision,
        recall=recall,
        f1=f1_val,
        ref_occurrences=ref_total,
        hyp_occurrences=hyp_total,
        true_positives=tp_total,
        per_term=per_term,
    )


# --- DER ---------------------------------------------------------------------

Interval = tuple[float, float, str]  # start, end, speaker


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    res: list[tuple[float, float]] = []
    b_m = _merge(b)
    for s, e in _merge(a):
        cur = [(s, e)]
        for bs, be in b_m:
            nxt: list[tuple[float, float]] = []
            for cs, ce in cur:
                if be <= cs or bs >= ce:
                    nxt.append((cs, ce))
                else:
                    if cs < bs:
                        nxt.append((cs, bs))
                    if ce > be:
                        nxt.append((be, ce))
            cur = nxt
        res.extend(cur)
    return res


def _total(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    am, bm = _merge(a), _merge(b)
    i = j = 0
    acc = 0.0
    while i < len(am) and j < len(bm):
        lo, hi = max(am[i][0], bm[j][0]), min(am[i][1], bm[j][1])
        if hi > lo:
            acc += hi - lo
        if am[i][1] <= bm[j][1]:
            i += 1
        else:
            j += 1
    return acc


def _scored(segs: list[Interval], collar: float) -> list[tuple[float, float]]:
    bounds = sorted({s for s, _, _ in segs} | {e for _, e, _ in segs})
    excl = [(max(0.0, b - collar), b + collar) for b in bounds]
    res: list[tuple[float, float]] = []
    for s, e, _ in segs:
        res.extend(_subtract([(s, e)], excl))
    return _merge(res)


def der(
    reference: list[Interval],
    hypothesis: list[Interval],
    collar: float = 0.25,
    mapping: dict[str, str] | None = None,
) -> ErrorRate:
    """DER по классической схеме md-eval.pl с collar 0.25 с."""
    ref_scored = _scored(reference, collar)
    hyp_scored = _scored(hypothesis, collar)
    total_ref = _total(ref_scored)
    if total_ref <= 0:
        return ErrorRate(0.0, 0, 0, 0, 0, 0)

    covered = _overlap(ref_scored, hyp_scored)
    miss = max(0.0, total_ref - covered)
    fa = max(0.0, _total(hyp_scored) - covered)
    conf = min(_confusion_scored(reference, hypothesis, collar, mapping), covered)

    errors = miss + fa + conf
    return ErrorRate(
        errors / total_ref,
        round(errors, 3),
        round(total_ref, 3),
        round(conf, 3),
        round(miss, 3),
        round(fa, 3),
    )


def _confusion_scored(
    reference: list[Interval],
    hypothesis: list[Interval],
    collar: float,
    mapping: dict[str, str] | None,
) -> float:
    hyp_names = {spk for _, _, spk in hypothesis}
    if hyp_names == {UNKNOWN_SPEAKER}:
        return 0.0
    if mapping:
        reference = [(s, e, mapping.get(nm, nm)) for s, e, nm in reference]

    step = 0.01
    lo = min([s for s, _, _ in reference] + [s for s, _, _ in hypothesis])
    hi = max([e for _, e, _ in reference] + [e for _, e, _ in hypothesis])
    n = int((hi - lo) / step) + 1
    ref_map: list[set[str]] = [set() for _ in range(n)]
    hyp_map: list[set[str]] = [set() for _ in range(n)]
    for s, e, nm in reference:
        for i in range(int((s - lo) / step), int((e - lo) / step)):
            if 0 <= i < n:
                ref_map[i].add(nm)
    for s, e, nm in hypothesis:
        for i in range(int((s - lo) / step), int((e - lo) / step)):
            if 0 <= i < n:
                hyp_map[i].add(nm)

    bounds = sorted(
        {s for s, _, _ in reference}
        | {e for _, e, _ in reference}
        | {s for s, _, _ in hypothesis}
        | {e for _, e, _ in hypothesis}
    )
    excl = [(b - collar, b + collar) for b in bounds]
    conf = 0.0
    for k, (rf, hf) in enumerate(zip(ref_map, hyp_map)):
        if not rf or not hf or rf == hf:
            continue
        t = lo + (k + 0.5) * step
        if any(a <= t < b for a, b in excl):
            continue
        shared = rf & hf
        conf += (len(rf) - len(shared)) / len(rf) * step
    return conf
