"""Метрики качества STT/диаризации (NFR-01, NFR-02) для приёмочных испытаний.

WER/CER считаются классическим расстоянием редактирования (Левенштейна) между
эталонной и машинной расшифровками после нормализации: приведение к нижнему
регистру, удаление пунктуации, разбивка на слова/символы. Метка [НБЗ] из
расчёта исключается с обеих сторон (иначе маркировка искусственно завысит WER).

DER считается по временным интервалам: доля времени с ошибкой (пропуск речи,
ложное срабатывание, перепутанный спикер) относительно общего времени речи
эталона — упрощённая реализация с collar 0.25 с (см. docs/metrics_protocol.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import UNKNOWN_SPEAKER

_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]«»…\-—–]")


def normalize(text: str) -> str:
    """Нижний регистр, удаление метки [НБЗ] и пунктуации.

    Метка удаляется до снятия пунктуации, иначе скобки «[]» разобьются на
    отдельные токены. Пробелы схлопываются, чтобы не появлялись пустые слова.
    """
    t = re.sub(r"\[нбз\]", " ", text.lower(), flags=re.IGNORECASE)
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def words(text: str) -> list[str]:
    return normalize(text).split()


def chars(text: str) -> list[str]:
    return [c for c in normalize(text) if not c.isspace()]


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Возвращает (расстояние, замены, удаления, вставки) по алгоритму Левенштейна
    с бэктрейсингом. O(n*m), достаточно для фрагментов до нескольких тысяч слов.

    Сравнение чувствительно к регистру: WER/CER считаются на словах как есть
    (TTR-стиль); нормализация регистра — ответственность normalize().

    При бэктрейсинге сначала проверяется diagonal (замена/совпадение), затем
    удаление и вставка — иначе путь с заменой ошибочно разлагается на пару
    «удаление + вставка» и статистика sub/ins искажается при равной стоимости.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # удаление
                          d[i][j - 1] + 1,          # вставка
                          d[i - 1][j - 1] + cost)   # замена/совпадение
    # бэктрейсинг
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
    errors: int
    total: int
    substitutions: int
    deletions: int
    insertions: int


def wer(reference: str, hypothesis: str) -> ErrorRate:
    # WER считается без нормализации регистра (TTR-стиль): normalize() ниже
    # не вызывается, только удаление метки [НБЗ] и пунктуации.
    r = _for_scoring(reference).split()
    h = _for_scoring(hypothesis).split()
    dist, s, d_, i = edit_distance(r, h)
    total = len(r)
    return ErrorRate(dist / total if total else 0.0, dist, total, s, d_, i)


def cer(reference: str, hypothesis: str) -> ErrorRate:
    # CER — по символам исходного текста с учётом регистра (TTR-стиль);
    # метка [НБЗ] вырезается целиком, чтобы маркировка не завышала CER.
    r = _chars_raw(reference)
    h = _chars_raw(hypothesis)
    dist, s, d_, i = edit_distance(r, h)
    total = len(r)
    return ErrorRate(dist / total if total else 0.0, dist, total, s, d_, i)


_NBZ_RE = re.compile(r"\[нбз\]", flags=re.IGNORECASE)


def _for_scoring(text: str) -> str:
    """Для WER: нижний регистр НЕ приводим; убираем только метку [НБЗ] и пунктуацию."""
    t = _NBZ_RE.sub(" ", text)
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def _chars_raw(text: str) -> list[str]:
    t = _NBZ_RE.sub("", text)
    return [c for c in t if not c.isspace()]


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
    """Множественное вычитание интервалов: a \\ b (оба списка — любые, нормализуем сами)."""
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
    """Суммарное время пересечения двух списков интервалов (линейный sweep)."""
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
    """Интервалы речи без «защитных зон» ±collar вокруг ГЛАВНЫХ границ сегментов.

    Как в pyannote: из области оценки исключаются только границы самих сегментов
    (каждого списка по отдельности), а не границы каждого спикера в отдельности.
    """
    bounds = sorted({s for s, _, _ in segs} | {e for _, e, _ in segs})
    excl = [(max(0.0, b - collar), b + collar) for b in bounds]
    res: list[tuple[float, float]] = []
    for s, e, _ in segs:
        res.extend(_subtract([(s, e)], excl))
    return _merge(res)


def _scored_intervals(intervals: list[tuple[float, float]],
                      bounds: list[float], collar: float) -> list[tuple[float, float]]:
    """То же, что _scored, но для плоского списка интервалов с явным списком
    границ, вокруг которых вырезаются защитные зоны (нужно для confusion:
    границы берём от ПОЛНОГО набора сегментов, а не от одного имени)."""
    excl = [(max(0.0, b - collar), b + collar) for b in sorted(set(bounds))]
    res: list[tuple[float, float]] = []
    for s, e in intervals:
        res.extend(_subtract([(s, e)], excl))
    return _merge(res)


def der(reference: list[Interval], hypothesis: list[Interval],
        collar: float = 0.25) -> ErrorRate:
    """Упрощённый DER (как в pyannote, но с полным перебором имён спикеров —
    при ≤ 3 спикеров это ≤ 6 перестановок):
      miss        — эталонная речь (после collar), не покрытая ни одной гипотезой;
      false alarm — гипотетическая речь (после collar), не покрытая эталоном;
      confusion   — покрытое время, при котором имена спикеров не совпадают
                    (берётся лучшее сопоставление имён, чтобы тривиальная
                    путаница меток SPEAKER_0/1 не считалась ошибкой).
    collar (сек) вырезается вокруг границ сегментов с обеих сторон — стандартная
    практика pyannote: точность границ зависит от VAD, а не от диаризации.
    """
    ref_scored = _scored(reference, collar)
    hyp_scored = _scored(hypothesis, collar)
    total_ref = _total(ref_scored)
    if total_ref <= 0:
        return ErrorRate(0.0, 0, 0, 0, 0, 0)

    covered = _overlap(ref_scored, hyp_scored)
    miss = max(0.0, total_ref - covered)
    fa = max(0.0, _total(hyp_scored) - covered)

    # лучшее сопоставление имён: максимум времени, где ref-имя отображено на
    # hyp-имя и интервалы одного и того же имени пересекаются в scored-области.
    # Перебираются ВСЕ полные функции (для ≤3 спикеров — ≤ 9 вариантов): при
    # путанице меток оптимальное отображение может быть неинъективным (оба
    # эталонных имени показываются на одну гипотетскую метку) — перебор по
    # перестановкам такое решение не рассматривает и занижает DER до 0.
    # Ограничение matched <= covered гарантирует корректность confusion.
    # Если гипотеза содержит только безымянную речь (UNKNOWN_SPEAKER) — имена не
    # путались, просто диаризация не выполнялась: confusion = 0.
    ref_names = sorted({spk for _, _, spk in reference})
    hyp_names = sorted({spk for _, _, spk in hypothesis})
    import itertools
    matched = 0.0
    if hyp_names == [UNKNOWN_SPEAKER]:
        matched = covered
    elif len(hyp_names) >= len(ref_names) and ref_names:
        ref_bounds = sorted({s for s, _, _ in reference} | {e for _, e, _ in reference})
        hyp_bounds = sorted({s for s, _, _ in hypothesis} | {e for _, e, _ in hypothesis})
        for mapping_tuple in itertools.product(hyp_names, repeat=len(ref_names)):
            mapping = dict(zip(ref_names, mapping_tuple))
            t = 0.0
            for rname in ref_names:
                rint = [(s, e) for s, e, nm in reference if nm == rname]
                hint = [(s, e) for s, e, nm in hypothesis if nm == mapping[rname]]
                rs = _merge(_intersect(rint, _scored_intervals(hint, hyp_bounds, collar)))
                hs_ = _merge(_intersect(hint, _scored_intervals(rint, ref_bounds, collar)))
                t += _overlap(rs, hs_)
            matched = max(matched, t)
    confusion = max(0.0, min(covered, covered - matched))

    errors = miss + fa + confusion
    return ErrorRate(errors / total_ref, round(errors, 3), round(total_ref, 3),
                     round(confusion, 3), round(miss, 3), round(fa, 3))


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s1, e1 in _merge(a):
        for s2, e2 in _merge(b):
            lo, hi = max(s1, s2), min(e1, e2)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)
