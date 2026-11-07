"""Лингвистическое обеспечение (FR-03): словарь терминов и маркировка ненормативной лексики.

* Словарь терминов — текстовый файл (по одному термину в строке, '#' — комментарий).
  Используется трояко:
    1) как initial prompt Whisper (модель «видит» трудные термины и чаще распознаёт их верно);
    2) для постобработки транскрипта (исправление близких опечаток ASR по словарю);
    3) для оценки полноты терминов и F1 относительно эталона (NFR-01).

* Ненормативная лексика маркируется меткой [НБЗ] как в тексте сегмента, так и в
  пословной разметке (Segment.words) для интерактивного плеера (FR-07).
  Список ведётся отдельно от словаря терминов (п. 4.10 ТЗ).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

PROFANITY_MARKER = "[НБЗ]"

# Уточнённые корни (включая приставочные формы) без широкого корня «ху»,
# чтобы не задевать нейтральную академическую лексику («худший», «художник» и т.д.).
DEFAULT_PROFANITY_ROOTS = {
    "хуй", "хуе", "хуё", "хуи", "хуя", "хули",
    "наху", "поху", "охуе", "охуё",
    "пизд", "спизд",
    "ёб", "еба", "ебл", "ебн", "ебу", "ебё", "еби",
    "заеб", "заёб", "проеб", "проёб", "выеб", "выёб", "уеб", "уёб",
    "бля", "сук", "суч", "говн", "муда", "муди", "залуп",
}

# Белый список префиксов нейтральных слов, которые не должны маркироваться [НБЗ]
SAFE_WORD_PREFIXES = {
    "худож", "худш", "хутор", "худо", "хулиган",
    "сукн", "сучк", "сучон", "суккулент", "сукцесс",
    "ебонит",
}

_PUNCT_CHARS = ".,!?;:\"'()[]«»…—–-"
_TOKEN_RE = re.compile(r"\S+")


def load_terms(path: str | Path) -> list[str]:
    """Читает словарь терминов: строка = термин, пустые строки и '#' пропускаются."""
    p = Path(path)
    if not p.is_file():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if not t or t.startswith("#"):
            continue
        low = t.lower()
        if low not in seen:
            seen.add(low)
            out.append(t)
    return out


def build_initial_prompt(terms: list[str], max_chars: int = 224) -> str | None:
    """Собирает initial_prompt из терминов (Whisper принимает ~≤224 символа)."""
    if not terms:
        return None
    parts: list[str] = []
    size = 0
    for t in terms:
        add = (", " if parts else "") + t
        if size + len(add) > max_chars:
            break
        parts.append(t)
        size += len(add)
    return ", ".join(parts) if parts else None


def _roots_from_lines(lines: list[str]) -> set[str]:
    roots: set[str] = set()
    for ln in lines:
        w = ln.strip().lower()
        if not w or w.startswith("#"):
            continue
        roots.add(w)
    return roots


def load_badwords(path: str | Path | None) -> set[str]:
    """Корни нецензурных слов из файла (или дефолтный набор, если файла нет)."""
    if path is None:
        return set(DEFAULT_PROFANITY_ROOTS)
    p = Path(path)
    if not p.is_file():
        return set(DEFAULT_PROFANITY_ROOTS)
    user = _roots_from_lines(p.read_text(encoding="utf-8").splitlines())
    return set(DEFAULT_PROFANITY_ROOTS) | user


def _split_affixes(token: str) -> tuple[str, str, str]:
    """Отделяет ведущую и завершающую пунктуацию от ядра слова."""
    start = 0
    end = len(token)
    while start < end and token[start] in _PUNCT_CHARS:
        start += 1
    while end > start and token[end - 1] in _PUNCT_CHARS:
        end -= 1
    return token[:start], token[start:end], token[end:]


def _word_hits(word: str, roots: set[str]) -> bool:
    """Проверяет, начинается ли очищенное слово с запрещённого корня (с учётом белого списка)."""
    _, core, _ = _split_affixes(word)
    w = core.lower().replace("ё", "е")
    w_orig = core.lower()
    if len(w_orig) < 3:
        return False
    if any(w_orig.startswith(sw) or w.startswith(sw.replace("ё", "е")) for sw in SAFE_WORD_PREFIXES):
        return False
    return any(w_orig.startswith(r) for r in roots)


def censor_text(text: str, roots: set[str]) -> tuple[str, int]:
    """Заменяет каждое нецензурное слово на [НБЗ], сохраняя обрамляющую пунктуацию."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        tok = m.group(0)
        if tok == PROFANITY_MARKER or PROFANITY_MARKER in tok:
            return tok
        if _word_hits(tok, roots):
            count += 1
            lead, _, trail = _split_affixes(tok)
            return f"{lead}{PROFANITY_MARKER}{trail}"
        return tok

    return _TOKEN_RE.sub(repl, text), count


def apply_censorship(segments, roots: set[str]) -> int:
    """Маркирует [НБЗ] как текст сегментов, так и пословную разметку Segment.words."""
    total = 0
    for seg in segments:
        new_text, n = censor_text(seg.text, roots)
        if n:
            seg.text = new_text
            total += n
        for w in getattr(seg, "words", []) or []:
            w_new, wn = censor_text(w.text, roots)
            if wn:
                w.text = w_new
                if not n:
                    total += wn
    return total


def _preserve_case(original: str, replacement: str) -> str:
    if not original:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def correct_terms_in_text(
    text: str,
    terms: list[str],
    *,
    threshold: float = 0.82,
) -> tuple[str, list[dict[str, str]]]:
    """Исправляет близкие фонетические/орфографические опечатки ASR по словарю терминов.

    Поддерживает как однословные, так и многословные термины (до 4 слов).
    Возвращает ``(исправленный_текст, список_замен)``.
    """
    if not text.strip() or not terms:
        return text, []

    single_terms: list[str] = []
    multi_terms: list[list[str]] = []
    for raw in terms:
        parts = raw.strip().split()
        if len(parts) == 1 and len(parts[0]) >= 5:
            single_terms.append(parts[0])
        elif 2 <= len(parts) <= 4:
            multi_terms.append(parts)

    tokens = text.split(" ")
    corrections: list[dict[str, str]] = []

    # 1. Многословные термины (скользящее окно)
    for m_parts in multi_terms:
        k = len(m_parts)
        target_clean = " ".join(p.lower() for p in m_parts)
        if len(target_clean) < 8:
            continue
        i = 0
        while i <= len(tokens) - k:
            window_affixes = [_split_affixes(tokens[i + j]) for j in range(k)]
            if any(not core for _, core, _ in window_affixes):
                i += 1
                continue
            cand_clean = " ".join(core.lower() for _, core, _ in window_affixes)
            if cand_clean == target_clean:
                i += k
                continue
            if cand_clean[0] == target_clean[0] and abs(len(cand_clean) - len(target_clean)) <= 3:
                ratio = SequenceMatcher(None, cand_clean, target_clean).ratio()
                if ratio >= threshold:
                    orig_phrase = " ".join(core for _, core, _ in window_affixes)
                    for j, term_word in enumerate(m_parts):
                        lead, core, trail = window_affixes[j]
                        fixed = _preserve_case(core, term_word)
                        tokens[i + j] = f"{lead}{fixed}{trail}"
                    corrections.append({"from": orig_phrase, "to": " ".join(m_parts)})
                    i += k
                    continue
            i += 1

    # 2. Однословные термины
    for idx, tok in enumerate(tokens):
        lead, core, trail = _split_affixes(tok)
        if len(core) < 5 or core == PROFANITY_MARKER:
            continue
        low = core.lower()
        if any(low == t.lower() for t in single_terms):
            continue
        best_term: str | None = None
        best_score = threshold
        for term in single_terms:
            t_low = term.lower()
            if low[0] != t_low[0] or abs(len(low) - len(t_low)) > 2:
                continue
            # Если отличаются только падежные окончания у одного и того же корня — не трогаем!
            stem_len = max(4, len(t_low) - 2)
            if low[:stem_len] == t_low[:stem_len]:
                continue
            score = SequenceMatcher(None, low, t_low).ratio()
            if score >= best_score:
                best_score = score
                best_term = term
        if best_term is not None:
            fixed = _preserve_case(core, best_term)
            tokens[idx] = f"{lead}{fixed}{trail}"
            corrections.append({"from": core, "to": fixed})

    return " ".join(tokens), corrections


def apply_terms_postprocessing(
    segments,
    terms: list[str],
    *,
    threshold: float = 0.82,
) -> list[dict[str, str]]:
    """Применяет коррекцию опечаток по словарю терминов к сегментам и их словам."""
    all_corrections: list[dict[str, str]] = []
    if not terms:
        return all_corrections
    for seg in segments:
        new_text, corr = correct_terms_in_text(seg.text, terms, threshold=threshold)
        if corr:
            seg.text = new_text
            all_corrections.extend(corr)
        for w in getattr(seg, "words", []) or []:
            w_new, _ = correct_terms_in_text(w.text, terms, threshold=threshold)
            w.text = w_new
    return all_corrections


def russian_term_stem(word: str) -> str:
    """Приближённая основа русского слова для учёта падежных форм академических терминов."""
    w = word.lower().strip(_PUNCT_CHARS)
    if len(w) <= 4:
        return w
    for suffix in ("ами", "ями", "ого", "его", "ому", "ему", "иях", "ией", "иям",
                   "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ом", "ем", "ой", "ою",
                   "ую", "юю", "ия", "ие", "ии", "ию", "а", "я", "ы", "и", "у", "ю", "е", "о"):
        if len(w) - len(suffix) >= 4 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def count_term_occurrences(text: str, term: str) -> int:
    """Считает вхождения термина в текст с учётом морфологических окончаний русского языка."""
    clean_words = [w for w in re.sub(r"[^\w\s-]", " ", text.lower()).split() if w]
    term_words = [w for w in re.sub(r"[^\w\s-]", " ", term.lower()).split() if w]
    if not clean_words or not term_words:
        return 0
    term_stems = [russian_term_stem(tw) for tw in term_words]
    word_stems = [russian_term_stem(cw) for cw in clean_words]
    k = len(term_stems)
    hits = 0
    i = 0
    while i <= len(word_stems) - k:
        match = True
        for j in range(k):
            ts = term_stems[j]
            cw = clean_words[i + j]
            ws = word_stems[i + j]
            if len(term_words[j]) <= 4:
                if cw != term_words[j]:
                    match = False
                    break
            else:
                if ws != ts and not cw.startswith(ts):
                    match = False
                    break
        if match:
            hits += 1
            i += k
        else:
            i += 1
    return hits


def term_coverage(transcript_text: str, terms: list[str]) -> dict:
    """Метрика полноты присутствия терминов словаря в транскрипте (без эталона)."""
    hits = {t: count_term_occurrences(transcript_text, t) for t in terms}
    found = sum(1 for c in hits.values() if c > 0)
    total = len(terms)
    return {
        "terms_total": total,
        "terms_found": found,
        "total_occurrences": sum(hits.values()),
        "recall": (found / total) if total else 0.0,
        "hits": hits,
    }
