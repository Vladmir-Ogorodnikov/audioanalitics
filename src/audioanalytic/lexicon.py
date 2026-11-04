"""Лингвистическое обеспечение (FR-03): словарь терминов и маркировка ненормативной лексики.

* Словарь терминов — текстовый файл (по одному термину в строке, '#' — комментарий).
  Используется двояко:
    1) как initial prompt Whisper (модель «видит» трудные термины и чаще распознаёт их верно);
    2) для оценки полноты терминов и F1 после транскрибации.

* Матрица (ненормативная лексика) маркируется меткой [НБЗ]. Список ведётся отдельно
  от словаря терминов (как требует ТЗ, п. 4.10). Matching регистронезависимый,
  учитывает падежи по простому префикс-правилу (корень + окончание до 3 символов).
"""
from __future__ import annotations

import re
from pathlib import Path

PROFANITY_MARKER = "[НБЗ]"

# Базовый список: корни слов (без последней гласной/суффикса), чтобы покрыть падежи.
# Пользователь может дополнить своим файлом (badwords.txt).
DEFAULT_PROFANITY_ROOTS = {"ху", "пизд", "ёб", "еб", "бля", "суч", "говн", "мочал"}


def load_terms(path: str | Path) -> list[str]:
    """Читает словарь терминов: строка = термин, пустые строки и '#' пропускаются."""
    p = Path(path)
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def build_initial_prompt(terms: list[str], max_chars: int = 224) -> str | None:
    """Собирает initial_prompt из терминов (Whisper принимает ~≤224 символа).

    Формат — перечисление через запятую: модель склонна воспроизводить знакомые
    слова из контекста подсказки.
    """
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


def _word_hits(word: str, roots: set[str]) -> bool:
    """Слово считается совпавшим, если начинается с корня из списка и длиннее
    него не более чем на 4 символа (покрытие падежных форм: блядь/блядский)."""
    w = word.lower().strip(".,!?;:\"'()")
    if len(w) < 3:
        return False
    for r in roots:
        if len(w) >= len(r) and w.startswith(r) and len(w) - len(r) <= 4:
            return True
    return False


_TOKEN_RE = re.compile(r"\S+")


def censor_text(text: str, roots: set[str]) -> tuple[str, int]:
    """Заменяет каждое слово, совпавшее с корнем из roots, на метку [НБЗ].

    Возвращает (текст, число замен). Пунктуация при слове теряется — для
    транскрипта это приемлемо (важен факт наличия, а не форма).
    """
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        if _word_hits(m.group(0), roots):
            count += 1
            return PROFANITY_MARKER
        return m.group(0)

    return _TOKEN_RE.sub(repl, text), count


def apply_censorship(segments, roots: set[str]) -> int:
    """Маркирует [НБЗ] сегменты транскрипта на месте. Возвращает число замен."""
    total = 0
    for seg in segments:
        new_text, n = censor_text(seg.text, roots)
        if n:
            seg.text = new_text
            total += n
    return total


def term_coverage(transcript_text: str, terms: list[str]) -> dict:
    """Метрики полноты и F1 по словарю терминов (для отчётов NFR-01).

    Precision считать по определению нельзя без эталонных вхождений, поэтому
    здесь: recall (доля терминов, встречающихся в транскрипте) и hit-count.
    Полный F1 вычисляется на этапе приёмочных испытаний относительно ручной
    эталонной расшифровки (см. docs/metrics_protocol.md).
    """
    low = transcript_text.lower()
    hits = {t: low.count(t.lower()) for t in terms}
    found = sum(1 for c in hits.values() if c > 0)
    total = len(terms)
    return {
        "terms_total": total,
        "terms_found": found,
        "recall": (found / total) if total else 0.0,
        "hits": hits,
    }
