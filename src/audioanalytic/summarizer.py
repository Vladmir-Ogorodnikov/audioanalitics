"""Аналитический контур Спринта 2 (FR-05, FR-06): конспектирование, верификация таймкодов и NER.

Поддерживает:
  * локальную языковую модель через Ollama (по умолчанию ``qwen2.5:3b`` на ``127.0.0.1:11434/v1``)
    с полным соблюдением сетевой изоляции ``NFR-04.2`` и ``152-ФЗ``;
  * любой OpenAI-совместимый сервер (LM Studio, vLLM, GigaChat, OpenRouter и др.);
  * автоматический Map-Reduce для длинных лекций (> 12 000 символов);
  * привязку каждого тезиса к реальному таймкоду реплики и расчёт ``confidence`` / ``needs_review``
    для защиты от галлюцинаций LLM (FR-06);
  * автономный алгоритмический резервный режим, если LLM-сервис временно не запущен.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import LlmConfig
from .formatting import fmt_ts

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")
_TIMECODE_RE = re.compile(r"\[?(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]?")
_YEAR_DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)(?:\s+\d{4})?|(?:18|19|20)\d{2}\s*(?:г\.|год[ауе]?))\b",
    re.IGNORECASE,
)
_INITIALS_PERSON_RE = re.compile(r"\b[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]{2,}\b")
_NAMED_LAW_RE = re.compile(
    r"(?:теорем[аыеу]|метод[ауе]?|формул[аыеу]|закон[ауе]?|неравенств[оау]|критери[йяю]|"
    r"распределени[еяю]|преобразовани[еяю])\s+([А-ЯЁ][а-яё]{2,}(?:\s*[—\-]\s*[А-ЯЁ][а-яё]{2,})?)"
)
_TASK_HINTS = (
    "задач", "задани", "домашн", "к следующ", "подготов", "сдать", "дедлайн",
    "прочита", "решить", "разобрать", "контрольн", "коллоквиум", "семинар", "лабораторн",
)

_SYSTEM_PROMPT_RU = (
    "Ты — академический ассистент системы «АудиоАналитик ИИ». "
    "Твоя задача — составить точный структурированный конспект учебной лекции или семинара "
    "СТРОГО по предоставленному тексту расшифровки на русском языке, ничего не выдумывая от себя.\n"
    "Верни ТОЛЬКО валидный JSON-объект следующей структуры:\n"
    "{\n"
    '  "goal": "Краткая цель или основная тема занятия (1–3 предложения)",\n'
    '  "key_concepts": [\n'
    '    {"term": "Термин", "definition": "Краткое пояснение из лекции"}\n'
    "  ],\n"
    '  "theses": [\n'
    '    {"timecode": "[00:01:15]", "text": "Ёмкий тезис лекции с сохранением сути"}\n'
    "  ],\n"
    '  "tasks_decisions": [\n'
    '    "Поставленная задача, домашнее задание, литература или орг. решение (если были в тексте)"\n'
    "  ],\n"
    '  "self_check_questions": [\n'
    '    "Контрольный вопрос для самопроверки по материалу занятия"\n'
    "  ],\n"
    '  "entities": {\n'
    '    "persons": ["Упомянутые учёные / авторы"],\n'
    '    "dates": ["Упомянутые даты / годы / сроки"],\n'
    '    "terms": ["Ключевые научные термины"]\n'
    "  }\n"
    "}\n"
    "Правила:\n"
    "1. В поле theses указывай реальные таймкоды [ЧЧ:ММ:СС] из начала соответствующих реплик.\n"
    "2. Выдели от 3 до 8 ключевых тезисов и от 3 до 5 вопросов для самопроверки.\n"
    "3. Если организационных задач или домашних заданий в записи не было, оставь tasks_decisions пустым списком [].\n"
    "4. Пиши грамотным русским языком, без markdown-обёрток вокруг JSON."
)


@dataclass
class ThesisItemData:
    seq_num: int
    text: str
    timecode_sec: float
    timecode_formatted: str
    confidence: float = 0.9
    needs_review: bool = False


@dataclass
class SummaryData:
    model_name: str
    goal: str
    key_concepts: list[dict[str, str]] = field(default_factory=list)
    theses: list[ThesisItemData] = field(default_factory=list)
    tasks_decisions: list[str] = field(default_factory=list)
    self_check_questions: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    used_fallback: bool = False

    def build_search_text(self) -> str:
        parts: list[str] = [self.goal]
        for c in self.key_concepts:
            parts.append(f"{c.get('term', '')}: {c.get('definition', '')}")
        for t in self.theses:
            parts.append(t.text)
        parts.extend(self.tasks_decisions)
        parts.extend(self.self_check_questions)
        for vals in self.entities.values():
            if isinstance(vals, list):
                parts.extend(str(v) for v in vals)
        return "\n".join(p.strip() for p in parts if p and p.strip())


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "audioanalytic/0.2",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def probe_llm_service(cfg: LlmConfig | None = None, *, timeout: float = 2.5) -> dict[str, Any]:
    """Проверяет доступность локальной Ollama / OpenAI-совместимого сервера и список моделей."""
    cfg = cfg or LlmConfig()
    base = cfg.base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base

    available: list[str] = []
    reachable = False
    provider = "openai-compatible"
    error_note = ""

    # 1. Проверка нативного эндпоинта Ollama (/api/tags)
    try:
        tags_resp = _http_json(f"{root}/api/tags", api_key=cfg.api_key, timeout=timeout)
        if isinstance(tags_resp, dict) and "models" in tags_resp:
            reachable = True
            provider = "ollama"
            available = [
                str(m.get("name") or m.get("model"))
                for m in (tags_resp.get("models") or [])
                if isinstance(m, dict) and (m.get("name") or m.get("model"))
            ]
    except Exception as exc:
        error_note = str(exc)

    # 2. Фоллбэк на стандартный OpenAI /v1/models
    if not reachable:
        try:
            models_resp = _http_json(f"{base}/models", api_key=cfg.api_key, timeout=timeout)
            if isinstance(models_resp, dict) and isinstance(models_resp.get("data"), list):
                reachable = True
                available = [
                    str(m.get("id"))
                    for m in models_resp["data"]
                    if isinstance(m, dict) and m.get("id")
                ]
        except Exception as exc:
            error_note = error_note or str(exc)

    selected = cfg.model
    if reachable and available:
        exact = [m for m in available if m == cfg.model]
        prefix = [m for m in available if m.split(":")[0] == cfg.model.split(":")[0]]
        if exact:
            selected = exact[0]
        elif prefix:
            selected = prefix[0]
        else:
            selected = available[0]

    return {
        "reachable": reachable,
        "provider": provider,
        "base_url": cfg.base_url,
        "localhost_only": cfg.is_localhost,
        "configured_model": cfg.model,
        "selected_model": selected,
        "available_models": available,
        "note": (
            f"Доступна модель {selected}"
            if reachable
            else f"Сервис LLM на {cfg.base_url} недоступен ({error_note or 'нет ответа'}); активен резервный локальный режим"
        ),
    }


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Извлекает первый валидный JSON-объект даже если модель обернула его в ```json ... ```."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : idx + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            break
        start = cleaned.find("{", start + 1)
    raise ValueError("Ответ LLM не содержит корректного JSON-объекта")


def call_llm_json(
    messages: list[dict[str, str]],
    cfg: LlmConfig,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Вызывает /chat/completions и возвращает распарсенный JSON-словарь."""
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    target_model = model_name or cfg.model
    body: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "temperature": cfg.temperature,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = _http_json(
            url, method="POST", payload=body, api_key=cfg.api_key, timeout=cfg.timeout_sec
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            body.pop("response_format", None)
            resp = _http_json(
                url, method="POST", payload=body, api_key=cfg.api_key, timeout=cfg.timeout_sec
            )
        else:
            raise

    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError("Пустой список choices в ответе LLM")
    content = str((choices[0].get("message") or {}).get("content") or "")
    return _extract_json_object(content)


def parse_timecode_sec(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    s = str(raw).strip()
    m = _TIMECODE_RE.search(s)
    if m:
        hh = int(m.group(1) or 0)
        mm = int(m.group(2) or 0)
        ss = int(m.group(3) or 0)
        return float(hh * 3600 + mm * 60 + ss)
    try:
        return max(0.0, float(s))
    except ValueError:
        return None


def _stems(text: str) -> set[str]:
    out: set[str] = set()
    for w in _WORD_RE.findall(text.lower().replace("ё", "е")):
        if len(w) >= 4:
            out.add(w[:6])
    return out


def ground_and_verify_thesis(
    thesis_text: str,
    raw_tc: Any,
    turns: list[dict[str, Any]],
) -> tuple[float, float, bool]:
    """Привязывает тезис к реальной реплике и вычисляет достоверность ``confidence`` (FR-06).

    Возвращает ``(timecode_sec, confidence, needs_review)``.
    """
    if not turns:
        sec = parse_timecode_sec(raw_tc) or 0.0
        return sec, 0.5, True

    t_stems = _stems(thesis_text)
    parsed_sec = parse_timecode_sec(raw_tc)

    # Находим реплику с лучшим лексическим пересечением
    best_turn = turns[0]
    best_overlap = -1.0
    for u in turns:
        u_stems = _stems(str(u.get("text", "")))
        overlap = len(t_stems & u_stems) / max(1, len(t_stems)) if t_stems else 0.0
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = u

    max_end = max(float(u.get("end_sec") or u.get("start_sec") or 0.0) for u in turns)
    if parsed_sec is None or parsed_sec > max_end + 5.0:
        anchored_sec = float(best_turn.get("start_sec") or 0.0)
    else:
        # Привязываем к ближайшей по времени реплике (если она в пределах ±12 с), иначе оставляем таймкод
        nearest = min(turns, key=lambda u: abs(float(u.get("start_sec") or 0.0) - parsed_sec))
        if abs(float(nearest.get("start_sec") or 0.0) - parsed_sec) <= 12.0:
            anchored_sec = float(nearest.get("start_sec") or 0.0)
        elif best_overlap >= 0.45:
            anchored_sec = float(best_turn.get("start_sec") or 0.0)
        else:
            anchored_sec = parsed_sec

    # Проверяем лексическую опору тезиса в окне ±45 с вокруг таймкода и по всей лекции
    if not t_stems:
        return round(anchored_sec, 2), 0.6, False

    window_stems: set[str] = set()
    global_stems: set[str] = set()
    for u in turns:
        us = _stems(str(u.get("text", "")))
        global_stems |= us
        u_start = float(u.get("start_sec") or 0.0)
        u_end = float(u.get("end_sec") or u_start)
        if (u_end >= anchored_sec - 45.0) and (u_start <= anchored_sec + 45.0):
            window_stems |= us

    local_cov = len(t_stems & window_stems) / len(t_stems)
    global_cov = len(t_stems & global_stems) / len(t_stems)
    raw_conf = max(local_cov, global_cov * 0.85)
    confidence = round(min(0.99, 0.35 + 0.65 * raw_conf), 2)
    needs_review = confidence < 0.58
    return round(anchored_sec, 2), confidence, needs_review


def extract_academic_entities(
    turns: list[dict[str, Any]],
    known_terms: list[str] | None = None,
    llm_entities: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Извлекает именованные сущности (термины, персоналии, даты/сроки) из текста и ответа LLM."""
    full_text = " ".join(str(u.get("text", "")) for u in turns)
    low_text = full_text.lower().replace("ё", "е")

    terms_found: list[str] = []
    for term in (known_terms or []):
        t_clean = term.strip()
        if not t_clean:
            continue
        probe = t_clean.lower().replace("ё", "е")
        stem = probe[: max(4, len(probe) - 2)]
        if stem in low_text and t_clean not in terms_found:
            terms_found.append(t_clean)

    persons: list[str] = []
    for m in _INITIALS_PERSON_RE.findall(full_text):
        val = re.sub(r"\s+", " ", m.strip())
        if val not in persons:
            persons.append(val)
    for m in _NAMED_LAW_RE.findall(full_text):
        val = m.strip()
        if val and val not in persons:
            persons.append(val)

    dates: list[str] = []
    for m in _YEAR_DATE_RE.findall(full_text):
        val = m.strip()
        if val and val not in dates:
            dates.append(val)

    if isinstance(llm_entities, dict):
        for p in (llm_entities.get("persons") or []):
            s = str(p).strip()
            if s and s not in persons:
                persons.append(s)
        for d in (llm_entities.get("dates") or []):
            s = str(d).strip()
            if s and s not in dates:
                dates.append(s)
        for t in (llm_entities.get("terms") or []):
            s = str(t).strip()
            if s and s.lower() not in {x.lower() for x in terms_found}:
                terms_found.append(s)

    return {
        "terms": terms_found[:20],
        "persons": persons[:15],
        "dates": dates[:15],
    }


def _turns_to_prompt_chunks(turns: list[dict[str, Any]], max_chars: int = 12_000) -> list[str]:
    """Форматирует реплики в строки ``[ЧЧ:ММ:СС] Спикер: текст`` и режет без разрыва реплик."""
    lines = [
        f"[{fmt_ts(float(u.get('start_sec') or 0.0))}] {u.get('speaker', 'Спикер 1')}: {u.get('text', '').strip()}"
        for u in turns
        if str(u.get("text", "")).strip()
    ]
    if not lines:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    cur_len = 0
    for ln in lines:
        add = len(ln) + 1
        if buf and cur_len + add > max_chars:
            chunks.append("\n".join(buf))
            buf, cur_len = [], 0
        buf.append(ln)
        cur_len += add
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _normalize_llm_payload(
    raw: dict[str, Any],
    turns: list[dict[str, Any]],
    model_name: str,
    known_terms: list[str] | None = None,
) -> SummaryData:
    goal = str(raw.get("goal") or "").strip() or "Разбор основных тем и положений аудиозаписи."

    concepts: list[dict[str, str]] = []
    for item in (raw.get("key_concepts") or []):
        if isinstance(item, dict):
            term = str(item.get("term") or item.get("name") or "").strip()
            defn = str(item.get("definition") or item.get("description") or "").strip()
            if term:
                concepts.append({"term": term, "definition": defn or "Ключевое понятие занятия"})
        elif isinstance(item, str) and item.strip():
            part_t, _, part_d = item.partition(":")
            concepts.append({
                "term": part_t.strip(" -•"),
                "definition": part_d.strip() or "Ключевое понятие занятия",
            })

    theses: list[ThesisItemData] = []
    raw_theses = raw.get("theses") or raw.get("key_theses") or []
    for item in raw_theses:
        if isinstance(item, dict):
            txt = str(item.get("text") or item.get("thesis") or "").strip()
            tc = item.get("timecode") if item.get("timecode") is not None else item.get("timecode_sec")
        else:
            s_raw = str(item).strip()
            m = _TIMECODE_RE.search(s_raw)
            tc = m.group(0) if m else None
            txt = _TIMECODE_RE.sub("", s_raw, count=1).strip(" —-:")
        if not txt:
            continue
        sec, conf, review = ground_and_verify_thesis(txt, tc, turns)
        theses.append(
            ThesisItemData(
                seq_num=len(theses) + 1,
                text=txt,
                timecode_sec=sec,
                timecode_formatted=f"[{fmt_ts(sec)}]",
                confidence=conf,
                needs_review=review,
            )
        )

    if not theses and turns:
        # Гарантируем хотя бы базовые тезисы по репликам, если модель вернула пустой массив
        fallback = build_extractive_summary(turns, known_terms=known_terms)
        theses = fallback.theses

    tasks = [
        str(x).strip(" -•")
        for x in (raw.get("tasks_decisions") or [])
        if str(x).strip(" -•") and str(x).strip().lower() not in ("нет", "не выявлено", "отсутствуют")
    ]
    questions = [
        str(x).strip(" -•")
        for x in (raw.get("self_check_questions") or [])
        if str(x).strip(" -•")
    ]

    entities = extract_academic_entities(
        turns,
        known_terms=known_terms,
        llm_entities=raw.get("entities") if isinstance(raw.get("entities"), dict) else None,
    )

    return SummaryData(
        model_name=model_name,
        goal=goal,
        key_concepts=concepts[:12],
        theses=theses[:12],
        tasks_decisions=tasks[:10],
        self_check_questions=questions[:8],
        entities=entities,
        used_fallback=False,
    )


def build_extractive_summary(
    turns: list[dict[str, Any]],
    *,
    title: str = "",
    known_terms: list[str] | None = None,
) -> SummaryData:
    """Автономный алгоритмический конспект (используется как мгновенный резерв без запущенной LLM)."""
    entities = extract_academic_entities(turns, known_terms=known_terms)
    matched_terms = entities.get("terms", [])

    if not turns:
        return SummaryData(
            model_name="extractive-local (резервный режим)",
            goal="В аудиозаписи не обнаружено распознанных реплик.",
            entities=entities,
            used_fallback=True,
        )

    first_text = str(turns[0].get("text", "")).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", first_text)[0].strip()
    if title and title.lower() not in first_sentence.lower():
        goal = f"Тема занятия «{title}»: {first_sentence}"
    else:
        goal = first_sentence or f"Разбор материала записи «{title or 'Аудиозапись'}»."

    # Выбираем наиболее содержательные реплики равномерно по ходу записи
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, u in enumerate(turns):
        txt = str(u.get("text", "")).strip()
        words = txt.split()
        if len(words) < 4:
            continue
        term_bonus = sum(3.0 for t in matched_terms if t.lower()[:5] in txt.lower())
        score = min(len(words), 45) + term_bonus
        scored.append((score, idx, u))

    if not scored:
        scored = [(1.0, idx, u) for idx, u in enumerate(turns[:5])]

    top_indices = sorted(idx for _, idx, _ in sorted(scored, reverse=True)[: min(6, len(scored))])
    theses: list[ThesisItemData] = []
    for idx in top_indices:
        u = turns[idx]
        sec = round(float(u.get("start_sec") or 0.0), 2)
        raw_txt = str(u.get("text", "")).strip()
        # Берём первые 1–2 предложения реплики для лаконичности тезиса
        sents = re.split(r"(?<=[.!?])\s+", raw_txt)
        concise = " ".join(sents[:2]).strip()
        if len(concise) > 240:
            concise = concise[:237].rstrip() + "…"
        theses.append(
            ThesisItemData(
                seq_num=len(theses) + 1,
                text=concise,
                timecode_sec=sec,
                timecode_formatted=f"[{fmt_ts(sec)}]",
                confidence=0.95,
                needs_review=False,
            )
        )

    concepts: list[dict[str, str]] = []
    for term in matched_terms[:8]:
        context_sent = ""
        for u in turns:
            for sent in re.split(r"(?<=[.!?])\s+", str(u.get("text", ""))):
                if term.lower()[:5] in sent.lower():
                    context_sent = sent.strip()
                    break
            if context_sent:
                break
        concepts.append({
            "term": term,
            "definition": context_sent[:200] if context_sent else "Ключевой термин, рассмотренный в записи.",
        })

    tasks: list[str] = []
    for u in turns:
        for sent in re.split(r"(?<=[.!?])\s+", str(u.get("text", ""))):
            low_s = sent.lower()
            if any(h in low_s for h in _TASK_HINTS):
                clean_s = sent.strip()
                if clean_s and clean_s not in tasks:
                    tasks.append(clean_s)

    questions: list[str] = []
    for c in concepts[:3]:
        questions.append(f"Как определяется и где применяется понятие «{c['term']}»?")
    if len(questions) < 3:
        for th in theses[: 3 - len(questions)]:
            questions.append(f"Поясните основной тезис фрагмента {th.timecode_formatted}: «{th.text[:90]}».")

    return SummaryData(
        model_name="extractive-local (резервный режим)",
        goal=goal,
        key_concepts=concepts,
        theses=theses,
        tasks_decisions=tasks[:6],
        self_check_questions=questions[:5],
        entities=entities,
        used_fallback=True,
    )


def generate_summary_for_turns(
    turns: list[dict[str, Any]],
    *,
    title: str = "",
    cfg: LlmConfig | None = None,
    known_terms: list[str] | None = None,
    require_llm: bool = False,
) -> SummaryData:
    """Генерирует структурированный 5-раздельный конспект через локальную LLM (или резервный контур)."""
    cfg = cfg or LlmConfig()
    if not turns:
        return build_extractive_summary([], title=title, known_terms=known_terms)

    probe = probe_llm_service(cfg)
    if not probe["reachable"]:
        if require_llm:
            raise RuntimeError(probe["note"])
        log.info("LLM недоступна (%s) — формирую резервный алгоритмический конспект", probe["note"])
        return build_extractive_summary(turns, title=title, known_terms=known_terms)

    active_model = str(probe["selected_model"] or cfg.model)
    chunks = _turns_to_prompt_chunks(turns, max_chars=cfg.max_chunk_chars)

    try:
        if len(chunks) == 1:
            user_prompt = (
                f"Название записи: «{title or 'Академическая запись'}».\n\n"
                f"Расшифровка с таймкодами:\n{chunks[0]}"
            )
            raw_json = call_llm_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT_RU},
                    {"role": "user", "content": user_prompt},
                ],
                cfg,
                model_name=active_model,
            )
        else:
            # Map-Reduce для длинных лекций (> max_chunk_chars)
            partial_notes: list[str] = []
            for idx, ch in enumerate(chunks, start=1):
                part_json = call_llm_json(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT_RU},
                        {
                            "role": "user",
                            "content": (
                                f"Название записи: «{title}» (фрагмент {idx} из {len(chunks)}).\n"
                                f"Выдели ключевые понятия, тезисы с точными таймкодами [ЧЧ:ММ:СС] и задачи:\n\n{ch}"
                            ),
                        },
                    ],
                    cfg,
                    model_name=active_model,
                )
                partial_notes.append(json.dumps(part_json, ensure_ascii=False))

            synth_prompt = (
                f"Название лекции: «{title or 'Академическая лекция'}».\n"
                "Ниже приведены структурированные выжимки последовательных частей одной лекции. "
                "Объедини их в единый итоговый конспект по всем 5 разделам без дублирования, "
                "сохранив точные таймкоды [ЧЧ:ММ:СС] у тезисов:\n\n"
                + "\n\n".join(partial_notes)
            )
            raw_json = call_llm_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT_RU},
                    {"role": "user", "content": synth_prompt},
                ],
                cfg,
                model_name=active_model,
            )

        return _normalize_llm_payload(raw_json, turns, model_name=active_model, known_terms=known_terms)
    except Exception as exc:
        if require_llm:
            raise
        log.warning("Ошибка генерации конспекта через LLM (%s) — использую резервный режим", exc)
        return build_extractive_summary(turns, title=title, known_terms=known_terms)
