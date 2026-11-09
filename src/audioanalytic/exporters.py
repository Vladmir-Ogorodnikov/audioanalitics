"""Экспорт конспектов и расшифровок в академические форматы TXT, MD, DOCX, SRT, JSON."""
from __future__ import annotations

import io
import zipfile
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from .formatting import fmt_ts


def _srt_timestamp(sec: float) -> str:
    ms_total = max(0, int(round(float(sec) * 1000)))
    hh, rem = divmod(ms_total, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1_000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def build_srt(utterances: list[dict[str, Any]]) -> str:
    """Формирует субтитры в формате SubRip (.srt) с указанием спикеров."""
    blocks: list[str] = []
    for idx, u in enumerate(utterances, start=1):
        start_s = float(u.get("start_sec") or 0.0)
        end_s = max(start_s + 0.5, float(u.get("end_sec") or (start_s + 2.0)))
        spk = str(u.get("speaker") or "").strip()
        txt = str(u.get("text") or "").strip()
        line = f"{spk}: {txt}" if spk else txt
        blocks.append(f"{idx}\n{_srt_timestamp(start_s)} --> {_srt_timestamp(end_s)}\n{line}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def build_markdown(
    *,
    title: str,
    duration_sec: float,
    word_count: int,
    utterances: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> str:
    """Формирует структурированный Markdown-документ (конспект + полная расшифровка)."""
    lines: list[str] = [
        f"# {title}",
        "",
        f"**Длительность:** `{fmt_ts(duration_sec)}` · **Слов:** {word_count}",
        "",
    ]

    if summary:
        lines.extend([
            "## 1. Цель и тема занятия",
            "",
            str(summary.get("goal") or "не выявлено"),
            "",
        ])

        theses = summary.get("theses") or []
        if theses:
            lines.extend(["## 2. Основные тезисы", ""])
            for th in theses:
                tc = th.get("timecode_formatted") or f"[{fmt_ts(float(th.get('timecode_sec') or 0))}]"
                rev = " *(требует сверки)*" if th.get("needs_review") else ""
                lines.append(f"- **`{tc}`** — {th.get('text', '')}{rev}")
            lines.append("")

        concepts = summary.get("key_concepts") or []
        if concepts:
            lines.extend(["## 3. Ключевые понятия и определения", ""])
            for c in concepts:
                lines.append(f"- **{c.get('term', '')}** — {c.get('definition', '')}")
            lines.append("")

        tasks = summary.get("tasks_decisions") or []
        lines.extend(["## 4. Задачи и организационные решения", ""])
        if tasks:
            for t in tasks:
                lines.append(f"- {t}")
        else:
            lines.append("_Организационных поручений не зафиксировано._")
        lines.append("")

        questions = summary.get("self_check_questions") or []
        if questions:
            lines.extend(["## 5. Вопросы для самопроверки", ""])
            for idx, q in enumerate(questions, start=1):
                lines.append(f"{idx}. {q}")
            lines.append("")

        entities = summary.get("entities") or {}
        ent_parts: list[str] = []
        if entities.get("terms"):
            ent_parts.append("**Термины:** " + ", ".join(entities["terms"]))
        if entities.get("persons"):
            ent_parts.append("**Персоналии:** " + ", ".join(entities["persons"]))
        if entities.get("dates"):
            ent_parts.append("**Даты:** " + ", ".join(entities["dates"]))
        if ent_parts:
            lines.extend(["### Упомянутые сущности", ""] + [f"- {p}" for p in ent_parts] + [""])

        lines.extend(["---", ""])

    lines.extend(["## Полная расшифровка", ""])
    for u in utterances:
        tc = fmt_ts(float(u.get("start_sec") or 0.0))
        lines.append(f"**[{tc}] {u.get('speaker', 'Спикер 1')}:** {u.get('text', '')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_txt(
    *,
    title: str,
    duration_sec: float,
    word_count: int,
    utterances: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> str:
    """Формирует текстовый файл с конспектом (если есть) и полной расшифровкой."""
    lines: list[str] = [
        f"АС «АудиоАналитик ИИ» — Расшифровка записи: {title}",
        f"Длительность: {fmt_ts(duration_sec)} | Слов: {word_count}",
        "=" * 72,
        "",
    ]
    if summary:
        lines.append("КОНСПЕКТ ЗАНЯТИЯ")
        lines.append("-" * 72)
        lines.append(f"Цель занятия: {summary.get('goal') or 'не выявлено'}")
        lines.append("")
        if summary.get("theses"):
            lines.append("Основные тезисы:")
            for th in summary["theses"]:
                tc = th.get("timecode_formatted") or f"[{fmt_ts(float(th.get('timecode_sec') or 0))}]"
                rev = " [требует сверки]" if th.get("needs_review") else ""
                lines.append(f"  • {tc} {th.get('text', '')}{rev}")
            lines.append("")
        if summary.get("key_concepts"):
            lines.append("Ключевые понятия:")
            for c in summary["key_concepts"]:
                lines.append(f"  • {c.get('term', '')} — {c.get('definition', '')}")
            lines.append("")
        if summary.get("tasks_decisions"):
            lines.append("Задачи и решения:")
            for t in summary["tasks_decisions"]:
                lines.append(f"  • {t}")
            lines.append("")
        if summary.get("self_check_questions"):
            lines.append("Вопросы для самопроверки:")
            for idx, q in enumerate(summary["self_check_questions"], start=1):
                lines.append(f"  {idx}. {q}")
            lines.append("")
        lines.append("=" * 72)
        lines.append("")

    for u in utterances:
        lines.append(f"[{fmt_ts(float(u.get('start_sec') or 0.0))}] {u.get('speaker', 'Спикер 1')}: {u.get('text', '')}")
    return "\n".join(lines) + "\n"


def _w_p(runs: list[tuple[str, bool]], *, style: str = "", spacing_after: int = 120) -> str:
    """Собирает валидный параграф WordprocessingML ``<w:p>``."""
    style_tag = f'<w:pStyle w:val="{style}"/>' if style else ""
    ppr = f'<w:pPr>{style_tag}<w:spacing w:after="{spacing_after}"/></w:pPr>'
    r_xml: list[str] = []
    for text_val, is_bold in runs:
        if not text_val:
            continue
        safe = xml_escape(str(text_val))
        rpr = "<w:rPr><w:b/></w:rPr>" if is_bold else ""
        r_xml.append(f'<w:r>{rpr}<w:t xml:space="preserve">{safe}</w:t></w:r>')
    return f"<w:p>{ppr}{''.join(r_xml)}</w:p>"


def build_docx_bytes(
    *,
    title: str,
    duration_sec: float,
    word_count: int,
    utterances: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
) -> bytes:
    """Генерирует валидный документ Microsoft Word (.docx) без внешних C-зависимостей."""
    paragraphs: list[str] = [
        _w_p([(title, True)], style="Title", spacing_after=160),
        _w_p(
            [(f"Длительность: {fmt_ts(duration_sec)}  |  Слов: {word_count}", False)],
            spacing_after=240,
        ),
    ]

    if summary:
        paragraphs.append(_w_p([("1. Цель и тема занятия", True)], style="Heading1", spacing_after=100))
        paragraphs.append(_w_p([(str(summary.get("goal") or "не выявлено"), False)], spacing_after=180))

        theses = summary.get("theses") or []
        if theses:
            paragraphs.append(_w_p([("2. Основные тезисы", True)], style="Heading1", spacing_after=100))
            for th in theses:
                tc = th.get("timecode_formatted") or f"[{fmt_ts(float(th.get('timecode_sec') or 0))}]"
                suffix = " (требует сверки)" if th.get("needs_review") else ""
                paragraphs.append(
                    _w_p([(f"• {tc} ", True), (f"{th.get('text', '')}{suffix}", False)], spacing_after=90)
                )

        concepts = summary.get("key_concepts") or []
        if concepts:
            paragraphs.append(_w_p([("3. Ключевые понятия", True)], style="Heading1", spacing_after=100))
            for c in concepts:
                paragraphs.append(
                    _w_p(
                        [(f"• {c.get('term', '')}: ", True), (str(c.get("definition") or ""), False)],
                        spacing_after=90,
                    )
                )

        tasks = summary.get("tasks_decisions") or []
        paragraphs.append(_w_p([("4. Задачи и организационные решения", True)], style="Heading1", spacing_after=100))
        if tasks:
            for t in tasks:
                paragraphs.append(_w_p([(f"• {t}", False)], spacing_after=80))
        else:
            paragraphs.append(_w_p([("Организационных поручений не зафиксировано.", False)], spacing_after=140))

        questions = summary.get("self_check_questions") or []
        if questions:
            paragraphs.append(_w_p([("5. Вопросы для самопроверки", True)], style="Heading1", spacing_after=100))
            for idx, q in enumerate(questions, start=1):
                paragraphs.append(_w_p([(f"{idx}. {q}", False)], spacing_after=80))

    paragraphs.append(_w_p([("Полная расшифровка", True)], style="Heading1", spacing_after=140))
    for u in utterances:
        tc = fmt_ts(float(u.get("start_sec") or 0.0))
        spk = str(u.get("speaker") or "Спикер 1")
        paragraphs.append(
            _w_p([(f"[{tc}] {spk}: ", True), (str(u.get("text") or ""), False)], spacing_after=100)
        )

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1417"/></w:sectPr>'
        "</w:body></w:document>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Segoe UI" w:hAnsi="Segoe UI" w:cs="Segoe UI"/>'
        '<w:sz w:val="22"/><w:lang w:val="ru-RU"/>'
        "</w:rPr></w:rPrDefault></w:docDefaults>"
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:pPr><w:spacing w:before="220" w:after="100"/></w:pPr>'
        '<w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
        "</w:styles>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        "</Types>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    doc_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()
