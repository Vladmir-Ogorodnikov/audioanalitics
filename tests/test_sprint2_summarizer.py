"""Интеграционные и модульные тесты Спринта 2: Ollama (qwen2.5:3b), верификация таймкодов, NER и экспорт DOCX/MD/SRT."""
from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from audioanalytic.cli import main as cli_main
from audioanalytic.config import LlmConfig, ServerSettings
from audioanalytic.models import Segment, Transcript
from audioanalytic.server.app import create_app
from audioanalytic.summarizer import (
    generate_summary_for_turns,
    ground_and_verify_thesis,
    probe_llm_service,
)


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/tags":
            payload = {"models": [{"name": "qwen2.5:3b", "size": 1950000000}]}
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self.__class__.calls.append(body)

            summary_obj = {
                "goal": "Изучение метода наименьших квадратов и условий теоремы Гаусса — Маркова.",
                "key_concepts": [
                    {
                        "term": "Метод наименьших квадратов",
                        "definition": "Способ оценки параметров регрессии путём минимизации суммы квадратов остатков.",
                    },
                    {
                        "term": "Гомоскедастичность",
                        "definition": "Постоянство дисперсии случайных ошибок наблюдения.",
                    },
                ],
                "theses": [
                    {
                        "timecode": "[00:00:06]",
                        "text": "При выполнении гомоскедастичности оценки метода наименьших квадратов несмещённые и эффективные.",
                    },
                    {
                        "timecode": "[00:00:14]",
                        "text": "Квантовая хромодинамика сверхпроводящих тахионов в галактических туманностях.",
                    },
                ],
                "tasks_decisions": [
                    "К следующему семинару 15 ноября решить задачи 1–4 по линейной регрессии."
                ],
                "self_check_questions": [
                    "В чём состоит условие гомоскедастичности ошибок?",
                    "К чему приводит гетероскедастичность при оценке стандартных ошибок?",
                ],
                "entities": {
                    "persons": ["К. Ф. Гаусс", "А. А. Марков"],
                    "dates": ["15 ноября"],
                    "terms": ["метод наименьших квадратов", "гомоскедастичность"],
                },
            }
            resp = {
                "id": "chatcmpl-ollama-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "```json\n" + json.dumps(summary_obj, ensure_ascii=False) + "\n```",
                        },
                    }
                ],
            }
            raw = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(404)
        self.end_headers()


def _start_fake_ollama() -> tuple[HTTPServer, str]:
    _FakeOllamaHandler.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}/v1"


def test_ollama_probe_summarization_grounding_and_map_reduce():
    srv, base_url = _start_fake_ollama()
    try:
        cfg = LlmConfig(base_url=base_url, model="qwen2.5:3b", max_chunk_chars=180)
        info = probe_llm_service(cfg)
        assert info["reachable"] is True
        assert info["provider"] == "ollama"
        assert info["selected_model"] == "qwen2.5:3b"
        assert info["localhost_only"] is True

        turns = [
            {
                "start_sec": 0.0,
                "end_sec": 5.5,
                "speaker": "Спикер 1",
                "text": "Сегодня мы разбираем метод наименьших квадратов и теорему Гаусса — Маркова 1933 года.",
            },
            {
                "start_sec": 6.0,
                "end_sec": 13.0,
                "speaker": "Спикер 1",
                "text": "При выполнении условия гомоскедастичности оценки метода наименьших квадратов эффективные.",
            },
            {
                "start_sec": 14.0,
                "end_sec": 21.0,
                "speaker": "Спикер 2",
                "text": "К следующему семинару 15 ноября нужно решить домашнее задание по линейной регрессии.",
            },
        ]

        # За счёт max_chunk_chars=180 проверяется сразу и Map-Reduce склейка нескольких чанков!
        summary = generate_summary_for_turns(
            turns,
            title="Эконометрика: МНК",
            cfg=cfg,
            known_terms=["метод наименьших квадратов", "гомоскедастичность"],
        )
        assert len(_FakeOllamaHandler.calls) >= 2  # map по чанкам + финальный синтез
        assert summary.model_name == "qwen2.5:3b"
        assert summary.used_fallback is False
        assert "наименьших квадратов" in summary.goal.lower()
        assert len(summary.key_concepts) == 2
        assert len(summary.theses) == 2

        # Первый тезис опирается на реальную реплику -> высокая уверенность, сверка не нужна
        th_grounded = summary.theses[0]
        assert abs(th_grounded.timecode_sec - 6.0) < 0.5
        assert th_grounded.needs_review is False
        assert th_grounded.confidence >= 0.70

        # Второй тезис («тахионы в туманностях») отсутствует в лекции -> автоматически помечен needs_review=True (FR-06)
        th_hallucinated = summary.theses[1]
        assert th_hallucinated.needs_review is True
        assert th_hallucinated.confidence < 0.58

        # Проверим NER
        assert "К. Ф. Гаусс" in summary.entities["persons"]
        assert "15 ноября" in summary.entities["dates"]
    finally:
        srv.shutdown()


def test_offline_fallback_when_ollama_not_running():
    closed_cfg = LlmConfig(base_url="http://127.0.0.1:59999/v1", model="qwen2.5:3b", timeout_sec=0.5)
    turns = [
        {
            "start_sec": 0.0,
            "end_sec": 8.0,
            "speaker": "Спикер 1",
            "text": "Тема занятия — сингулярное разложение матриц и метод главных компонент.",
        },
        {
            "start_sec": 8.5,
            "end_sec": 16.0,
            "speaker": "Спикер 1",
            "text": "Домашнее задание к 20 ноября: подготовить расчёт собственных значений.",
        },
    ]
    summary = generate_summary_for_turns(
        turns,
        title="Линейная алгебра",
        cfg=closed_cfg,
        known_terms=["сингулярное разложение", "собственные значения"],
    )
    assert summary.used_fallback is True
    assert "extractive-local" in summary.model_name
    assert len(summary.theses) >= 1
    assert len(summary.tasks_decisions) >= 1
    assert "сингулярное разложение" in summary.entities["terms"]


def test_server_summary_api_search_and_docx_md_srt_exports(tmp_path: Path):
    srv, base_url = _start_fake_ollama()
    try:
        settings = ServerSettings(
            host="127.0.0.1",
            port=8098,
            data_dir=tmp_path,
            llm=LlmConfig(base_url=base_url, model="qwen2.5:3b"),
        )
        app = create_app(settings=settings, start_worker=True)
        with TestClient(app) as client:
            st = client.get("/api/v1/system/status").json()
            assert st["llm"]["reachable"] is True
            assert st["llm"]["selected_model"] == "qwen2.5:3b"

            demo = client.post("/api/v1/recordings/demo?priority=3")
            assert demo.status_code == 200
            rec_id = demo.json()["id"]

            deadline = time.monotonic() + 25.0
            detail = None
            while time.monotonic() < deadline:
                resp = client.get(f"/api/v1/recordings/{rec_id}")
                body = resp.json()
                if body["status"] == "completed":
                    detail = body
                    break
                time.sleep(0.25)

            assert detail is not None, "Обработка демо-записи с конспектом не завершилась вовремя"
            assert detail["summary"] is not None
            assert detail["summary"]["model_name"] == "qwen2.5:3b"
            assert len(detail["summary"]["theses"]) == 2

            # Поиск находит запись даже по слову из сгенерированного конспекта (FR-08)
            found = client.get("/api/v1/recordings", params={"q": "тахионов"}).json()
            assert len(found) == 1
            assert found[0]["id"] == rec_id
            assert found[0]["matched_timecode_sec"] is not None

            # Ручное пересоздание конспекта по кнопке
            regen = client.post(f"/api/v1/recordings/{rec_id}/summarize", json={})
            assert regen.status_code == 200
            assert regen.json()["model_name"] == "qwen2.5:3b"

            # Экспорт Markdown
            md_res = client.get(f"/api/v1/recordings/{rec_id}/export?format=md")
            assert md_res.status_code == 200
            assert "## 1. Цель и тема занятия" in md_res.text
            assert "Метод наименьших квадратов" in md_res.text

            # Экспорт субтитров SRT
            srt_res = client.get(f"/api/v1/recordings/{rec_id}/export?format=srt")
            assert srt_res.status_code == 200
            assert "-->" in srt_res.text

            # Экспорт Microsoft Word (.docx)
            docx_res = client.get(f"/api/v1/recordings/{rec_id}/export?format=docx")
            assert docx_res.status_code == 200
            with zipfile.ZipFile(io.BytesIO(docx_res.content), "r") as zf:
                names = set(zf.namelist())
                assert "word/document.xml" in names
                xml_body = zf.read("word/document.xml").decode("utf-8")
                assert "Основные тезисы" in xml_body
                assert "Метод наименьших квадратов" in xml_body
    finally:
        srv.shutdown()


def test_cli_summarize_command(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:59998/v1")
    tr = Transcript(
        language="ru",
        segments=[
            Segment(0.0, 5.0, "Разбираем градиентный спуск и регуляризацию моделей.", "Спикер 1"),
            Segment(5.5, 11.0, "К следующему занятию подготовить отчёт по лабораторной работе.", "Спикер 1"),
        ],
    )
    tr_file = tmp_path / "lec1" / "transcript.json"
    tr_file.parent.mkdir(parents=True, exist_ok=True)
    tr_file.write_text(json.dumps(tr.to_dict(), ensure_ascii=False), encoding="utf-8")

    rc = cli_main(["summarize", str(tr_file), "--title", "Машинное обучение"])
    assert rc == 0
    assert (tr_file.parent / "summary.md").is_file()
    assert (tr_file.parent / "summary.json").is_file()
    md = (tr_file.parent / "summary.md").read_text(encoding="utf-8")
    assert "# Машинное обучение" in md
    assert "градиентный спуск" in md.lower()


def test_ground_and_verify_thesis_snaps_out_of_range_timestamp():
    turns = [
        {"start_sec": 12.0, "end_sec": 20.0, "text": "Обратное распространение ошибки вычисляет градиент функции потерь."}
    ]
    sec, conf, review = ground_and_verify_thesis(
        "Обратное распространение ошибки вычисляет градиент.",
        "[05:00:00]",
        turns,
    )
    assert sec == 12.0
    assert conf >= 0.70
    assert review is False
