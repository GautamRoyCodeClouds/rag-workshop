"""API contracts for the ingestion wizard and its background jobs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace

import chromadb
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.jobs import JobRegistry, sse_format


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Use a disposable session directory and in-process Chroma for each test."""
    from app.session import SessionStore

    monkeypatch.setattr(main_module, "store", SessionStore(data_dir=tmp_path))
    chroma_client = chromadb.EphemeralClient()
    monkeypatch.setattr(main_module, "get_client", lambda *args, **kwargs: chroma_client)
    isolated_settings = replace(
        main_module.settings, data_dir=tmp_path, chroma_collection=f"test-{tmp_path.name}"
    )
    monkeypatch.setattr(main_module, "settings", isolated_settings)
    monkeypatch.setattr(main_module.vector_store, "settings", isolated_settings)
    yield


@pytest.fixture
def client():
    with TestClient(main_module.app) as test_client:
        yield test_client


class TestPage:
    def test_the_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_the_page_lists_all_five_strategies(self, client):
        body = client.get("/").text
        for label in (
            "Fixed size",
            "Recursive",
            "Structure aware",
            "Semantic",
            "Parent document",
        ):
            assert label in body

    def test_config_exposes_the_deck_defaults(self, client):
        data = client.get("/api/config").json()
        assert data["default_chunk_size"] == 700
        assert data["default_chunk_overlap"] == 100
        assert data["default_strategy"] == "recursive"
        assert len(data["strategies"]) == 5

    def test_config_reports_whether_a_local_document_exists(self, client):
        assert client.get("/api/config").json()["has_local_pdf"] is False


class TestUpload:
    def test_rejects_a_non_pdf(self, client):
        response = client.post(
            "/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower()

    def test_rejects_a_file_over_the_size_limit(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module, "settings", replace(main_module.settings, max_upload_mb=0)
        )
        response = client.post(
            "/api/upload",
            files={"file": ("big.pdf", b"%PDF-1.4" + b"x" * 2048, "application/pdf")},
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_accepts_a_real_pdf_and_unlocks_step_two(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("handbook.pdf", handle.read(), "application/pdf")},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["upload"]["page_count"] == 5
        assert body["unlocked_step"] == 2
        assert "boilerplate_lines_removed" in body["upload"]
        assert "rag_session=" in response.headers["set-cookie"]

    def test_reports_a_scanned_pdf_clearly(self, client, tmp_path):
        from reportlab.pdfgen import canvas

        path = tmp_path / "scanned.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.showPage()
        pdf.save()
        with open(path, "rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("scanned.pdf", handle.read(), "application/pdf")},
            )
        assert response.status_code == 400
        assert "text layer" in response.json()["detail"]

    def test_use_local_is_unavailable_when_unconfigured(self, client):
        assert client.post("/api/use-local").status_code == 404

    def test_use_local_issues_a_session_cookie(self, client, structured_pdf, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "settings",
            replace(main_module.settings, local_pdf_path=structured_pdf),
        )
        response = client.post("/api/use-local")
        assert response.status_code == 200
        assert "rag_session=" in response.headers["set-cookie"]

    def test_malformed_session_cookie_becomes_400_and_is_replaced(self, client):
        response = client.get("/", cookies={"rag_session": "../../bad"})
        assert response.status_code == 400
        assert "session" in response.json()["detail"].lower()
        assert "rag_session=" in response.headers["set-cookie"]
        replacement = response.headers["set-cookie"].split("rag_session=", 1)[1].split(";", 1)[0]
        assert replacement.isalnum()


class TestChunkAndEmbed:
    def upload(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            return client.post(
                "/api/upload",
                files={"file": ("handbook.pdf", handle.read(), "application/pdf")},
            ).json()

    def test_chunking_requires_an_upload_first(self, client):
        response = client.post("/api/chunk", json={"strategy": "recursive"})
        assert response.status_code == 409
        assert "upload" in response.json()["detail"].lower()

    def test_chunking_rejects_an_unknown_strategy(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post("/api/chunk", json={"strategy": "nonsense"})
        assert response.status_code == 400

    def test_chunking_returns_a_job_id(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post(
            "/api/chunk", json={"strategy": "recursive", "size": 200, "overlap": 20}
        )
        assert response.status_code == 202
        assert response.json()["job_id"]

    def test_status_reports_a_finished_chunk_job(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        job_id = client.post(
            "/api/chunk", json={"strategy": "recursive", "size": 200, "overlap": 20}
        ).json()["job_id"]
        status = None
        for _ in range(100):
            status = client.get(f"/api/status/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.02)
        assert status["status"] == "done", f"job did not finish: {status}"
        assert any(event["type"] == "chunk" for event in status["events"])
        assert any(event["type"] == "done" for event in status["events"])

    def test_embedding_requires_chunking_first(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post("/api/embed")
        assert response.status_code == 409

    def test_embedding_starts_an_async_job_after_chunking(self, client, structured_pdf, monkeypatch):
        from app.pipeline.chunkers import Chunk

        self.upload(client, structured_pdf)
        state = main_module.store.get_or_create(client.cookies.get("rag_session"))
        state.chunks = [Chunk(index=0, text="A chunk", start=0, strategy="recursive")]
        state.chunking = {"size": 200, "overlap": 20}
        main_module.store.save(state)

        monkeypatch.setattr(main_module, "build_embeddings", lambda: object())

        def fake_embed(_model, _texts, _batch_size, progress):
            progress(1, 1)
            return [[1.0] + [0.0] * 383]

        monkeypatch.setattr(main_module, "embed_batched", fake_embed)
        response = client.post("/api/embed")
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(100):
            status = client.get(f"/api/status/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.02)
        assert status["status"] == "done"
        assert {"type": "embedded", "done": 1, "total": 1} in status["events"]

    def test_status_for_an_unknown_job_is_404(self, client):
        assert client.get("/api/status/nope").status_code == 404


class TestCollectionAndReset:
    def test_the_collection_reads_empty(self, client):
        body = client.get("/api/collection").json()
        assert body["total"] == 0
        assert body["records"] == []

    def test_reset_returns_a_clean_session(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            client.post(
                "/api/upload", files={"file": ("h.pdf", handle.read(), "application/pdf")}
            )
        body = client.post("/api/reset", json={"drop_collection": True}).json()
        assert body["unlocked_step"] == 1
        assert body["upload"] is None

    def test_reset_reports_a_requested_collection_drop_failure(self, client, monkeypatch):
        def unavailable(_client):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(main_module.vector_store, "drop_collection", unavailable)
        response = client.post("/api/reset", json={"drop_collection": True})
        assert response.status_code == 503
        assert "chromadb" in response.json()["detail"].lower()


class TestSseFormatting:
    def test_formats_as_a_data_line_pair(self):
        assert sse_format({"type": "done"}) == 'data: {"type": "done"}\n\n'

    def test_payload_is_valid_json(self):
        raw = sse_format({"type": "chunk", "index": 3})
        assert json.loads(raw.removeprefix("data: ").strip())["index"] == 3


class TestJobRegistry:
    def test_publish_records_events_for_the_polling_fallback(self):
        registry = JobRegistry()
        job = registry.create()
        registry.publish(job, {"type": "chunk", "index": 0})
        assert job.events == [{"type": "chunk", "index": 0}]

    def test_finish_sets_the_terminal_status(self):
        registry = JobRegistry()
        job = registry.create()
        registry.finish(job, "error", "boom")
        assert job.status == "error"
        assert job.error == "boom"

    def test_unknown_job_lookup_returns_none(self):
        assert JobRegistry().get("absent") is None

    def test_worker_thread_publication_is_visible_to_the_async_consumer(self):
        async def consume() -> list[dict]:
            registry = JobRegistry(loop=asyncio.get_running_loop())
            job = registry.create()
            thread = threading.Thread(
                target=registry.publish, args=(job, {"type": "chunk", "index": 1}),
            )
            thread.start()
            event = await asyncio.wait_for(job.queue.get(), timeout=1)
            thread.join()
            assert job.events == [{"type": "chunk", "index": 1}]
            return [event]

        assert asyncio.run(consume()) == [{"type": "chunk", "index": 1}]

    def test_late_sse_subscriber_replays_each_event_exactly_once(self, client):
        job = main_module.registry.create()
        main_module.registry.publish(job, {"type": "stage", "message": "already here"})
        main_module.registry.finish(job, "done")
        response = client.get(f"/api/events/{job.job_id}")
        frames = [line for line in response.text.splitlines() if line.startswith("data: ")]
        payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames]
        assert payloads == [{"type": "stage", "message": "already here"}, {"type": "done"}]
