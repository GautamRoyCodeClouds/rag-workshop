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
    monkeypatch.setattr(main_module, "registry", JobRegistry())
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

    def test_the_page_has_the_five_numbered_pipeline_sections(self, client):
        body = client.get("/").text
        for step, heading in enumerate(
            (
                "Load a document",
                "Choose how to cut it",
                "The chunks",
                "Embed and store",
                "What ChromaDB is holding",
            ),
            start=1,
        ):
            assert f'id="step-{step}"' in body
            assert f"STEP {step}" in body
            assert heading in body

    def test_the_page_serves_its_local_stylesheet_and_script(self, client):
        body = client.get("/").text
        assert 'href="/static/app.css"' in body
        assert 'src="/static/app.js"' in body
        stylesheet = client.get("/static/app.css")
        script = client.get("/static/app.js")
        assert stylesheet.status_code == 200
        assert "--cyan: #38e0cf" in stylesheet.text
        assert script.status_code == 200
        assert "Progressive-unlock client" in script.text

    def test_the_page_embeds_server_rendered_session_state(self, client):
        body = client.get("/").text
        assert "window.__STATE__" in body
        assert '"unlocked_step": 1' in body

    def test_the_page_has_no_external_font_or_network_urls(self, client):
        body = client.get("/").text
        stylesheet = client.get("/static/app.css").text
        script = client.get("/static/app.js").text
        combined = "\n".join((body, stylesheet, script)).lower()
        assert "fonts.googleapis.com" not in combined
        assert "fonts.gstatic.com" not in combined
        assert "@import url(" not in combined
        assert "http://" not in combined
        assert "https://" not in combined

    def test_the_stylesheet_keeps_hidden_controls_out_of_the_layout(self, client):
        stylesheet = client.get("/static/app.css").text
        assert "[hidden] { display:none !important; }" in stylesheet

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
        # TestClient keeps the issued cookie; a later server-rendered page must
        # rehydrate the document rather than silently beginning a new session.
        assert client.get("/").status_code == 200
        assert main_module.store.get_or_create(client.cookies["rag_session"]).upload["filename"] == structured_pdf.split("/")[-1]

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
        assert status["session"]["unlocked_step"] == 4

    def test_embedding_requires_chunking_first(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post("/api/embed")
        assert response.status_code == 409

    def test_embedding_starts_an_async_job_after_chunking(self, client, structured_pdf, monkeypatch):
        from app.pipeline.chunkers import Chunk

        class DeterministicEmbeddings:
            def __init__(self) -> None:
                self.batches: list[list[str]] = []

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.batches.append(list(texts))
                return [[float(len(text))] + [0.0] * 383 for text in texts]

        self.upload(client, structured_pdf)
        state = main_module.store.get_or_create(client.cookies.get("rag_session"))
        state.chunks = [
            Chunk(index=0, text="First chunk", start=0, strategy="recursive"),
            Chunk(index=1, text="Second chunk", start=12, strategy="recursive"),
        ]
        state.chunking = {"size": 200, "overlap": 20}
        main_module.store.save(state)
        monkeypatch.setattr(main_module, "settings", replace(main_module.settings, embed_batch_size=1))

        embeddings = DeterministicEmbeddings()
        monkeypatch.setattr(main_module, "build_embeddings", lambda: embeddings)
        response = client.post("/api/embed")
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(100):
            status = client.get(f"/api/status/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.02)
        assert status["status"] == "done"
        progress = [event for event in status["events"] if event["type"] == "embedded"]
        assert [(event["done"], event["total"]) for event in progress] == [(1, 2), (2, 2)]
        assert embeddings.batches == [["First chunk"], ["Second chunk"]]
        assert status["session"]["unlocked_step"] == 5

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

    def test_event_id_is_rendered_as_an_sse_cursor(self):
        assert sse_format({"id": 7, "type": "chunk"}).startswith("id: 7\ndata: ")


class TestJobRegistry:
    def test_publish_records_events_for_the_polling_fallback(self):
        registry = JobRegistry()
        job = registry.create()
        registry.publish(job, {"type": "chunk", "index": 0})
        assert job.events == [{"id": 1, "type": "chunk", "index": 0}]

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
            assert job.events == [{"id": 1, "type": "chunk", "index": 1}]
            return [event]

        assert asyncio.run(consume()) == [{"id": 1, "type": "chunk", "index": 1}]

    def test_late_sse_subscriber_replays_each_event_exactly_once(self, client):
        job = main_module.registry.create()
        main_module.registry.publish(job, {"type": "stage", "message": "already here"})
        main_module.registry.finish(job, "done")
        response = client.get(f"/api/events/{job.job_id}")
        frames = [line for line in response.text.splitlines() if line.startswith("data: ")]
        payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames]
        assert [payload["type"] for payload in payloads] == ["stage", "done"]
        assert [payload["id"] for payload in payloads] == [1, 2]

    def test_live_reconnect_replays_only_events_after_the_cursor(self):
        async def reconnect() -> None:
            registry = JobRegistry(loop=asyncio.get_running_loop())
            job = registry.create()
            registry.publish(job, {"type": "stage"})
            history, queue, terminal = registry.subscribe(job, after=0)
            assert terminal is False
            assert [event["id"] for event in history] == [1]
            registry.publish(job, {"type": "chunk"})
            assert (await queue.get())["id"] == 2
            registry.unsubscribe(job, queue)
            registry.publish(job, {"type": "summary"})
            replay, _queue, terminal = registry.subscribe(job, after=2)
            assert terminal is False
            assert [event["id"] for event in replay] == [3]

        asyncio.run(reconnect())

    def test_polling_after_an_sse_cursor_has_no_duplicate_events(self, client):
        job = main_module.registry.create()
        main_module.registry.publish(job, {"type": "stage"})
        main_module.registry.publish(job, {"type": "chunk"})
        main_module.registry.finish(job, "done")
        sse = client.get(f"/api/events/{job.job_id}", headers={"Last-Event-ID": "1"})
        seen = [
            json.loads(line.removeprefix("data: "))["id"]
            for line in sse.text.splitlines()
            if line.startswith("data: ")
        ]
        polling = client.get(f"/api/status/{job.job_id}?after=3").json()
        assert seen == [2, 3]
        assert polling["events"] == []
        assert polling["cursor"] == 3

    def test_terminal_snapshot_contains_the_terminal_event(self):
        registry = JobRegistry()
        job = registry.create()
        registry.finish(job, "done")
        snapshot = registry.snapshot(job)
        assert snapshot["status"] == "done"
        assert snapshot["events"][-1]["type"] == "done"

    def test_worker_progress_cannot_follow_a_terminal_cancellation(self):
        async def publish_after_cancel() -> None:
            registry = JobRegistry(loop=asyncio.get_running_loop())
            job = registry.create()
            registry.publish(job, {"type": "stage"})
            registry.finish(job, "cancelled", "replaced")
            worker = threading.Thread(
                target=registry.publish,
                args=(job, {"type": "embedded", "done": 1, "total": 2}),
            )
            worker.start()
            worker.join()
            assert [event["type"] for event in job.events] == ["stage", "cancelled"]
            assert job.events[-1]["type"] == "cancelled"

        asyncio.run(publish_after_cancel())


class TestJobOwnershipAndAsyncBoundaries:
    def test_delayed_chunk_cannot_resurrect_a_reset_session(self, client, structured_pdf, monkeypatch):
        from app.pipeline.chunkers import Chunk, ChunkResult

        with open(structured_pdf, "rb") as handle:
            client.post("/api/upload", files={"file": ("old.pdf", handle.read(), "application/pdf")})
        started, release = threading.Event(), threading.Event()

        def delayed_chunk(*_args, **_kwargs):
            started.set()
            assert release.wait(1)
            return ChunkResult(chunks=[Chunk(0, "old", 0, "recursive")], strategy="recursive")

        monkeypatch.setattr(main_module, "chunk", delayed_chunk)
        client.post("/api/chunk", json={"strategy": "recursive"})
        assert started.wait(1)
        assert client.post("/api/reset", json={}).status_code == 200
        release.set()
        time.sleep(0.05)
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        assert state.upload is None
        assert state.chunking is None

    def test_delayed_chunk_cannot_overwrite_a_new_document(self, client, structured_pdf, flat_pdf, monkeypatch):
        from app.pipeline.chunkers import Chunk, ChunkResult

        with open(structured_pdf, "rb") as handle:
            client.post("/api/upload", files={"file": ("old.pdf", handle.read(), "application/pdf")})
        started, release = threading.Event(), threading.Event()

        def delayed_chunk(*_args, **_kwargs):
            started.set()
            assert release.wait(1)
            return ChunkResult(chunks=[Chunk(0, "old", 0, "recursive")], strategy="recursive")

        monkeypatch.setattr(main_module, "chunk", delayed_chunk)
        client.post("/api/chunk", json={"strategy": "recursive"})
        assert started.wait(1)
        with open(flat_pdf, "rb") as handle:
            client.post("/api/upload", files={"file": ("new.pdf", handle.read(), "application/pdf")})
        release.set()
        time.sleep(0.05)
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        assert state.upload["filename"] == "new.pdf"
        assert state.chunking is None

    def test_pdf_parse_runs_without_blocking_the_event_loop(self, monkeypatch, tmp_path):
        from app.pipeline.loader import LoadResult

        release = threading.Event()
        result = LoadResult("text", 1, 4, 0, 0, 0, "doc", [(0, 1)])
        monkeypatch.setattr(main_module, "load_pdf", lambda _path: (release.wait(0.5), result)[1])
        state = main_module.store.get_or_create(None)

        async def run() -> None:
            asyncio.get_running_loop().call_later(0.02, release.set)
            task = asyncio.create_task(main_module._ingest_path(state, tmp_path / "source.pdf", "x.pdf"))
            await asyncio.sleep(0.005)
            assert not task.done()
            await task

        asyncio.run(run())

    def test_collection_lookup_runs_without_blocking_the_event_loop(self, monkeypatch):
        release = threading.Event()
        monkeypatch.setattr(
            main_module.vector_store,
            "get_collection",
            lambda _client: (release.wait(0.5), object())[1],
        )

        async def run() -> None:
            asyncio.get_running_loop().call_later(0.02, release.set)
            task = asyncio.create_task(main_module._collection())
            await asyncio.sleep(0.005)
            assert not task.done()
            await task

        asyncio.run(run())
