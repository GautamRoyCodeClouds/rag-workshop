"""API contracts for the ingestion wizard and its background jobs."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

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

    def test_embedding_progress_exposes_accessible_range_semantics(self, client):
        body = client.get("/").text
        assert (
            'id="embed-progress" role="progressbar" aria-label="Embedding progress" '
            'aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"'
        ) in body

    def test_new_document_reset_and_stale_follow_suppression_run_in_the_client(self):
        """Execute the real browser script with a minimal DOM.

        This catches removal of the operation epoch or downstream reset by
        observing their effects, without pinning either helper's source text.
        """
        script_path = Path(main_module.BASE_DIR) / "static" / "app.js"
        if shutil.which("node") is None:
            pytest.skip("Node.js is not installed in the Python app image.")
        harness = r"""
const fs = require('fs');
const vm = require('vm');
class Element {
  constructor(id = '') {
    this.id = id; this.children = []; this.dataset = {}; this.style = {};
    this.hidden = false; this.disabled = false; this.value = '0';
    this.attributes = {}; this.classList = { add() {}, remove() {} };
  }
  replaceChildren(...items) { this.children = items; }
  appendChild(item) { this.children.push(item); return item; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  toggleAttribute(name, enabled) { this.attributes[name] = String(enabled); }
  addEventListener() {}
}
const elements = new Map();
const get = (id) => {
  if (!elements.has(id)) elements.set(id, new Element(id));
  return elements.get(id);
};
const strategy = new Element('recursive');
strategy.dataset = { key:'recursive', usesSize:'true', usesOverlap:'true', extra:'' };
strategy.setAttribute('aria-pressed', 'true');
const sources = [];
let collectionResolve = null;
class FakeEventSource {
  constructor(url) { this.url = url; this.closed = false; sources.push(this); }
  close() { this.closed = true; }
}
const context = {
  console, setTimeout, clearTimeout, EventSource: FakeEventSource, Element,
  FormData: class { append() {} },
  fetch: async (url) => {
    if (url.startsWith('/api/collection')) {
      return new Promise((resolve) => { collectionResolve = resolve; });
    }
    return { ok:true, json:async () => ({ job_id:'embed-job' }) };
  },
  confirm: () => true,
  window: { __STATE__: { unlocked_step:1 } },
  document: {
    getElementById: get,
    createElement: () => new Element(),
    querySelector: () => strategy,
    querySelectorAll: () => [strategy],
  },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext(`
  state.chunkCount = 4; state.offset = 50;
  $('chunk-preview').appendChild(new Element('old-chunk'));
  $('chunk-notes').appendChild(new Element('old-note'));
  $('records').appendChild(new Element('old-record'));
  $('embed-progress').setAttribute('aria-valuenow', '76');
  $('embed-bar').style.width = '76%';
  $('chunk-count').textContent = '4 chunks';
  $('record-count').textContent = '4 records';
  loadRecords();
`, context);
vm.runInContext(`
  loadDocument(() => Promise.resolve({
    upload: {
      page_count:1, char_count:10, pages_without_text:0,
      boilerplate_lines_removed:0, invisible_chars_removed:0
    },
    unlocked_step:2
  }))
`, context);
setImmediate(() => {
  const preview = get('chunk-preview');
  if (preview.children.length !== 1 || preview.children[0].textContent !== 'Chunks will appear here as the document is split.') process.exit(10);
  if (get('records').children.length !== 1 || get('records').children[0].textContent !== 'Stored vectors will appear here.') process.exit(11);
  if (get('embed-progress').getAttribute('aria-valuenow') !== '0') process.exit(12);
  if (get('embed-bar').style.width !== '0%') process.exit(13);
  if (get('chunk-count').textContent !== '' || get('record-count').textContent !== '') process.exit(14);

  vm.runInContext(`
    staleEvents = 0;
    const token = beginOperation();
    follow('old-job', () => { staleEvents += 1; }, () => {}, token);
    beginOperation();
  `, context);
  sources[0].onmessage({ data:'{"type":"chunk","index":0}', lastEventId:'1' });
  if (vm.runInContext('staleEvents', context) !== 0) process.exit(15);

  collectionResolve({
    ok:true,
    json:async () => ({
      total:1, offset:0,
      records:[{id:'stale', text:'stale record', vector_preview:[1], dims:1, vector_norm:1, metadata:{}}]
    })
  });
  setImmediate(() => {
    if (get('records').children.length !== 1 || get('records').children[0].textContent !== 'Stored vectors will appear here.') process.exit(16);
    get('run-embed').onclick();
    setImmediate(() => {
      const embedSource = sources[sources.length - 1];
      embedSource.onmessage({ data:'{"type":"embedded","done":1,"total":2}', lastEventId:'1' });
      if (get('embed-bar').style.width !== '50%') process.exit(17);
      if (get('embed-progress').getAttribute('aria-valuenow') !== '50') process.exit(18);

      get('run-chunk').disabled = true;
      get('run-embed').disabled = true;
      vm.runInContext('beginOperation()', context);
      if (get('run-chunk').disabled) process.exit(19);
      if (get('run-embed').disabled) process.exit(20);

      get('embed-bar').style.width = '100%';
      get('embed-progress').setAttribute('aria-valuenow', '100');
      get('embed-status').textContent = 'old vectors';
      get('records').replaceChildren(new Element('old-record'));
      get('record-count').textContent = '1 record';
      vm.runInContext(`
        fetch = async (url) => ({
          ok:true,
          json:async () => url.startsWith('/api/status/')
            ? { status:'done', events:[], cursor:1, session:{unlocked_step:4} }
            : { job_id:'chunk-job' }
        });
        $('run-chunk').onclick();
      `, context);
      setImmediate(() => {
        const chunkSource = sources[sources.length - 1];
        chunkSource.onmessage({ data:'{"type":"done"}', lastEventId:'1' });
        setImmediate(() => {
          if (get('embed-bar').style.width !== '0%') process.exit(21);
          if (get('embed-progress').getAttribute('aria-valuenow') !== '0') process.exit(22);
          if (get('embed-status').textContent !== '') process.exit(23);
          if (get('records').children.length !== 1 || get('records').children[0].textContent !== 'Stored vectors will appear here.') process.exit(24);
          if (get('record-count').textContent !== '') process.exit(25);
        });
      });
    });
  });
});
"""
        result = subprocess.run(
            ["node", "-e", harness, str(script_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout


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

    def test_failed_embed_collection_preflight_leaves_no_running_job(
        self, client, structured_pdf, monkeypatch
    ):
        from app.pipeline.chunkers import Chunk

        self.upload(client, structured_pdf)
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        state.chunks = [Chunk(0, "chunk", 0, "recursive")]
        state.chunking = {"size": 200, "overlap": 20}
        main_module.store.save(state)
        monkeypatch.setattr(
            main_module.vector_store,
            "get_collection",
            lambda _client: (_ for _ in ()).throw(RuntimeError("offline")),
        )

        response = client.post("/api/embed")

        assert response.status_code == 503
        assert all(
            job.status != "running"
            for job in main_module.registry._jobs.values()
            if job.session_id == state.session_id
        )


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

    def test_collection_read_failure_uses_the_actionable_503_contract(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module.vector_store,
            "read_records",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("read failed")),
        )

        response = client.get("/api/collection")

        assert response.status_code == 503
        assert "chromadb is unreachable" in response.json()["detail"].lower()


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
        client.get("/")
        job = main_module.registry.create(client.cookies["rag_session"])
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
        client.get("/")
        job = main_module.registry.create(client.cookies["rag_session"])
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
    @pytest.mark.parametrize("endpoint", ["status", "events"])
    def test_job_output_is_hidden_from_another_session(self, client, endpoint):
        client.get("/")
        owner_id = client.cookies["rag_session"]
        job = main_module.registry.create(owner_id)
        main_module.registry.publish(job, {"type": "chunk", "text": "private text"})
        main_module.registry.finish(job, "done")

        with TestClient(main_module.app) as stranger:
            stranger.get("/")
            response = stranger.get(f"/api/{endpoint}/{job.job_id}")

        assert response.status_code == 404
        assert "private text" not in response.text

    @pytest.mark.parametrize("endpoint", ["status", "events"])
    def test_job_reads_rotate_a_malformed_cookie_before_ownership_lookup(
        self, client, endpoint
    ):
        job = main_module.registry.create("somevalidowner")
        main_module.registry.finish(job, "done")
        response = client.get(
            f"/api/{endpoint}/{job.job_id}",
            cookies={"rag_session": "../../bad"},
        )

        assert response.status_code == 400
        assert "rag_session=" in response.headers["set-cookie"]

    def test_delayed_upload_cannot_resurrect_a_reset_session(self, client, monkeypatch):
        from app.pipeline.loader import LoadResult

        client.get("/")
        started, release = threading.Event(), threading.Event()
        result = LoadResult("old", 1, 3, 0, 0, 0, "old-doc", [(0, 1)])

        def delayed_load(_path):
            started.set()
            assert release.wait(1)
            return result

        monkeypatch.setattr(main_module, "load_pdf", delayed_load)
        response_box = {}

        def upload_old():
            response_box["response"] = client.post(
                "/api/upload",
                files={"file": ("old.pdf", b"%PDF-old", "application/pdf")},
            )

        thread = threading.Thread(target=upload_old)
        thread.start()
        assert started.wait(1)
        assert client.post("/api/reset", json={}).status_code == 200
        release.set()
        thread.join(2)

        assert not thread.is_alive()
        assert response_box["response"].status_code == 409
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        assert state.upload is None
        assert state.chunking is None

    def test_delayed_upload_cannot_overwrite_a_newer_upload(self, client, monkeypatch):
        from app.pipeline.loader import LoadResult

        client.get("/")
        started, release = threading.Event(), threading.Event()
        call_lock = threading.Lock()
        calls = 0

        def ordered_load(_path):
            nonlocal calls
            with call_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                started.set()
                assert release.wait(1)
                return LoadResult("old", 1, 3, 0, 0, 0, "old-doc", [(0, 1)])
            return LoadResult("new", 1, 3, 0, 0, 0, "new-doc", [(0, 1)])

        monkeypatch.setattr(main_module, "load_pdf", ordered_load)
        response_box = {}
        old_thread = threading.Thread(
            target=lambda: response_box.setdefault(
                "response",
                client.post(
                    "/api/upload",
                    files={"file": ("old.pdf", b"%PDF-old", "application/pdf")},
                ),
            ),
        )
        old_thread.start()
        assert started.wait(1)
        newer = client.post(
            "/api/upload",
            files={"file": ("new.pdf", b"%PDF-new", "application/pdf")},
        )
        assert newer.status_code == 200
        release.set()
        old_thread.join(2)

        assert not old_thread.is_alive()
        assert response_box["response"].status_code == 409
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        assert state.upload["filename"] == "new.pdf"
        assert state.upload["doc_id"] == "new-doc"
        assert Path(state.pdf_path).read_bytes() == b"%PDF-new"

    def test_delayed_local_load_cannot_resurrect_a_reset_session(
        self, client, monkeypatch, tmp_path
    ):
        from app.pipeline.loader import LoadResult

        local = tmp_path / "local.pdf"
        local.write_bytes(b"%PDF-local")
        monkeypatch.setattr(
            main_module,
            "settings",
            replace(main_module.settings, local_pdf_path=str(local)),
        )
        client.get("/")
        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(
            main_module,
            "load_pdf",
            lambda _path: (
                started.set(),
                release.wait(1),
                LoadResult("local", 1, 5, 0, 0, 0, "local-doc", [(0, 1)]),
            )[-1],
        )
        response_box = {}
        thread = threading.Thread(
            target=lambda: response_box.setdefault(
                "response", client.post("/api/use-local")
            )
        )
        thread.start()
        assert started.wait(1)
        assert client.post("/api/reset", json={}).status_code == 200
        release.set()
        thread.join(2)

        assert not thread.is_alive()
        assert response_box["response"].status_code == 409
        assert main_module.store.get_or_create(client.cookies["rag_session"]).upload is None

    def test_embed_claims_ownership_before_delayed_collection_lookup(
        self, client, monkeypatch
    ):
        from app.pipeline.chunkers import Chunk

        client.get("/")
        state = main_module.store.get_or_create(client.cookies["rag_session"])
        state.upload = {"filename": "old.pdf", "doc_id": "old-doc"}
        state.chunks = [Chunk(0, "old chunk", 0, "recursive")]
        state.chunking = {"size": 200, "overlap": 20}
        main_module.store.save(state)
        started, release = threading.Event(), threading.Event()
        monkeypatch.setattr(
            main_module.vector_store,
            "get_collection",
            lambda _client: (started.set(), release.wait(1), object())[-1],
        )
        monkeypatch.setattr(main_module, "build_embeddings", lambda: object())
        monkeypatch.setattr(main_module, "embed_batched", lambda *_args, **_kwargs: [[1.0]])
        monkeypatch.setattr(main_module.vector_store, "write_chunks", lambda *_args, **_kwargs: 1)
        response_box = {}

        def start_embed():
            response_box["response"] = client.post("/api/embed")

        thread = threading.Thread(target=start_embed)
        thread.start()
        assert started.wait(1)
        assert client.post("/api/reset", json={}).status_code == 200
        release.set()
        thread.join(2)
        time.sleep(0.05)

        assert not thread.is_alive()
        assert response_box["response"].status_code == 202
        fresh = main_module.store.get_or_create(client.cookies["rag_session"])
        assert fresh.upload is None
        assert fresh.embedding is None

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

    def test_session_persistence_does_not_block_the_event_loop(self, monkeypatch, tmp_path):
        from app.pipeline.loader import LoadResult

        started, release = threading.Event(), threading.Event()
        result = LoadResult("text", 1, 4, 0, 0, 0, "doc", [(0, 1)])
        monkeypatch.setattr(main_module, "load_pdf", lambda _path: result)
        monkeypatch.setattr(
            main_module.store,
            "save",
            lambda _state: (started.set(), release.wait(0.5)),
        )
        state = main_module.store.get_or_create(None)

        async def run() -> None:
            task = asyncio.create_task(
                main_module._ingest_path(state, tmp_path / "source.pdf", "x.pdf")
            )
            assert await asyncio.to_thread(started.wait, 0.2)
            await asyncio.sleep(0.01)
            assert not task.done()
            release.set()
            await task

        started_at = time.monotonic()
        asyncio.run(run())
        assert time.monotonic() - started_at < 0.3
