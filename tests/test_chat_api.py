"""API contract for the retrieval-chat feature: GET /chat, POST /api/chat,
POST /api/chat/reset.

Same isolation strategy as test_api.py -- a disposable session directory and
an in-process EphemeralClient per test -- plus a fake embeddings object so
retrieval's similarity scores are exact numbers this file controls, not
whatever the real ~90MB model happens to produce. Ollama itself is never
contacted: `probe`/`stream_answer` are monkeypatched, matching
test_generator.py's own hermetic style.
"""

from __future__ import annotations

import time
from dataclasses import replace

import chromadb
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.jobs import JobRegistry
from app.pipeline.generator import GeneratorStatus


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Same fixture as test_api.py's, duplicated rather than imported: this
    file is deliberately standalone (see the task's "keep chat tests
    separate" option), and importing test_api.py just for its fixture would
    make that file's collection changes silently affect this one.
    """
    from app.session import SessionStore

    monkeypatch.setattr(main_module, "store", SessionStore(data_dir=tmp_path))
    chroma_client = chromadb.EphemeralClient()
    monkeypatch.setattr(main_module, "get_client", lambda *args, **kwargs: chroma_client)
    isolated_settings = replace(
        main_module.settings, data_dir=tmp_path, chroma_collection=f"test-chat-{tmp_path.name}"
    )
    monkeypatch.setattr(main_module, "settings", isolated_settings)
    monkeypatch.setattr(main_module.vector_store, "settings", isolated_settings)
    monkeypatch.setattr(main_module, "registry", JobRegistry())
    yield


@pytest.fixture
def client():
    with TestClient(main_module.app) as test_client:
        yield test_client


class FakeEmbeddings:
    """Same shape as test_retriever.py's fake: retrieve() only ever touches
    .model_name and .embed_query(text). A fixed vector regardless of the
    query text is what lets citation/similarity assertions below use exact
    numbers instead of "something plausible".
    """

    model_name = "fake-model"

    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = query_vector

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector


def seed(collection, entries: list[tuple[str, str, list[float], dict]]) -> None:
    """entries: (id, text, vector, metadata). Bypasses write_chunks() so
    these tests can hand-pick exact vectors, same reasoning as
    test_retriever.py's own `add()` helper.
    """
    collection.add(
        ids=[e[0] for e in entries],
        embeddings=[e[2] for e in entries],
        documents=[e[1] for e in entries],
        metadatas=[e[3] for e in entries],
    )


def use_fake_embeddings(monkeypatch, query_vector: list[float]) -> None:
    monkeypatch.setattr(main_module, "build_embeddings", lambda: FakeEmbeddings(query_vector))


def collection(main_module_settings) -> "chromadb.Collection":
    return main_module.vector_store.get_collection(
        main_module.get_client(), main_module_settings.chroma_collection
    )


class TestChatPage:
    def test_chat_is_reachable_with_no_upload_at_all(self, client):
        """The headline requirement: /chat must not depend on this session's
        own pipeline progress, only on what is already in ChromaDB.
        """
        response = client.get("/chat")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "rag_session=" in response.headers["set-cookie"]
        state = main_module.store.get_or_create(response.cookies["rag_session"])
        assert state.unlocked_step() == 1


class TestChatAnswerStates:
    def test_empty_collection_is_unknown(self, client, monkeypatch):
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        response = client.post("/api/chat", json={"message": "What is the leave policy?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]["kind"] == "unknown"
        assert "no documents are indexed" in body["answer"]["text"].lower()
        assert body["answer"]["citations"] == []
        assert body["trace"]["answerable"] is False
        assert body["trace"]["pool_size"] == 0

    def test_nothing_above_threshold_is_unknown(self, client, monkeypatch):
        # The stored vector is orthogonal to the query vector (similarity
        # 0.0), so the default min_score (0.25) rejects it -- a non-empty
        # pool that still can't answer, the second distinct "unknown" path.
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        seed(collection(main_module.settings), [
            ("c1", "irrelevant text", [0.0, 1.0],
             {"source": "handbook.pdf", "page": 1, "chunk_index": 0, "embed_model": "fake-model",
              "doc_id": "d", "strategy": "recursive", "chunk_size": 700, "overlap": 100, "char_count": 15, "parent_id": ""}),
        ])
        response = client.post("/api/chat", json={"message": "anything"})
        body = response.json()
        assert body["answer"]["kind"] == "unknown"
        assert "threshold" in body["answer"]["text"].lower()
        assert body["trace"]["pool_size"] == 1
        assert body["trace"]["answerable"] is False

    def test_generation_unavailable_is_extractive_with_exact_citation(self, client, monkeypatch):
        # Default settings.ollama_base_url is "" -- generation must be
        # unavailable with no monkeypatching of probe() at all, matching the
        # app's offline-by-default posture.
        assert main_module.settings.ollama_base_url == ""
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        seed(collection(main_module.settings), [
            ("c1", "Employees get 20 days of annual leave.", [1.0, 0.0],
             {"source": "handbook.pdf", "page": 4, "chunk_index": 12, "embed_model": "fake-model",
              "doc_id": "d", "strategy": "recursive", "chunk_size": 700, "overlap": 100, "char_count": 39, "parent_id": ""}),
        ])
        response = client.post("/api/chat", json={"message": "How much annual leave?"})
        body = response.json()
        assert body["answer"]["kind"] == "extractive"
        assert "Employees get 20 days of annual leave." in body["answer"]["text"]
        assert body["answer"]["citations"] == [{"page": 4, "source": "handbook.pdf", "chunk_index": 12}]
        assert body["generation"]["available"] is False
        assert body["generation"]["job_id"] is None
        assert body["trace"]["answerable"] is True

    def test_generation_available_streams_through_the_job_registry(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module, "settings", replace(main_module.settings, ollama_base_url="http://fake-ollama:11434")
        )
        monkeypatch.setattr(main_module.vector_store, "settings", main_module.settings)
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        seed(collection(main_module.settings), [
            ("c1", "Employees get 20 days of annual leave.", [1.0, 0.0],
             {"source": "handbook.pdf", "page": 4, "chunk_index": 12, "embed_model": "fake-model",
              "doc_id": "d", "strategy": "recursive", "chunk_size": 700, "overlap": 100, "char_count": 39, "parent_id": ""}),
        ])
        monkeypatch.setattr(
            main_module, "probe",
            lambda base_url, model, timeout=2.0: GeneratorStatus(True, model, base_url, ""),
        )
        monkeypatch.setattr(
            main_module, "stream_answer",
            lambda base_url, model, prompt, timeout=120.0: iter(["20 ", "days."]),
        )

        response = client.post("/api/chat", json={"message": "How much annual leave?"})
        body = response.json()
        assert body["answer"]["kind"] == "generated"
        assert body["generation"]["available"] is True
        job_id = body["generation"]["job_id"]
        assert job_id

        status = None
        for _ in range(100):
            status = client.get(f"/api/status/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.02)
        assert status["status"] == "done", status
        tokens = [e["text"] for e in status["events"] if e["type"] == "token"]
        assert "".join(tokens) == "20 days."

    def test_blank_message_is_400(self, client):
        assert client.post("/api/chat", json={"message": "   "}).status_code == 400
        assert client.post("/api/chat", json={"message": ""}).status_code == 400

    def test_non_json_body_is_400(self, client):
        response = client.post(
            "/api/chat", content=b"not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

    def test_non_object_body_is_400(self, client):
        response = client.post(
            "/api/chat", content=b"[1, 2]", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_non_positive_top_k_is_400(self, client, top_k):
        response = client.post("/api/chat", json={"message": "hi", "top_k": top_k})
        assert response.status_code == 400
        assert "top_k" in response.json()["detail"]

    @pytest.mark.parametrize("min_score", [1.5, -2.0])
    def test_out_of_range_min_score_is_400(self, client, min_score):
        response = client.post("/api/chat", json={"message": "hi", "min_score": min_score})
        assert response.status_code == 400
        assert "-1" in response.json()["detail"] and "1" in response.json()["detail"]

    @pytest.mark.parametrize("mmr_lambda", [5.0, -3.0, 1.01])
    def test_out_of_range_mmr_lambda_is_400(self, client, mmr_lambda):
        """min_score was range-checked and mmr_lambda was not, so 5.0 and -3.0
        both returned 200.

        Not merely inconsistent. MMR scores lambda*sim - (1 - lambda)*redundancy,
        so at lambda=5 the second term becomes -(1-5) = +4 times redundancy: the
        ranking starts actively *rewarding* near-duplicate chunks, while the
        panel keeps labelling the column "MMR score". The room would be shown
        confident, wrong numbers -- and every one of this project's worst bugs
        has been exactly that.
        """
        response = client.post(
            "/api/chat",
            json={"message": "hi", "algorithm": "mmr", "mmr_lambda": mmr_lambda},
        )
        assert response.status_code == 400
        assert "mmr_lambda" in response.json()["detail"]

    def test_lambda_at_both_ends_of_the_range_is_accepted(self, client, monkeypatch):
        # The boundaries are meaningful values, not edge cases to reject: 0 is
        # "diversity only" and 1 is "relevance only", the two ends the panel
        # lets a presenter contrast. A naive exclusive check would break both.
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        for mmr_lambda in (0.0, 1.0):
            response = client.post(
                "/api/chat",
                json={"message": "hi", "algorithm": "mmr", "mmr_lambda": mmr_lambda},
            )
            assert response.status_code == 200, mmr_lambda

    def test_unknown_algorithm_is_400(self, client):
        response = client.post("/api/chat", json={"message": "hi", "algorithm": "bogus"})
        assert response.status_code == 400

    def test_chroma_down_is_503(self, client, monkeypatch):
        monkeypatch.setattr(
            main_module.vector_store,
            "get_collection",
            lambda _client: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        response = client.post("/api/chat", json={"message": "hi"})
        assert response.status_code == 503
        assert "chromadb" in response.json()["detail"].lower()

    def test_bad_body_is_400_even_when_chroma_is_also_down(self, client, monkeypatch):
        # Validation must run before the Chroma round-trip -- otherwise an
        # unreachable store would mask an equally-real client error as a 503.
        monkeypatch.setattr(
            main_module.vector_store,
            "get_collection",
            lambda _client: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        response = client.post("/api/chat", json={"message": "  "})
        assert response.status_code == 400


class TestChatHistory:
    def test_history_length_increments_and_survives_a_store_round_trip(self, client, monkeypatch):
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        seed(collection(main_module.settings), [
            ("c1", "some text", [1.0, 0.0],
             {"source": "h.pdf", "page": 1, "chunk_index": 0, "embed_model": "fake-model",
              "doc_id": "d", "strategy": "recursive", "chunk_size": 700, "overlap": 100, "char_count": 9, "parent_id": ""}),
        ])
        first = client.post("/api/chat", json={"message": "first question"}).json()
        assert first["history_length"] == 1
        second = client.post("/api/chat", json={"message": "second question"}).json()
        assert second["history_length"] == 2

        rehydrated = main_module.store.get_or_create(client.cookies["rag_session"])
        assert len(rehydrated.chat) == 2
        assert [entry["question"] for entry in rehydrated.chat] == ["first question", "second question"]
        # Chat must never affect the pipeline's own unlock rule.
        assert rehydrated.unlocked_step() == 1

    def test_reset_clears_chat_but_leaves_the_pipeline_untouched(self, client, structured_pdf, monkeypatch):
        use_fake_embeddings(monkeypatch, [1.0, 0.0])
        seed(collection(main_module.settings), [
            ("c1", "some text", [1.0, 0.0],
             {"source": "h.pdf", "page": 1, "chunk_index": 0, "embed_model": "fake-model",
              "doc_id": "d", "strategy": "recursive", "chunk_size": 700, "overlap": 100, "char_count": 9, "parent_id": ""}),
        ])
        with open(structured_pdf, "rb") as handle:
            client.post("/api/upload", files={"file": ("handbook.pdf", handle.read(), "application/pdf")})
        client.post("/api/chat", json={"message": "one"})

        before = main_module.store.get_or_create(client.cookies["rag_session"])
        assert len(before.chat) == 1
        assert before.upload is not None

        response = client.post("/api/chat/reset")
        assert response.status_code == 200

        after = main_module.store.get_or_create(client.cookies["rag_session"])
        assert after.chat == []
        assert after.upload == before.upload
        assert after.chunking == before.chunking
        assert after.embedding == before.embedding


class TestChatFrontend:
    """Markup and asset contract for the /chat page itself: chat.html,
    chat.css, chat.js. Distinct from TestChatPage above, which only checks
    that the route doesn't gate on session progress -- these check what a
    presenter and a screen reader actually see.
    """

    def test_page_has_both_columns_stage_names_and_controls(self, client):
        body = client.get("/chat").text
        # The two named columns from the brief -- chat on the left, the
        # inspector on the right -- not just "some markup exists".
        assert 'class="chat-col"' in body
        assert 'id="inspector"' in body
        # The five retrieval stages, in the fixed pipeline order the panel
        # reads top to bottom: embed_query -> search -> rank -> filter ->
        # assemble. Asserting all five by name, not just "a stage list
        # exists", so removing one silently would fail this test.
        for stage in ("embed_query", "search", "rank", "filter", "assemble"):
            assert f'data-stage="{stage}"' in body
        # The controls the brief calls out by name: top_k, algorithm,
        # mmr_lambda (present but disabled until MMR is chosen), min_score.
        assert 'id="top-k"' in body
        assert 'id="algorithm"' in body
        assert 'id="mmr-lambda"' in body and 'disabled' in body
        assert 'id="min-score"' in body
        assert 'id="message"' in body

    def test_css_and_js_are_served_with_no_external_reference(self, client):
        body = client.get("/chat").text
        css = client.get("/static/chat.css")
        js = client.get("/static/chat.js")
        assert css.status_code == 200
        assert js.status_code == 200
        # The talk is presented offline (see CLAUDE.md); a CDN or font-host
        # reference here would render blank with no network, exactly like
        # the stock Swagger UI test_api.py already guards against.
        combined = "\n".join((body, css.text, js.text)).lower()
        assert "https://" not in combined
        assert "http://" not in combined
        assert "cdn." not in combined

    def test_page_references_its_own_assets(self, client):
        body = client.get("/chat").text
        assert 'href="/static/app.css"' in body
        assert 'href="/static/chat.css"' in body
        assert 'src="/static/chat.js"' in body

    def test_state_is_injected_and_tojson_escapes_a_script_close_tag(self, client):
        # A chat answer's text is arbitrary retrieved document content, so a
        # chunk that happens to contain "</script>" must not be able to break
        # out of the inline <script> block below it. Jinja's tojson filter
        # escapes "<" to "\u003c" for exactly this reason -- this pins that
        # chat.html actually renders state through that filter, not a raw
        # string interpolation that would skip the escaping.
        state = main_module.store.get_or_create(None)
        state.chat = [{
            "message_id": "m1",
            "question": "one",
            "answer": {
                "kind": "extractive",
                "text": "</script><script>alert(1)</script>",
                "citations": [],
            },
            "generation": {"available": False, "model": "m", "detail": "", "job_id": None},
        }]
        main_module.store.save(state)

        body = client.get("/chat", cookies={"rag_session": state.session_id}).text
        assert "window.__STATE__" in body
        assert "</script><script>alert(1)</script>" not in body
        assert "\\u003c/script\\u003e" in body
