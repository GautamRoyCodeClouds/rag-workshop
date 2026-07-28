"""FastAPI routes for the five-step ingestion page."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.jobs import registry, sse_format
from app.pipeline import store as vector_store
from app.pipeline.chunkers import STRATEGIES, UnknownStrategyError, chunk
from app.pipeline.embedder import build_embeddings, embed_batched
from app.pipeline.generator import build_prompt, probe, stream_answer
from app.pipeline.loader import EmptyDocumentError, load_pdf
from app.pipeline.retriever import Candidate, retrieve
from app.pipeline.store import get_client
from app.session import store

BASE_DIR = Path(__file__).parent
SESSION_COOKIE = "rag_session"


@asynccontextmanager
async def lifespan(_app):
    yield
    await registry.shutdown()


# docs_url=None disables FastAPI's built-in Swagger page, which is replaced
# below. The stock one loads its JavaScript and CSS from cdn.jsdelivr.net, so it
# renders blank with no network -- and this app is presented offline.
app = FastAPI(
    title="RAG Ingestion Pipeline",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/api/docs", include_in_schema=False)
def api_docs():
    """Swagger UI served from assets vendored into the image at build time.

    Same trick as the embedding model: fetch it once while building, so runtime
    needs no network. app/static/vendor/ is populated by the Dockerfile and is
    gitignored -- downloaded, not committed.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="RAG Ingestion Pipeline - API",
        swagger_js_url="/static/vendor/swagger-ui-bundle.js",
        swagger_css_url="/static/vendor/swagger-ui.css",
        swagger_favicon_url="data:,",
    )


class MalformedSessionError(Exception):
    """A bad client cookie accompanied by a safe replacement session id."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


@app.exception_handler(MalformedSessionError)
async def malformed_session_error(_request: Request, exc: MalformedSessionError):
    response = JSONResponse(status_code=400, content={"detail": "Malformed session cookie."})
    _with_session_cookie(response, exc.session_id)
    return response


def _session(request: Request):
    try:
        return store.get_or_create(request.cookies.get(SESSION_COOKIE))
    except ValueError as exc:
        fresh = store.get_or_create(None)
        raise MalformedSessionError(fresh.session_id) from exc


def _with_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")


def _json_response(body: dict, status_code: int = 200) -> Response:
    return Response(content=json.dumps(body), media_type="application/json", status_code=status_code)


async def _collection():
    try:
        return await asyncio.to_thread(lambda: vector_store.get_collection(get_client()))
    except Exception as exc:  # noqa: BLE001 - displayed as a retryable UI error
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB is unreachable ({type(exc).__name__}). Check ChromaDB and retry.",
        ) from exc


def _document_text(state) -> str:
    """The cleaned text to chunk.

    Read from the cache written at upload time. Re-parsing here instead would
    redo the most expensive step in the pipeline on every strategy change:
    measured at 0.60s for load_pdf on a structurally simple 270-page PDF -- and
    materially worse on a real one, where layout, not character count, drives
    pypdf's cost -- against 0.01-0.43s for the chunking itself. The presenter
    compares all five strategies live, so that was paid on every comparison.

    It also makes the pipeline stages honest. The deck teaches load -> clean ->
    chunk as distinct steps; re-parsing here quietly re-ran the first two inside
    the third.

    Falls back to re-parsing when the cache is absent: app-data survives image
    rebuilds, so a session persisted before this cache existed must still work.
    """
    cache = store.text_cache_path(state.session_id)
    if cache.is_file():
        return cache.read_text(encoding="utf-8")
    return load_pdf(state.pdf_path).text


def _superseded() -> HTTPException:
    return HTTPException(status_code=409, detail="Superseded by a newer session action.")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    state = _session(request)
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "strategies": list(STRATEGIES.values()),
            "settings": settings,
            "state": state.to_json(),
            "has_local_pdf": settings.local_pdf is not None,
        },
    )
    _with_session_cookie(response, state.session_id)
    return response


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    """The retrieval-chat page.

    Deliberately does not gate on unlocked_step(): chat depends only on
    whatever is already in ChromaDB, not on this browser's session having run
    the ingestion pipeline. A second machine, or the same one after "Reset
    everything", must still be able to open /chat and query whatever a prior
    session already embedded.
    """
    state = _session(request)
    response = templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "settings": settings,
            "state": state.to_json(),
        },
    )
    _with_session_cookie(response, state.session_id)
    return response


@app.get("/api/config")
def config() -> dict:
    return {
        "default_chunk_size": settings.default_chunk_size,
        "default_chunk_overlap": settings.default_chunk_overlap,
        "default_strategy": settings.default_strategy,
        "semantic_percentile": settings.semantic_percentile,
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
        "max_upload_mb": settings.max_upload_mb,
        "has_local_pdf": settings.local_pdf is not None,
        "strategies": [
            {
                "key": info.key,
                "label": info.label,
                "verdict": info.verdict,
                "uses_size": info.uses_size,
                "uses_overlap": info.uses_overlap,
                "extra_control": info.extra_control,
            }
            for info in STRATEGIES.values()
        ],
    }


@app.get("/api/health")
def health() -> dict:
    chroma_ok, detail = False, ""
    try:
        chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port).heartbeat()
        chroma_ok = True
    except Exception as exc:  # noqa: BLE001 - health is explicitly non-fatal
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "status": "ok" if chroma_ok else "degraded",
        "chroma": {"reachable": chroma_ok, "detail": detail},
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
    }


async def _load_ingest(path: Path, display_name: str):
    try:
        return await asyncio.to_thread(load_pdf, path)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - invalid/encrypted PDFs are client errors
        raise HTTPException(
            status_code=400,
            detail=(f"Could not read {display_name}: {type(exc).__name__}. " "Remove password protection and retry."),
        ) from exc


def _apply_ingest(state, path: Path, display_name: str, result) -> None:
    state.pdf_path = str(path)
    state.page_offsets = result.page_offsets
    state.upload = {
        "filename": display_name,
        "doc_id": result.doc_id,
        "page_count": result.page_count,
        "char_count": result.char_count,
        "pages_without_text": result.pages_without_text,
        "boilerplate_lines_removed": result.boilerplate_lines_removed,
        "invisible_chars_removed": result.invisible_chars_removed,
    }
    state.chunking = None
    state.embedding = None
    state.chunks = []

    # Cache the cleaned text so comparing chunking strategies does not re-parse
    # the document each time -- see _document_text for the measured cost. Not
    # wrapped in try/except: the uploads directory was just written to, so a
    # failure here means the data volume is broken and the upload should say so
    # rather than silently degrade every later chunk run.
    cache = store.text_cache_path(state.session_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(result.text, encoding="utf-8")


async def _ingest_path(state, path: Path, display_name: str) -> dict:
    """Load and persist a path; production routes use the claimed variant below."""
    result = await _load_ingest(path, display_name)
    _apply_ingest(state, path, display_name, result)
    await asyncio.to_thread(store.save, state)
    return state.to_json()


async def _ingest_claimed(
    state,
    path: Path,
    display_name: str,
    generation: int,
    *,
    promote_to: Path | None = None,
) -> dict:
    """Parse outside the lock, then atomically promote and persist if still owned."""
    result = await _load_ingest(path, display_name)
    async with registry.session_lock(state.session_id):
        if not registry.generation_is_current(state.session_id, generation):
            raise _superseded()
        final_path = promote_to or path
        if promote_to is not None:
            await asyncio.to_thread(path.replace, promote_to)
        _apply_ingest(state, final_path, display_name, result)
        await asyncio.to_thread(store.save, state)
        return state.to_json()


def _write_staged_upload(uploads: Path, payload: bytes) -> Path:
    uploads.mkdir(parents=True, exist_ok=True)
    staged = uploads / f"source.pdf.{uuid.uuid4().hex}.tmp"
    staged.write_bytes(payload)
    return staged


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> Response:
    state = _session(request)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    generation = await registry.claim_session(state.session_id)
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large: {len(payload) / 1_048_576:.1f} MB exceeds the {settings.max_upload_mb} MB limit.",
        )
    # The store owns this path so that upload and reset cannot disagree about
    # where the document lives -- a mismatch would leave the file behind on
    # reset with nothing to reveal it.
    uploads = store.uploads_dir(state.session_id)
    target = uploads / "source.pdf"
    staged = await asyncio.to_thread(_write_staged_upload, uploads, payload)
    try:
        body = await _ingest_claimed(
            state,
            staged,
            file.filename or "document.pdf",
            generation,
            promote_to=target,
        )
    finally:
        await asyncio.to_thread(staged.unlink, missing_ok=True)
    response = _json_response(body)
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/use-local")
async def use_local(request: Request) -> Response:
    local = settings.local_pdf
    if local is None:
        raise HTTPException(status_code=404, detail="No local document is configured.")
    state = _session(request)
    generation = await registry.claim_session(state.session_id)
    response = _json_response(
        await _ingest_claimed(state, local, local.name, generation)
    )
    _with_session_cookie(response, state.session_id)
    return response


async def _request_body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be JSON.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return value


@app.post("/api/chunk")
async def start_chunking(request: Request) -> Response:
    state = _session(request)
    if not state.upload:
        raise HTTPException(status_code=409, detail="Upload a document first.")
    generation = await registry.claim_session(state.session_id)
    body = await _request_body(request)
    strategy = body.get("strategy", settings.default_strategy)
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy {strategy!r}.")
    try:
        size = int(body.get("size", settings.default_chunk_size))
        overlap = int(body.get("overlap", settings.default_chunk_overlap))
        percentile = int(body.get("percentile", settings.semantic_percentile))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Chunk settings must be integers.") from exc
    async with registry.session_lock(state.session_id):
        if not registry.generation_is_current(state.session_id, generation):
            raise _superseded()
        state = store.get_or_create(state.session_id)
        if not state.upload:
            raise HTTPException(status_code=409, detail="Upload a document first.")
        job = registry.create(state.session_id)

    async def run() -> None:
        try:
            registry.publish(job, {"type": "stage", "message": f"Splitting with {strategy}..."})
            text = await asyncio.to_thread(_document_text, state)
            embeddings = await asyncio.to_thread(build_embeddings) if strategy == "semantic" else None
            result = await asyncio.to_thread(
                chunk,
                text,
                strategy=strategy,
                size=size,
                overlap=overlap,
                embeddings=embeddings,
                percentile=percentile,
            )
            for note in result.notes:
                registry.publish(job, {"type": "note", "message": note})
            # Splitters return a full result atomically; these events only stream rendering.
            for piece in result.chunks:
                registry.publish(job, {
                    "type": "chunk",
                    "index": piece.index,
                    "text": piece.text,
                    "char_count": len(piece.text),
                    "page": state.page_for_offset(piece.start),
                    "parent_id": piece.parent_id,
                })
            async with registry.session_lock(state.session_id):
                if not registry.is_current(job):
                    return
                state.chunks = result.chunks
                state.chunking = {
                    "strategy": strategy,
                    "size": size,
                    "overlap": overlap,
                    "percentile": percentile,
                    "chunk_count": len(result.chunks),
                    "sections_detected": result.sections_detected,
                    "fell_back": result.fell_back,
                    "notes": result.notes,
                }
                state.embedding = None
                await asyncio.to_thread(store.save, state)
                registry.publish(job, {
                    "type": "summary",
                    "chunk_count": len(result.chunks),
                    "sections_detected": result.sections_detected,
                    "fell_back": result.fell_back,
                    "unlocked_step": state.unlocked_step(),
                })
                registry.finish(job, "done", session=state.to_json())
        except UnknownStrategyError as exc:
            registry.finish(job, "error", str(exc))
        except Exception as exc:  # noqa: BLE001 - delivered to the job UI
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    registry.register_task(job, asyncio.create_task(run()))
    response = _json_response({"job_id": job.job_id}, status_code=202)
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/embed")
async def start_embedding(request: Request) -> Response:
    state = _session(request)
    if not state.chunks:
        raise HTTPException(status_code=409, detail="Chunk the document first.")
    generation = await registry.claim_session(state.session_id)
    async with registry.session_lock(state.session_id):
        if not registry.generation_is_current(state.session_id, generation):
            raise _superseded()
        state = store.get_or_create(state.session_id)
        if not state.chunks:
            raise HTTPException(status_code=409, detail="Chunk the document first.")
        job = registry.create(state.session_id)
    try:
        collection = await _collection()
    except HTTPException as exc:
        registry.finish(job, "error", str(exc.detail))
        raise
    if not registry.is_current(job):
        response = _json_response({"job_id": job.job_id}, status_code=202)
        _with_session_cookie(response, state.session_id)
        return response

    async def run() -> None:
        try:
            registry.publish(job, {"type": "stage", "message": f"Loading {settings.embed_model}..."})
            embeddings = await asyncio.to_thread(build_embeddings)
            vectors = await asyncio.to_thread(
                embed_batched,
                embeddings,
                [piece.text for piece in state.chunks],
                settings.embed_batch_size,
                lambda done, total: registry.publish(job, {"type": "embedded", "done": done, "total": total}),
            )
            registry.publish(job, {"type": "stage", "message": "Writing to ChromaDB..."})
            async with registry.session_lock(state.session_id):
                if not registry.is_current(job):
                    return
                written = await asyncio.to_thread(
                    vector_store.write_chunks,
                    collection,
                    chunks=state.chunks,
                    vectors=vectors,
                    doc_id=state.upload["doc_id"],
                    source=state.upload["filename"],
                    size=state.chunking["size"],
                    overlap=state.chunking["overlap"],
                    embed_model=settings.embed_model,
                    page_for_offset=state.page_for_offset,
                )
                state.embedding = {"model": settings.embed_model, "dims": settings.embed_dims, "vectors_written": written}
                await asyncio.to_thread(store.save, state)
                registry.publish(job, {
                    "type": "summary",
                    "vectors_written": written,
                    "unlocked_step": state.unlocked_step(),
                })
                registry.finish(job, "done", session=state.to_json())
        except Exception as exc:  # noqa: BLE001 - delivered to the job UI
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    registry.register_task(job, asyncio.create_task(run()))
    response = _json_response({"job_id": job.job_id}, status_code=202)
    _with_session_cookie(response, state.session_id)
    return response


def _cursor(value: str | int | None) -> int:
    try:
        cursor = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Event cursor must be a non-negative integer.") from exc
    if cursor < 0:
        raise HTTPException(status_code=400, detail="Event cursor must be a non-negative integer.")
    return cursor


@app.get("/api/events/{job_id}")
async def events(job_id: str, request: Request, after: int | None = None):
    state = _session(request)
    job = registry.get(job_id)
    if job is None or job.session_id != state.session_id:
        raise HTTPException(status_code=404, detail="Unknown job.")

    cursor = _cursor(request.headers.get("last-event-id") if request.headers.get("last-event-id") is not None else after)

    async def stream():
        history, queue, terminal = registry.subscribe(job, after=cursor)
        try:
            for event in history:
                yield sse_format(event)
            if terminal:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield sse_format(event)
        finally:
            registry.unsubscribe(job, queue)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/api/status/{job_id}")
def job_status(job_id: str, request: Request, after: int = 0) -> dict:
    state = _session(request)
    job = registry.get(job_id)
    if job is None or job.session_id != state.session_id:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return registry.snapshot(job, after=_cursor(after))


@app.get("/api/collection")
async def collection_records(offset: int = 0, limit: int = 25) -> dict:
    try:
        collection = await _collection()
        return await asyncio.to_thread(
            vector_store.read_records,
            collection,
            offset=offset,
            limit=min(limit, 100),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - same actionable read contract
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB is unreachable ({type(exc).__name__}). Check ChromaDB and retry.",
        ) from exc


@app.post("/api/reset")
async def reset(request: Request) -> Response:
    state = _session(request)
    generation = await registry.claim_session(state.session_id)
    body = await _request_body(request)
    async with registry.session_lock(state.session_id):
        if not registry.generation_is_current(state.session_id, generation):
            raise _superseded()
        if body.get("drop_collection"):
            try:
                await asyncio.to_thread(lambda: vector_store.drop_collection(get_client()))
            except Exception as exc:  # noqa: BLE001 - unavailable store must stay visible
                raise HTTPException(status_code=503, detail=f"ChromaDB collection reset failed ({type(exc).__name__}).") from exc
        fresh = await asyncio.to_thread(store.reset, state.session_id)
    response = _json_response(fresh.to_json())
    _with_session_cookie(response, fresh.session_id)
    return response


# --- Retrieval chat -----------------------------------------------------

def _trace_to_json(trace) -> dict:
    """RetrievalTrace (and its nested Candidate/Stage dataclasses) as plain
    JSON-safe data. dataclasses.asdict recurses through both nested types,
    so this stays correct if either dataclass gains a field.
    """
    return asdict(trace)


def _citation(candidate: Candidate) -> dict:
    # "" is Chroma's None-sentinel (see store.py); page/chunk_index/source
    # should always be present on anything write_chunks wrote, but a record
    # written by an older schema (app-data survives rebuilds) might not have
    # every key, so .get() with the same honest fallbacks store.py uses.
    return {
        "page": candidate.metadata.get("page", ""),
        "source": candidate.metadata.get("source", ""),
        "chunk_index": candidate.metadata.get("chunk_index", -1),
    }


def _extractive_answer(selected: list[Candidate]) -> str:
    """Assemble an answer straight from the selected chunks, no LLM involved.

    This is what the room sees whenever Ollama is not configured or not
    reachable -- the common case, since the app is offline by default. It is
    deliberately just the chunks themselves with their citations, not a
    summary: manufacturing prose here would blur the line this whole feature
    exists to show, between "the retriever found this text" and "a model
    said this".
    """
    blocks = []
    for i, candidate in enumerate(selected, start=1):
        metadata = candidate.metadata
        source = metadata.get("source") or "unknown source"
        page = metadata.get("page", "")
        label = f"[{i}] {source}, page {page}" if page != "" else f"[{i}] {source}"
        blocks.append(f"{label}\n{candidate.text}")
    return "\n\n".join(blocks)


@app.post("/api/chat/reset")
async def chat_reset(request: Request) -> Response:
    """Clear chat history only.

    Deliberately does not touch state.upload/chunking/embedding, does not
    call registry.claim_session (that would cancel any running chunk/embed
    job for no reason), and never drops the Chroma collection -- chat is a
    read-only consumer of whatever is already indexed, so "start the chat
    over" and "start the pipeline over" must stay two different buttons.
    """
    state = _session(request)
    state.chat = []
    await asyncio.to_thread(store.save, state)
    response = _json_response({**state.to_json(), "history_length": 0})
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/chat")
async def chat(request: Request) -> Response:
    """Answer one question against whatever is already in ChromaDB.

    Retrieval runs synchronously -- the transparency panel must be fully
    populated before any answer appears, in either the "I don't know" or the
    extractive/generated case, and that ordering is the actual point being
    taught here, not an implementation detail. Only the optional generated
    answer streams, over the same job registry the ingestion pipeline uses.
    """
    state = _session(request)
    body = await _request_body(request)

    message = body.get("message", "")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="Message must not be blank.")

    try:
        top_k = int(body.get("top_k", settings.retrieval_top_k))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="top_k must be an integer.") from exc
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer (> 0).")

    try:
        min_score = float(body.get("min_score", settings.retrieval_min_score))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="min_score must be a number.") from exc
    if not (-1.0 <= min_score <= 1.0):
        raise HTTPException(status_code=400, detail="min_score must be between -1 and 1.")

    algorithm = body.get("algorithm", "similarity")
    if algorithm not in ("similarity", "mmr"):
        raise HTTPException(status_code=400, detail="algorithm must be 'similarity' or 'mmr'.")

    try:
        mmr_lambda = float(body.get("mmr_lambda", settings.retrieval_mmr_lambda))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="mmr_lambda must be a number.") from exc
    # Range-checked for the same reason min_score is, and it matters more here.
    # MMR scores lambda*sim - (1 - lambda)*redundancy. Outside [0, 1] that stops
    # meaning anything: at lambda=5 the second term becomes -(1-5) = +4 times
    # redundancy, so the ranking actively *rewards* near-duplicate chunks while
    # the panel still labels the column "MMR score". A wrong number presented
    # confidently to a room is the exact failure this codebase keeps producing.
    if not (0.0 <= mmr_lambda <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="mmr_lambda must be between 0 and 1 (0 = diversity only, 1 = relevance only).",
        )

    # _collection() raises a 503 HTTPException on its own -- validation above
    # must run before this, so a bad request body is a 400 even when Chroma
    # also happens to be down, not the other way around.
    collection = await _collection()
    embeddings = await asyncio.to_thread(build_embeddings)
    trace = await asyncio.to_thread(
        retrieve,
        collection,
        query=message,
        embeddings=embeddings,
        top_k=top_k,
        min_score=min_score,
        algorithm=algorithm,
        mmr_lambda=mmr_lambda,
        pool_multiplier=settings.retrieval_pool_multiplier,
    )

    message_id = uuid.uuid4().hex

    if not trace.answerable:
        # The most important state this endpoint can return: an honest "I
        # don't know" that names *why*, not a 500 and not a guess. The two
        # reasons are genuinely different problems for the room -- "index
        # something first" versus "loosen the threshold or rephrase" -- so
        # they get different text, not one generic message.
        reason = (
            "no documents are indexed yet"
            if trace.pool_size == 0
            else "nothing retrieved scored above the similarity threshold"
        )
        answer = {"kind": "unknown", "text": f"I don't know -- {reason}.", "citations": []}
        generation = {"available": False, "model": settings.ollama_model, "detail": "", "job_id": None}
    else:
        citations = [_citation(c) for c in trace.selected]
        status = None
        if settings.ollama_base_url:
            status = await asyncio.to_thread(probe, settings.ollama_base_url, settings.ollama_model)

        if status is not None and status.available:
            # Registered under this session's own id, which makes the
            # relationship with ingestion jobs deliberately one-way:
            #
            #   chat -> ingestion:  no effect. This route never calls
            #     claim_session, so asking a question cannot cancel a running
            #     chunk or embed.
            #   ingestion -> chat:  cancels. Every ingestion route claims the
            #     session, and claim_session cancels *every* running job under
            #     that id, this one included.
            #
            # That asymmetry is wanted, not incidental. A streaming answer's
            # citations describe the vectors that existed when retrieval ran, so
            # a re-embed or a "Reset everything" underneath it makes the rest of
            # that answer describe a collection that no longer exists. Killing
            # it is more honest than finishing it against stale retrieval.
            job = registry.create(state.session_id)
            prompt = build_prompt(
                message,
                [{"text": c.text, "metadata": c.metadata} for c in trace.selected],
            )
            base_url, model = settings.ollama_base_url, settings.ollama_model

            def run_generation() -> None:
                try:
                    for delta in stream_answer(base_url, model, prompt):
                        registry.publish(job, {"type": "token", "text": delta})
                    registry.finish(job, "done")
                except Exception as exc:  # noqa: BLE001 - delivered to the job UI
                    registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

            # stream_answer is a blocking (urllib-based) generator, same as
            # every other transport call in this file -- to_thread keeps the
            # event loop free, same reasoning as _document_text/build_embeddings
            # above.
            registry.register_task(job, asyncio.create_task(asyncio.to_thread(run_generation)))
            answer = {"kind": "generated", "text": "", "citations": citations}
            generation = {
                "available": True,
                "model": settings.ollama_model,
                "detail": "",
                "job_id": job.job_id,
            }
        else:
            detail = (
                status.detail
                if status is not None
                else "OLLAMA_BASE_URL is not set -- generation is disabled by default, offline-first."
            )
            answer = {
                "kind": "extractive",
                "text": _extractive_answer(trace.selected),
                "citations": citations,
            }
            generation = {
                "available": False,
                "model": settings.ollama_model,
                "detail": detail,
                "job_id": None,
            }

    trace_json = _trace_to_json(trace)
    state.chat.append({
        "message_id": message_id,
        "question": message,
        "answer": answer,
        "generation": generation,
        "trace": trace_json,
    })
    await asyncio.to_thread(store.save, state)

    response = _json_response({
        "message_id": message_id,
        "question": message,
        "answer": answer,
        "generation": generation,
        "trace": trace_json,
        "history_length": len(state.chat),
    })
    _with_session_cookie(response, state.session_id)
    return response
