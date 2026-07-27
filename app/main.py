"""FastAPI routes for the five-step ingestion page."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.jobs import registry, sse_format
from app.pipeline import store as vector_store
from app.pipeline.chunkers import STRATEGIES, UnknownStrategyError, chunk
from app.pipeline.embedder import build_embeddings, embed_batched
from app.pipeline.loader import EmptyDocumentError, load_pdf
from app.pipeline.store import get_client
from app.session import store

BASE_DIR = Path(__file__).parent
SESSION_COOKIE = "rag_session"

app = FastAPI(title="RAG Ingestion Pipeline", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


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


def _collection():
    try:
        return vector_store.get_collection(get_client())
    except Exception as exc:  # noqa: BLE001 - displayed as a retryable UI error
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB is unreachable ({type(exc).__name__}). Check ChromaDB and retry.",
        ) from exc


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


def _ingest_path(state, path: Path, display_name: str) -> dict:
    try:
        result = load_pdf(path)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - invalid/encrypted PDFs are client errors
        raise HTTPException(
            status_code=400,
            detail=(f"Could not read {display_name}: {type(exc).__name__}. " "Remove password protection and retry."),
        ) from exc
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
    store.save(state)
    return state.to_json()


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> Response:
    state = _session(request)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large: {len(payload) / 1_048_576:.1f} MB exceeds the {settings.max_upload_mb} MB limit.",
        )
    uploads = Path(settings.data_dir) / "uploads" / state.session_id
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "source.pdf"
    target.write_bytes(payload)
    response = _json_response(_ingest_path(state, target, file.filename or "document.pdf"))
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/use-local")
def use_local(request: Request) -> Response:
    local = settings.local_pdf
    if local is None:
        raise HTTPException(status_code=404, detail="No local document is configured.")
    state = _session(request)
    response = _json_response(_ingest_path(state, local, local.name))
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
    job = registry.create()

    async def run() -> None:
        try:
            registry.publish(job, {"type": "stage", "message": f"Splitting with {strategy}..."})
            text = (await asyncio.to_thread(load_pdf, state.pdf_path)).text
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
            store.save(state)
            registry.publish(job, {
                "type": "summary",
                "chunk_count": len(result.chunks),
                "sections_detected": result.sections_detected,
                "fell_back": result.fell_back,
            })
            registry.finish(job, "done")
        except UnknownStrategyError as exc:
            registry.finish(job, "error", str(exc))
        except Exception as exc:  # noqa: BLE001 - delivered to the job UI
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    asyncio.create_task(run())
    response = _json_response({"job_id": job.job_id}, status_code=202)
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/embed")
async def start_embedding(request: Request) -> Response:
    state = _session(request)
    if not state.chunks:
        raise HTTPException(status_code=409, detail="Chunk the document first.")
    collection = _collection()
    job = registry.create()

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
            store.save(state)
            registry.publish(job, {"type": "summary", "vectors_written": written})
            registry.finish(job, "done")
        except Exception as exc:  # noqa: BLE001 - delivered to the job UI
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    asyncio.create_task(run())
    response = _json_response({"job_id": job.job_id}, status_code=202)
    _with_session_cookie(response, state.session_id)
    return response


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")

    async def stream():
        history, queue, terminal = registry.subscribe(job)
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
def job_status(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {"job_id": job.job_id, "status": job.status, "error": job.error, "events": list(job.events)}


@app.get("/api/collection")
def collection_records(offset: int = 0, limit: int = 25) -> dict:
    return vector_store.read_records(_collection(), offset=offset, limit=min(limit, 100))


@app.post("/api/reset")
async def reset(request: Request) -> Response:
    state = _session(request)
    body = await _request_body(request)
    if body.get("drop_collection"):
        try:
            vector_store.drop_collection(get_client())
        except Exception as exc:  # noqa: BLE001 - unavailable store must stay visible
            raise HTTPException(status_code=503, detail=f"ChromaDB collection reset failed ({type(exc).__name__}).") from exc
    fresh = store.reset(state.session_id)
    response = _json_response(fresh.to_json())
    _with_session_cookie(response, fresh.session_id)
    return response
