"""FastAPI application.

Later tasks add the pipeline routes. This task establishes the app object plus
a health check, so `docker compose up` can be validated before any pipeline
code exists.
"""

from __future__ import annotations

import chromadb
from fastapi import FastAPI

from app.config import settings

app = FastAPI(title="RAG Ingestion Pipeline", docs_url="/api/docs")


@app.get("/api/health")
def health() -> dict:
    """Report liveness and whether Chroma is reachable.

    A Chroma failure is reported as degraded rather than raised: the UI shows a
    retry banner, which beats a stack trace on a projector.
    """
    chroma_ok, detail = False, ""
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        chroma_ok = True
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        detail = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if chroma_ok else "degraded",
        "chroma": {"reachable": chroma_ok, "detail": detail},
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
    }
