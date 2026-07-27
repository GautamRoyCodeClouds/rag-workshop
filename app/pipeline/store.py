"""Writing vectors to ChromaDB -- step 3 of the pipeline.

Level 4 of the deck: "One record, four fields" -- an id, a vector, the original
text, and metadata. That is the whole data model, and it is what this module
writes.

We use the raw chromadb client rather than langchain-chroma, deliberately.
langchain-chroma computes embeddings internally, which would rule out both
per-batch progress reporting and showing the room the actual vectors. LangChain
still supplies the loader, the splitters and the embedding wrapper. Recorded in
CLAUDE.md so the choice is not mistaken for an oversight.
"""

from __future__ import annotations

from collections.abc import Callable

import chromadb

from app.config import settings
from app.pipeline.chunkers import Chunk
from app.pipeline.embedder import vector_norm


def get_client(host: str | None = None, port: int | None = None):
    """Connect to the Chroma server defined in config."""
    return chromadb.HttpClient(
        host=host or settings.chroma_host,
        port=port or settings.chroma_port,
    )


def get_collection(client, name: str | None = None):
    """Fetch or create the workshop collection.

    hnsw:space=cosine matches the deck's defaults slide. Since vectors are
    normalised at write time, cosine and dot product agree -- but stating cosine
    explicitly documents the intent.
    """
    return client.get_or_create_collection(
        name=name or settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def write_chunks(
    collection,
    *,
    chunks: list[Chunk],
    vectors: list[list[float]],
    doc_id: str,
    source: str,
    size: int,
    overlap: int,
    embed_model: str,
    page_for_offset: Callable[[int], int],
) -> int:
    """Write chunks and their vectors, replacing any previous run.

    Content-hash ids make a same-parameters re-run a clean overwrite. That alone
    is not enough, though: ingesting at size 700 yields far more chunks than at
    1500, so the tail of the earlier run would survive as orphans and quietly
    pollute every later result. Hence the delete first, scoped to this
    (doc_id, strategy) pair so a *different* strategy can sit alongside for
    comparison.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(
            f"{len(chunks)} chunks but {len(vectors)} vectors -- these must match."
        )

    strategy = chunks[0].strategy

    collection.delete(where={"$and": [{"doc_id": doc_id}, {"strategy": strategy}]})

    collection.add(
        ids=[f"{doc_id[:12]}-{strategy}-{c.index}" for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": doc_id,
                "source": source,
                # c.start is the character offset into the *source* text, not
                # the chunk index -- page_for_offset must see the real offset
                # to attribute each chunk to its own page. Passing c.index
                # here instead would compile and pass a vacuous test (page
                # numbers would still vary), but would silently cite the
                # wrong page for any document whose chunk order and page
                # order diverge, e.g. the parent-document strategy.
                "page": page_for_offset(c.start),
                "chunk_index": c.index,
                "strategy": c.strategy,
                "chunk_size": size,
                "overlap": overlap,
                # The deck: store the model name alongside every vector so you
                # can tell what is stale when you change models.
                "embed_model": embed_model,
                "char_count": len(c.text),
                # "" not None -- Chroma metadata rejects null values.
                "parent_id": c.parent_id,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def count_records(collection) -> int:
    return collection.count()


def read_records(
    collection,
    offset: int = 0,
    limit: int = 25,
    preview_dims: int = 8,
) -> dict:
    """Read a page of stored records for the collection browser.

    Returns a vector preview and its norm rather than 384 floats: the point is
    for the room to see that a record really is numbers, and that the norm is
    1.0, without a wall of digits.
    """
    total = collection.count()
    if total == 0:
        return {"records": [], "total": 0, "offset": offset, "limit": limit}

    result = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=limit,
        offset=offset,
    )

    records = []
    for position, record_id in enumerate(result["ids"]):
        # Chroma (0.6.3, verified against the installed image) returns
        # embeddings as a numpy.ndarray of numpy.float64, not plain Python
        # floats or a list of lists. Task 7 puts this dict straight through
        # FastAPI's JSON encoder, which chokes on numpy scalars -- and
        # round(np.float64, 4) itself returns another np.float64, so rounding
        # alone does not fix it. Coerce every component to float() explicitly,
        # both here and inside vector_norm's input, so nothing numpy survives
        # into the returned dict.
        vector = [float(component) for component in result["embeddings"][position]]
        records.append({
            "id": record_id,
            "text": result["documents"][position],
            "metadata": result["metadatas"][position],
            # len(vector) reflects the actual returned embedding, not the
            # configured embed_dims -- if a future Chroma version ever
            # truncated or omitted embeddings under some include combination,
            # this would surface that as a wrong dims count rather than
            # silently reporting the configured default to a room being
            # taught what a 384-dim vector is.
            "dims": len(vector),
            "vector_preview": [round(v, 4) for v in vector[:preview_dims]],
            "vector_norm": round(vector_norm(vector), 4),
        })

    return {"records": records, "total": total, "offset": offset, "limit": limit}


def drop_collection(client, name: str | None = None) -> None:
    """Delete the collection outright, for the UI's Reset control."""
    try:
        client.delete_collection(name or settings.chroma_collection)
    except Exception:  # noqa: BLE001 - already absent is success for a reset
        pass
