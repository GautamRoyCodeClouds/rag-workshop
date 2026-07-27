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

    Ids are `f"{doc_id[:12]}-{strategy}-{c.index}"` -- a truncated document
    hash plus the chunk's position, not a hash of the chunk's own content. That
    means a same-parameters re-run reuses the *same* ids, which sounds like an
    overwrite but is not one: Chroma's `add` does not overwrite an existing id,
    it logs "Insert of existing embedding ID" and keeps the old record. Without
    the delete below, a re-run at the same size/overlap would silently retain
    stale text under those ids. And a re-run at a *different* size is worse
    without it: ingesting at size 700 yields far more chunks than at 1500, so
    the tail of the earlier run would survive as orphans and quietly pollute
    every later result. Hence the delete first, scoped to this (doc_id,
    strategy) pair so a *different* strategy can sit alongside for comparison.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(
            f"{len(chunks)} chunks but {len(vectors)} vectors -- these must match."
        )

    # chunks[0].strategy stands in for "this batch's strategy" here, while the
    # metadata below records each chunk's *own* c.strategy. Those two only ever
    # agree because chunkers.py's structure-aware fallback relabels its chunks
    # to the requested strategy before returning (see chunkers.py's
    # _chunk_structure, "strategy=structure" on the fallback branch) -- a
    # result never mixes labels within one write_chunks call. If a future
    # strategy ever returned a mixed-label batch, this delete would target the
    # wrong scope.
    strategy = chunks[0].strategy

    collection.delete(where={"$and": [{"doc_id": doc_id}, {"strategy": strategy}]})

    collection.add(
        # doc_id is a 64-char sha256 hex digest (loader.py); 12 hex characters
        # is already 48 bits of entropy, far more than this demo's collection
        # sizes need to stay collision-free, and keeps ids readable when
        # they show up in logs or the browser UI. Not used for correctness --
        # the delete above scopes on the full doc_id, not the truncated one.
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
        # embeddings as a numpy.ndarray of numpy.float64. np.float64 is
        # actually a subclass of Python float and serialises through FastAPI's
        # JSON encoder fine as-is -- so this coercion is not fixing a break we
        # have seen. It guards against a dtype chroma does not currently
        # return: np.float32 is *not* a float subclass and would not survive
        # JSON encoding. float() is cheap insurance against a future chroma
        # version (or a different distance/quantization setting) handing back
        # float32, so nothing numpy-shaped survives into the returned dict
        # regardless of dtype.
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
    """Delete the collection outright, for the UI's Reset control.

    An absent collection should read as success -- resetting something that is
    already gone is not a failure. The tempting shortcut is `except Exception:
    pass` around delete_collection, but on chroma 0.6.3 that is too broad to
    do its job: deleting an absent collection and failing to reach a dead
    server both raise a plain ValueError, so catching Exception (or even
    ValueError) cannot tell "already gone" apart from "store unreachable".
    This is the presenter's mid-talk recovery control, so silently reporting
    success while Chroma is actually down is the wrong failure mode -- it
    would hide an outage behind a green checkmark. Checking membership via
    list_collections() first avoids the ambiguity: only call delete_collection
    when the collection is confirmed present, and let a connection failure
    (which list_collections() would also raise on) propagate as itself.
    """
    target = name or settings.chroma_collection
    if target in client.list_collections():
        client.delete_collection(target)
