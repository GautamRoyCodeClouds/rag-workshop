"""Turning text into vectors -- step 2 of the pipeline.

Level 2 of the deck: an embedding model is a function, text in and a fixed-length
list of floats out, trained so that related text lands in nearby positions.

Two deck instructions are enforced here:

  - Normalise once at write time, so cosine similarity becomes a dot product
    (gotcha #02: unnormalised vectors let long documents dominate).
  - Pin the model name in config, never as a default argument (the indexing and
    querying paths must use byte-identical models).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


def _build_embeddings_cached(model_name: str) -> HuggingFaceEmbeddings:
    """Internal: load the embedding model, cached by resolved model name.

    Weights are baked into the image under HF_HOME, so this never reaches the
    network -- the workshop is presented offline. First call costs a few seconds
    of load time; subsequent calls are free thanks to the cache.

    This function is @lru_cache with maxsize=1 per model because we expect
    build_embeddings() to be called with either no argument (using the default
    from config) or with the same model name repeatedly. Caching by the
    *resolved* string (never None) ensures that build_embeddings() and
    build_embeddings(settings.embed_model) return the same object.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # normalize_embeddings is the deck's "normalise once at write time".
        encode_kwargs={"normalize_embeddings": True},
    )


# Apply cache to the internal function that takes a required string argument.
_build_embeddings_cached = lru_cache(maxsize=1)(_build_embeddings_cached)


def build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Load the embedding model, cached for the process lifetime.

    Resolves the model name *before* the cached call, so None and the
    configured default resolve to the same cache key. This ensures that
    build_embeddings() and build_embeddings(settings.embed_model) return
    the *same object*, honoring the docstring's promise of "loaded once for
    the process lifetime".
    """
    # Resolve None to the configured default before passing to the cached loader.
    # This ensures every caller sees the same cached object, not separate
    # 90MB models per (None vs resolved string) pair.
    resolved_name = model_name or settings.embed_model
    return _build_embeddings_cached(resolved_name)


def embed_batched(
    embeddings,
    texts: list[str],
    batch_size: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed texts in batches, reporting progress after each one.

    Progress here is genuine: each callback fires only after a batch has
    actually been encoded, never on a timer or interpolated. A progress bar
    that moves for any other reason would be lying to the room about what the
    machine is actually doing.
    """
    # Guard against negative or zero batch_size early: range(0, N, 0) raises ValueError,
    # but range(0, N, -1) silently returns [] — which would cause all inputs to disappear
    # with no error, no callback, nothing to signal the problem to the caller.
    # This can happen if EMBED_BATCH_SIZE env var is misconfigured: int(value)
    # accepts negative strings like "-5" without complaint.
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be >= 1, got {batch_size}"
        )

    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embeddings.embed_documents(batch))
        if on_progress is not None:
            on_progress(min(start + len(batch), total), total)
    return vectors


def vector_norm(vector: list[float]) -> float:
    """Euclidean length of a vector.

    Shown in the collection preview: seeing 1.000 next to every record is how
    the room confirms normalisation happened rather than taking it on trust.
    """
    return math.sqrt(sum(component * component for component in vector))
