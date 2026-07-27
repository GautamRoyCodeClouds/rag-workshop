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


@lru_cache(maxsize=2)
def build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Load the embedding model, cached for the process lifetime.

    Weights are baked into the image under HF_HOME, so this never reaches the
    network -- the workshop is presented offline. First call costs a few seconds
    of load time; subsequent calls are free thanks to the cache (a later task
    calls this from a web request handler, where reloading per request would
    be visibly slow).
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or settings.embed_model,
        model_kwargs={"device": "cpu"},
        # normalize_embeddings is the deck's "normalise once at write time".
        encode_kwargs={"normalize_embeddings": True},
    )


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
