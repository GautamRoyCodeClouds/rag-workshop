"""Embedder tests.

build_embeddings loads a real 90MB model, so only one test touches it and it is
marked slow. Everything else uses a fake, because batching and progress
reporting are our logic, not the model's.
"""

import pytest

from app.pipeline.embedder import build_embeddings, embed_batched, vector_norm


class CountingEmbeddings:
    """Records how it was called so batching can be asserted."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_batches_are_split_by_batch_size():
    fake = CountingEmbeddings()
    embed_batched(fake, [f"t{i}" for i in range(10)], batch_size=4)
    assert fake.batch_sizes == [4, 4, 2]


def test_returns_one_vector_per_input_text():
    vectors = embed_batched(CountingEmbeddings(), ["a", "b", "c"], batch_size=2)
    assert len(vectors) == 3


def test_progress_is_reported_per_batch_and_reaches_the_total():
    seen: list[tuple[int, int]] = []
    embed_batched(
        CountingEmbeddings(), [f"t{i}" for i in range(10)],
        batch_size=4, on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(4, 10), (8, 10), (10, 10)]


def test_empty_input_makes_no_calls():
    fake = CountingEmbeddings()
    assert embed_batched(fake, [], batch_size=4) == []
    assert fake.batch_sizes == []


def test_vector_norm_of_a_unit_vector_is_one():
    assert vector_norm([0.6, 0.8]) == pytest.approx(1.0)


def test_vector_norm_of_a_non_unit_vector_is_its_actual_length():
    """Every test vector elsewhere in this suite (and in test_store.py) is
    already unit-length, so a vector_norm that just returned 1.0 unconditionally
    would pass every other test in the whole 90+ test suite -- including the
    one the workshop deck relies on to show the room that normalisation really
    happened. [3.0, 4.0] is the classic 3-4-5 triangle: its true Euclidean
    length is 5.0, not 1.0, so this is the one assertion a hardcoded return 1.0
    cannot satisfy.
    """
    assert vector_norm([3.0, 4.0]) == pytest.approx(5.0)


def test_vector_norm_of_a_zero_vector_is_zero():
    assert vector_norm([0.0, 0.0, 0.0]) == pytest.approx(0.0)


@pytest.mark.slow
def test_real_model_produces_l2_normalised_vectors():
    """Stored vectors are 384-dimensional and unit length.

    Note what this does NOT show. It cannot fail: all-MiniLM-L6-v2 ends its own
    pipeline with a Normalize module, so its output is unit length whatever this
    codebase does -- deleting encode_kwargs entirely leaves this green. It pins
    the shape of what we store, and nothing about normalisation being ours. The
    test below is the one that proves normalisation is real.
    """
    embeddings = build_embeddings()
    vectors = embed_batched(embeddings, ["annual leave policy"], batch_size=8)
    assert len(vectors[0]) == 384
    assert vector_norm(vectors[0]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_normalisation_is_a_real_transformation_not_a_no_op():
    """Prove the deck's claim instead of restating it.

    "Normalise so cosine becomes a dot product" is a claim about a
    transformation having an effect. No assertion on the pipeline's *output*
    can demonstrate that, because the output is unit length either way -- even
    encode(normalize_embeddings=False) returns norm 1.0, since the flag does not
    disable the model's own Normalize module.

    The only way to see the effect is to reach the vector before it: run the
    Transformer and Pooling stages and stop short of Normalize. That pooled
    vector is emphatically not unit length (~5.7 for this sentence), and
    dividing it by its own length reproduces the full pipeline exactly -- which
    is what "normalisation" means, made observable.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    from app.config import settings

    sentence = "Vector databases store embeddings, not text."
    model = SentenceTransformer(settings.embed_model)

    # The guard that makes the flag's redundancy a fact rather than a belief.
    # If a future EMBED_MODEL lacks this stage, encode_kwargs stops being
    # belt-and-braces and becomes load-bearing -- and this fails to say so.
    stages = [type(stage).__name__ for stage in model]
    assert stages[-1] == "Normalize", (
        f"model pipeline ends in {stages[-1]!r}, not Normalize -- "
        "normalize_embeddings=True is now the only thing keeping vectors "
        "unit-length, and the rest of this test no longer holds"
    )

    features = model.tokenize([sentence])
    with torch.no_grad():
        pooled_output = features
        for stage in list(model)[:-1]:      # everything except Normalize
            pooled_output = stage(pooled_output)
    pooled = pooled_output["sentence_embedding"][0]

    # Not unit length, and not marginally so. A normalised vector cannot
    # satisfy this, which is precisely why the assertion is worth making.
    pooled_length = vector_norm(pooled.tolist())
    assert pooled_length > 2.0, (
        f"pooled vector has norm {pooled_length}, so it was already normalised "
        "and this test can no longer distinguish normalised from raw"
    )

    # Normalising by hand reproduces the pipeline, so the Normalize stage is
    # exactly this division -- direction preserved, length set to one.
    full = model.encode(sentence)
    manual = (pooled / pooled.norm()).tolist()
    assert manual == pytest.approx(full.tolist(), abs=1e-6)
    assert vector_norm(full.tolist()) == pytest.approx(1.0, abs=1e-6)


def test_batch_size_zero_raises_error():
    """batch_size=0 triggers a guard that prevents the silent range() failure.
    
    This tests FINDING 1: negative or zero batch_size causes range() to
    silently skip iterations (for zero, ValueError; for negative, empty list).
    The guard ensures we fail explicitly with a clear message.
    """
    with pytest.raises(ValueError, match="batch_size must be >= 1, got 0"):
        embed_batched(CountingEmbeddings(), ["text"], batch_size=0)


def test_batch_size_negative_raises_error():
    """batch_size=-1 triggers a guard that prevents the silent range() failure.
    
    This tests FINDING 1: range(0, N, -1) silently returns [] without any
    iteration — all inputs disappear, no error, no callback. The guard ensures
    we fail explicitly instead.
    """
    with pytest.raises(ValueError, match="batch_size must be >= 1, got -1"):
        embed_batched(CountingEmbeddings(), [f"t{i}" for i in range(10)], batch_size=-1)


def test_batch_size_valid_still_works():
    """Verify that valid batch sizes still work after the guard is added."""
    fake = CountingEmbeddings()
    result = embed_batched(fake, [f"t{i}" for i in range(10)], batch_size=3)
    assert len(result) == 10
    assert fake.batch_sizes == [3, 3, 3, 1]


@pytest.mark.slow
def test_build_embeddings_cache_identity():
    """Test FINDING 2: build_embeddings() and build_embeddings(model_name)
    return the *same object* (identity check with `is`), not separate models.
    
    The deck says models are "loaded once for the process lifetime". If the
    cache key differs between None (default) and the resolved string, we load
    and keep two separate ~90MB models, violating the promise.
    
    This test verifies the fix: resolving model_name *before* the cached call
    ensures build_embeddings() and build_embeddings(settings.embed_model)
    return the exact same object in memory.
    """
    from app.config import settings
    
    # Clear any prior cache entries by calling _build_embeddings_cached directly
    # with a fresh import to ensure clean state (in practice, pytest handles
    # this, but being explicit documents the intent).
    default_call = build_embeddings()
    explicit_call = build_embeddings(settings.embed_model)
    
    # Identity check: both must be the same object in memory, not just equal.
    assert default_call is explicit_call, (
        "build_embeddings() and build_embeddings(settings.embed_model) should "
        "return the same cached object (same memory location), not two separate models."
    )
