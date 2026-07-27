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


@pytest.mark.slow
def test_real_model_produces_l2_normalised_vectors():
    """The deck says normalise once at write time so cosine becomes a dot
    product. This is the test that proves we actually did."""
    embeddings = build_embeddings()
    vectors = embed_batched(embeddings, ["annual leave policy"], batch_size=8)
    assert len(vectors[0]) == 384
    assert vector_norm(vectors[0]) == pytest.approx(1.0, abs=1e-3)
