"""Store tests, run against an in-process ephemeral Chroma client.

No server needed, so the suite stays fast and hermetic while still exercising
the real Chroma API rather than a mock of it.
"""

import uuid

import chromadb
import pytest

from app.pipeline.chunkers import Chunk
from app.pipeline.store import (
    count_records,
    get_collection,
    read_records,
    write_chunks,
)

DOC_ID = "a" * 64
OTHER_DOC_ID = "b" * 64


@pytest.fixture
def collection():
    # chromadb 0.6.3's EphemeralClient() instances all share one process-wide
    # "ephemeral" system (SharedSystemClient hardcodes that identifier), so a
    # fixed collection name would silently reattach to whatever the previous
    # test left behind instead of starting empty. A unique name per test keeps
    # this fixture actually hermetic, matching its own docstring's promise.
    client = chromadb.EphemeralClient()
    return get_collection(client, f"test-collection-{uuid.uuid4().hex}")


def make_chunks(count: int, strategy: str = "recursive") -> list[Chunk]:
    return [
        Chunk(index=i, text=f"chunk number {i}", start=i * 100, strategy=strategy)
        for i in range(count)
    ]


def make_vectors(count: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in range(count)]


def write(collection, chunks, *, doc_id=DOC_ID, size=700, overlap=100):
    return write_chunks(
        collection,
        chunks=chunks,
        vectors=make_vectors(len(chunks)),
        doc_id=doc_id,
        source="handbook.pdf",
        size=size,
        overlap=overlap,
        embed_model="all-MiniLM-L6-v2",
        page_for_offset=lambda offset: offset // 100 + 1,
    )


class TestWrite:
    def test_writes_every_chunk(self, collection):
        assert write(collection, make_chunks(5)) == 5
        assert count_records(collection) == 5

    def test_reingesting_the_same_document_does_not_duplicate(self, collection):
        write(collection, make_chunks(5))
        write(collection, make_chunks(5))
        assert count_records(collection) == 5

    def test_shrinking_the_chunk_count_leaves_no_orphans(self, collection):
        """The bug this exists to prevent.

        Ingest at size 700 -> 5 chunks. Re-ingest at 1500 -> 2 chunks. Without
        delete-before-write, chunks 2-4 from the first run survive as orphans and
        silently pollute every later result.
        """
        write(collection, make_chunks(5), size=700)
        write(collection, make_chunks(2), size=1500)
        assert count_records(collection) == 2

    def test_a_different_strategy_coexists(self, collection):
        # Comparing two strategies side by side is intentional, so only a re-run
        # of the *same* strategy replaces anything.
        write(collection, make_chunks(3, "recursive"))
        write(collection, make_chunks(4, "fixed"))
        assert count_records(collection) == 7

    def test_a_different_document_coexists(self, collection):
        write(collection, make_chunks(3), doc_id=DOC_ID)
        write(collection, make_chunks(3), doc_id=OTHER_DOC_ID)
        assert count_records(collection) == 6

    def test_writing_no_chunks_is_a_no_op(self, collection):
        assert write(collection, []) == 0
        assert count_records(collection) == 0


class TestMetadata:
    def test_every_record_records_the_embedding_model(self, collection):
        # The deck: store the model name alongside every vector so you can tell
        # what is stale.
        write(collection, make_chunks(3))
        for meta in read_records(collection)["records"]:
            assert meta["metadata"]["embed_model"] == "all-MiniLM-L6-v2"

    def test_carries_the_full_metadata_set(self, collection):
        write(collection, make_chunks(1))
        meta = read_records(collection)["records"][0]["metadata"]
        for key in (
            "doc_id", "source", "page", "chunk_index", "strategy",
            "chunk_size", "overlap", "embed_model", "char_count", "parent_id",
        ):
            assert key in meta, f"missing metadata key {key}"

    def test_page_is_derived_from_the_offset(self, collection):
        write(collection, make_chunks(3))
        pages = sorted(r["metadata"]["page"] for r in read_records(collection)["records"])
        assert pages == [1, 2, 3]

    def test_absent_parent_id_is_an_empty_string(self, collection):
        # Chroma rejects None in metadata; "" is the sentinel.
        write(collection, make_chunks(1))
        assert read_records(collection)["records"][0]["metadata"]["parent_id"] == ""


class TestRead:
    def test_returns_a_vector_preview_and_norm(self, collection):
        write(collection, make_chunks(1))
        record = read_records(collection, preview_dims=2)["records"][0]
        assert len(record["vector_preview"]) == 2
        assert record["vector_norm"] == pytest.approx(1.0)
        assert record["dims"] == 3

    def test_paginates(self, collection):
        write(collection, make_chunks(10))
        page = read_records(collection, offset=4, limit=3)
        assert len(page["records"]) == 3
        assert page["total"] == 10

    def test_includes_the_document_text(self, collection):
        write(collection, make_chunks(1))
        assert read_records(collection)["records"][0]["text"] == "chunk number 0"

    def test_empty_collection_reads_cleanly(self, collection):
        page = read_records(collection)
        assert page["records"] == []
        assert page["total"] == 0
