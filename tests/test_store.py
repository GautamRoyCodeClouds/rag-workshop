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
    drop_collection,
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
        # Asserting only key presence lets adjacent fields swap without the
        # test noticing -- chunk_size and overlap are adjacent ints written
        # adjacently in the metadata dict, so a transposition (or any other
        # wrong value) would still pass a keys-only check. Assert every value
        # against what write()'s defaults and make_chunks(1)'s single chunk
        # actually produce.
        write(collection, make_chunks(1))
        meta = read_records(collection)["records"][0]["metadata"]
        assert meta == {
            "doc_id": DOC_ID,
            "source": "handbook.pdf",
            "page": 1,  # page_for_offset(start=0) == 0 // 100 + 1
            "chunk_index": 0,
            "strategy": "recursive",
            "chunk_size": 700,
            "overlap": 100,
            "embed_model": "all-MiniLM-L6-v2",
            "char_count": len("chunk number 0"),
            "parent_id": "",
        }

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
        # chroma returns embeddings as numpy.float64. `isinstance(v, float)`
        # would pass even without store.py's float() coercion, since
        # numpy.float64 is a float subclass -- that would make this assertion
        # vacuous. `type(v) is float` only passes once every component really
        # has been coerced to a plain Python float, so removing the coercion
        # in store.py fails this test.
        assert all(type(v) is float for v in record["vector_preview"])
        assert type(record["vector_norm"]) is float

    def test_paginates(self, collection):
        # chunk indices 0-9 are single digits, so their ids
        # (f"{doc_id[:12]}-{strategy}-{index}") sort lexicographically in the
        # same order as numerically -- chroma 0.6.3's get() pagination is
        # stable and non-overlapping on that ordering, so asserting the actual
        # slice is safe. Checking only len() and total (as this test used to)
        # cannot tell a real page 2 apart from offset being silently ignored
        # and page 1 coming back every time.
        write(collection, make_chunks(10))
        page = read_records(collection, offset=4, limit=3)
        assert len(page["records"]) == 3
        assert page["total"] == 10
        assert [r["metadata"]["chunk_index"] for r in page["records"]] == [4, 5, 6]

    def test_includes_the_document_text(self, collection):
        write(collection, make_chunks(1))
        assert read_records(collection)["records"][0]["text"] == "chunk number 0"

    def test_empty_collection_reads_cleanly(self, collection):
        page = read_records(collection)
        assert page["records"] == []
        assert page["total"] == 0


class TestDropCollection:
    def test_dropping_an_absent_collection_is_a_no_op(self):
        # An `except Exception: pass` around delete_collection would also make
        # this pass -- the point of this test is only meaningful alongside
        # test_dropping_a_present_collection_removes_it below, which the
        # broad-except version could fail in a way this one alone can't catch.
        client = chromadb.EphemeralClient()
        name = f"test-collection-{uuid.uuid4().hex}"
        drop_collection(client, name)  # must not raise
        assert name not in client.list_collections()

    def test_dropping_a_present_collection_removes_it(self):
        client = chromadb.EphemeralClient()
        name = f"test-collection-{uuid.uuid4().hex}"
        get_collection(client, name)
        assert name in client.list_collections()
        drop_collection(client, name)
        assert name not in client.list_collections()

    def test_dropping_when_the_store_is_unreachable_propagates_the_error(self):
        # This is the test that actually distinguishes the fix from the broad
        # `except Exception: pass` it replaced: on chroma 0.6.3, an absent
        # collection and an unreachable server both raise plain ValueError
        # from delete_collection, so the two no-op/removes-it tests above
        # pass under *either* implementation. A stub whose list_collections()
        # fails the way a dead connection would stands in for "store
        # unreachable" -- the fixed drop_collection must let that raise,
        # not swallow it and report a false success to the Reset control.
        class _UnreachableClient:
            def list_collections(self):
                raise ValueError("Could not connect to a Chroma server")

            def delete_collection(self, name):
                raise AssertionError(
                    "delete_collection should not be reached when "
                    "list_collections itself fails"
                )

        with pytest.raises(ValueError, match="Could not connect"):
            drop_collection(_UnreachableClient(), "whatever")
