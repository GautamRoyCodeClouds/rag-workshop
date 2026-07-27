"""Session tests.

The persistence requirement is not academic: a browser refresh mid-demo must not
send the presenter back to step 1 in front of a room.
"""

from app.pipeline.chunkers import Chunk
from app.session import SessionState, SessionStore


def make_store(tmp_path) -> SessionStore:
    return SessionStore(data_dir=tmp_path)


class TestUnlocking:
    def test_no_upload_locks_at_step_one(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        assert state.unlocked_step() == 1

    def test_upload_unlocks_step_two(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf", "doc_id": "x" * 64, "page_count": 3}
        assert state.unlocked_step() == 2

    def test_chunking_unlocks_embedding(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf"}
        state.chunking = {"strategy": "recursive", "chunk_count": 12}
        assert state.unlocked_step() == 4

    def test_embedding_unlocks_browse(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf"}
        state.chunking = {"strategy": "recursive", "chunk_count": 12}
        state.embedding = {"vectors_written": 12}
        assert state.unlocked_step() == 5


class TestPersistence:
    def test_save_and_reload_restores_chunks_and_upload(self, tmp_path):
        first = make_store(tmp_path)
        state = first.get_or_create(None)
        state.upload = {"filename": "handbook.pdf", "page_count": 270}
        state.chunks = [Chunk(index=0, text="hello", start=0, strategy="recursive")]
        first.save(state)

        # A new store instance stands in for a restarted process.
        rehydrated = make_store(tmp_path).get_or_create(state.session_id)
        assert rehydrated.upload["page_count"] == 270
        assert len(rehydrated.chunks) == 1
        assert rehydrated.chunks[0].text == "hello"
        assert rehydrated.chunks[0].strategy == "recursive"

    def test_page_offsets_survive_json_round_trip(self, tmp_path):
        # JSON turns tuples into lists; page_for_offset must still work.
        store = make_store(tmp_path)
        state = store.get_or_create(None)
        state.page_offsets = [(0, 1), (500, 2), (900, 3)]
        store.save(state)

        rehydrated = make_store(tmp_path).get_or_create(state.session_id)
        assert rehydrated.page_for_offset(600) == 2
        assert rehydrated.page_for_offset(0) == 1
        # The lookups above would pass just as well if page_offsets were left
        # as JSON's plain lists -- bisect only cares about the ints inside.
        # Pin the actual restoration this test is named for: each pair must
        # come back as a tuple, not merely produce correct answers by luck.
        assert all(isinstance(pair, tuple) for pair in rehydrated.page_offsets)

    def test_unknown_session_id_creates_fresh_session(self, tmp_path):
        state = make_store(tmp_path).get_or_create("does-not-exist")
        assert state.unlocked_step() == 1

    def test_reset_clears_every_stage(self, tmp_path):
        store = make_store(tmp_path)
        state = store.get_or_create(None)
        state.upload = {"filename": "a.pdf"}
        state.chunking = {"chunk_count": 5}
        state.chunks = [Chunk(index=0, text="t", start=0, strategy="fixed")]
        store.save(state)

        fresh = store.reset(state.session_id)
        assert fresh.upload is None
        assert fresh.chunking is None
        assert fresh.chunks == []
        assert store.get_or_create(state.session_id).unlocked_step() == 1

    def test_to_json_excludes_chunk_bodies(self, tmp_path):
        """The client gets counts and metadata, not 423 chunk bodies twice."""
        state = make_store(tmp_path).get_or_create(None)
        state.chunks = [Chunk(index=0, text="x" * 700, start=0, strategy="fixed")]
        assert "x" * 700 not in str(state.to_json())
        # to_json() never looks at self.chunks at all, so the line above would
        # hold just as well if chunk bodies were never included anywhere --
        # it can't tell "deliberately excluded" apart from "not implemented".
        # Pin the actual distinction the docstring claims: the full disk view
        # DOES carry the body, proving to_json is a genuinely smaller view of
        # the same state rather than a view that merely never mentions chunks.
        assert "x" * 700 in str(state.to_disk())
