"""Session tests.

The persistence requirement is not academic: a browser refresh mid-demo must not
send the presenter back to step 1 in front of a room.
"""

import json

import pytest

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
        # 500 sits exactly on a page's recorded start -- the boundary case
        # the bisect_right-vs-bisect_left comment in loader.py is about.
        # Neither 600 (mid-page) nor 0 (clamped to page 1 regardless of
        # which bisect is used) can tell the two apart; this one can.
        assert rehydrated.page_for_offset(500) == 2
        # The lookups above would pass just as well if page_offsets were left
        # as JSON's plain lists -- bisect only cares about the ints inside.
        # Pin the actual restoration this test is named for: each pair must
        # come back as a tuple, not merely produce correct answers by luck.
        assert all(isinstance(pair, tuple) for pair in rehydrated.page_offsets)

    def test_unknown_session_id_creates_fresh_session(self, tmp_path):
        # Well-formed (alnum) but nothing on disk under this id yet -- distinct
        # from a malformed id, which is covered separately below and now
        # raises rather than falling through to this same fresh-session path.
        state = make_store(tmp_path).get_or_create("doesnotexist123")
        assert state.unlocked_step() == 1

    def test_reload_preserves_unlock_level_and_working_data(self, tmp_path):
        """The module's headline requirement: unlocked_step() must survive a
        reload from disk, not just individual fields in isolation. pdf_path
        in particular is how steps 4 and 5 relocate the document after a
        refresh -- losing it is invisible unless something asserts on it.
        """
        first = make_store(tmp_path)
        state = first.get_or_create(None)
        # A fixed, obviously-not-"now" timestamp: if from_json ever stopped
        # reading created_at and fell back to _now_iso() instead, comparing
        # against a value taken *at test time* could coincidentally match
        # (same wall-clock second). This can't.
        state.created_at = "2020-01-01T00:00:00+00:00"
        state.upload = {"filename": "a.pdf", "page_count": 3}
        state.chunking = {"strategy": "recursive", "chunk_count": 12}
        state.embedding = {"vectors_written": 12}
        state.pdf_path = "/data/uploads/handbook.pdf"
        first.save(state)

        rehydrated = make_store(tmp_path).get_or_create(state.session_id)
        assert rehydrated.unlocked_step() == 5
        assert rehydrated.pdf_path == "/data/uploads/handbook.pdf"
        assert rehydrated.chunking == {"strategy": "recursive", "chunk_count": 12}
        assert rehydrated.embedding == {"vectors_written": 12}
        assert rehydrated.created_at == "2020-01-01T00:00:00+00:00"

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
        # A NEW store instance, not `store` again: reset() must persist the
        # cleared state to disk, not merely overwrite the in-memory copy in
        # `store._live`. Asserting through the same store would still pass
        # if reset() only cleared `_live` and left the pre-reset file on
        # disk -- a second worker or a reloaded process would then rehydrate
        # the stale document and chunks right back.
        reloaded = make_store(tmp_path).get_or_create(state.session_id)
        assert reloaded.unlocked_step() == 1
        assert reloaded.upload is None
        assert reloaded.chunking is None
        assert reloaded.chunks == []

    def test_get_or_create_recovers_from_corrupt_file(self, tmp_path):
        """A truncated session file -- a crash mid-write with no atomic
        rename to protect it, or corruption from outside this process --
        must not 500 every request against this session. Fall back to a
        fresh one instead.
        """
        store = make_store(tmp_path)
        state = store.get_or_create(None)
        store.save(state)
        store._path(state.session_id).write_text('{"session_id": "trunc')

        fresh = make_store(tmp_path).get_or_create(state.session_id)
        assert fresh.unlocked_step() == 1

    def test_get_or_create_recovers_from_unknown_schema_file(self, tmp_path):
        """app-data is a named volume that survives image rebuilds, so a
        session file written under an older Chunk schema can still be on
        disk after a later change adds/removes/renames a field. That must
        not crash get_or_create -- Task 7's per-request hot path -- either.
        """
        store = make_store(tmp_path)
        state = store.get_or_create(None)
        state.chunks = [Chunk(index=0, text="t", start=0, strategy="fixed")]
        store.save(state)
        path = store._path(state.session_id)
        data = json.loads(path.read_text())
        data["chunks"][0]["field_from_a_future_schema"] = "x"
        path.write_text(json.dumps(data))

        fresh = make_store(tmp_path).get_or_create(state.session_id)
        assert fresh.unlocked_step() == 1

    @pytest.mark.parametrize("persisted_id", ["differentvalidid", "../../invalid"])
    def test_rehydration_rejects_a_persisted_id_that_does_not_match_its_filename(
        self, tmp_path, persisted_id
    ):
        store = make_store(tmp_path)
        requested_id = "requestedvalidid"
        store._path(requested_id).write_text(
            json.dumps(
                {
                    "session_id": persisted_id,
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "upload": {"filename": "leaked.pdf"},
                }
            )
        )

        fresh = make_store(tmp_path).get_or_create(requested_id)

        assert fresh.session_id not in {requested_id, persisted_id}
        assert fresh.upload is None

    def test_malformed_session_id_is_rejected_not_swapped(self, tmp_path):
        """The .isalnum() guard in _path has to actually run: if it were
        deleted, get_or_create would silently hand back a session under a
        *different* id instead of raising -- a permanent "back to step 1"
        loop for a client stuck sending a malformed cookie, and the failure
        would be invisible because a state object still comes back.
        """
        store = make_store(tmp_path)
        with pytest.raises(ValueError):
            store.get_or_create("a/b")

    def test_reset_rejects_malformed_session_id(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(ValueError):
            store.reset("a/b")

    def test_save_rejects_malformed_session_id(self, tmp_path):
        store = make_store(tmp_path)
        state = SessionState(session_id="a/b", created_at="2020-01-01T00:00:00+00:00")
        with pytest.raises(ValueError):
            store.save(state)

    def test_failed_malformed_save_does_not_poison_the_live_cache(self, tmp_path):
        store = make_store(tmp_path)
        state = SessionState(session_id="a/b", created_at="2020-01-01T00:00:00+00:00")

        with pytest.raises(ValueError):
            store.save(state)
        with pytest.raises(ValueError):
            store.get_or_create("a/b")

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
