"""Per-session state and its on-disk mirror.

This is a single-presenter demo, so state lives in memory. It is also mirrored to
JSON, for one specific reason: a browser refresh mid-demo must not drop the
presenter back to step 1 in front of a room.

The step-unlock rule lives here rather than in the frontend, so the server is the
single authority on what is reachable.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.pipeline.chunkers import Chunk
from app.pipeline.loader import page_for_offset as _page_for_offset


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionState:
    session_id: str
    created_at: str
    upload: dict | None = None
    chunking: dict | None = None
    embedding: dict | None = None

    # Working data, persisted so a refresh can resume mid-pipeline.
    chunks: list[Chunk] = field(default_factory=list)
    pdf_path: str = ""
    page_offsets: list[tuple[int, int]] = field(default_factory=list)

    # Chat is a separate, always-reachable feature (see app/main.py's /chat
    # route) that never touches the ingestion pipeline's own state. Each
    # entry is a plain dict -- message_id, question, answer, and the full
    # RetrievalTrace as JSON -- not a dataclass, since it is never read back
    # into typed objects, only replayed to the client as-is.
    chat: list[dict] = field(default_factory=list)

    def unlocked_step(self) -> int:
        """The highest step the user may interact with.

        1 upload, 2 configure, 3 chunk, 4 embed, 5 browse. Steps 2 and 3 unlock
        together -- once a document is loaded, choosing a strategy and running
        it happen on the same screen -- so this deliberately never returns 3:
        a return of 2 already means both step 2 and step 3 are reachable.
        Task 7 and the frontend must read it that way, not treat the missing
        3 as a bug to "fix".
        """
        if self.embedding:
            return 5
        if self.chunking:
            return 4
        if self.upload:
            return 2
        return 1

    def page_for_offset(self, offset: int) -> int:
        """Page number for a character offset.

        Delegates to app.pipeline.loader.page_for_offset -- the same bisect
        lookup LoadResult uses -- instead of keeping a second copy here.
        Character-offset attribution has repeatedly produced bugs in this
        codebase; one shared implementation is what keeps this side and the
        loader's from drifting apart, not a careful second copy.
        """
        return _page_for_offset(self.page_offsets, offset)

    def to_json(self) -> dict:
        """The client-facing view.

        Chunk bodies are excluded: they stream over SSE during step 3, and
        shipping hundreds of them again in a status payload would be waste.

        Chat entries are included so /chat can re-render past messages after a
        refresh, but each entry's "trace" key is stripped first: the client
        already received that full RetrievalTrace in the POST /api/chat
        response body that created the entry, so repeating it in every later
        page load / status response (this method is called from several
        routes, not just /chat) would be the same kind of waste chunk bodies
        already avoid -- a trace carries every pool candidate, which can be
        many times bigger than the message it belongs to.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "upload": self.upload,
            "chunking": self.chunking,
            "embedding": self.embedding,
            "unlocked_step": self.unlocked_step(),
            "chat": [
                {k: v for k, v in entry.items() if k != "trace"}
                for entry in self.chat
            ],
        }

    def to_disk(self) -> dict:
        """The full persisted view, chunk bodies and chat traces included."""
        return {
            **self.to_json(),
            "pdf_path": self.pdf_path,
            "page_offsets": [list(pair) for pair in self.page_offsets],
            "chunks": [
                {
                    "index": c.index,
                    "text": c.text,
                    "start": c.start,
                    "strategy": c.strategy,
                    "parent_id": c.parent_id,
                    "parent_text": c.parent_text,
                }
                for c in self.chunks
            ],
            # Overrides the trimmed "chat" from to_json() above: the on-disk
            # copy keeps each entry's trace so a restarted process (or a
            # future re-hydration route) has it, even though today's /chat
            # route never reads it back out.
            "chat": self.chat,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", _now_iso()),
            upload=data.get("upload"),
            chunking=data.get("chunking"),
            embedding=data.get("embedding"),
            pdf_path=data.get("pdf_path", ""),
            # JSON has no tuples; restore them so bisect comparisons behave.
            page_offsets=[tuple(pair) for pair in data.get("page_offsets", [])],
            chunks=[Chunk(**item) for item in data.get("chunks", [])],
            # Plain dicts, no reconstruction needed -- unlike Chunk above,
            # nothing here is ever read back into a dataclass.
            chat=list(data.get("chat", [])),
        )


class SessionStore:
    """In-memory sessions with a JSON mirror on disk."""

    def __init__(self, data_dir: Path | None = None):
        self.root = Path(data_dir or settings.data_dir)
        self.data_dir = self.root / "sessions"
        self._live: dict[str, SessionState] = {}
        # Deliberately no mkdir here. The module-level `store` below is built at
        # import time, so creating directories in this constructor makes merely
        # importing app.session a filesystem write against settings.data_dir
        # (/data by default). Inside the container that is a writable volume; on
        # any machine where /data does not exist and cannot be created -- a CI
        # runner, a bare checkout -- the import raises PermissionError and every
        # module that transitively imports this one fails to collect. That is
        # exactly what broke the first real CI run.
        #
        # save() creates the directory on first write instead. get_or_create
        # needs nothing: Path.is_file() is False for a path under a missing
        # directory, which is already the "no session on disk" case.

    def _path(self, session_id: str) -> Path:
        # Session ids arrive from a client-supplied cookie or header (Task 7
        # wires that up), not just the uuid4 hex this module mints -- so they
        # are validated, not trusted, before use as a filename.
        #
        # Policy: a malformed id (fails .isalnum(), which also rules out
        # "..", "/", "\0", etc.) is rejected by *raising* here, and every
        # caller -- get_or_create, save, reset -- lets that ValueError
        # propagate rather than catching it. The alternative (silently
        # substituting a fresh id) was tried and rejected: it turns a client
        # stuck sending a bad cookie into a session that resets on every
        # single request, with nothing in the response to say why. Task 7 is
        # expected to catch this ValueError at the API boundary and turn it
        # into a clean 400 plus a newly issued session -- not swallow it here.
        if not session_id.isalnum():
            raise ValueError(f"Malformed session id: {session_id!r}")
        return self.data_dir / f"{session_id}.json"

    def uploads_dir(self, session_id: str) -> Path:
        """Where this session's uploaded document is kept.

        Derived from the session id and nothing else -- deliberately not from
        state.pdf_path. With the presenter's "Use local document" shortcut,
        pdf_path points at a read-only bind mount of a file *outside* this
        volume: the presenter's own document on their own filesystem. Anything
        that deletes based on pdf_path would reach out of the volume and at
        that file. Going through _path first applies the same id guard used for
        session files, so a crafted id cannot escape the uploads directory.
        """
        self._path(session_id)
        return self.root / "uploads" / session_id

    def text_cache_path(self, session_id: str) -> Path:
        """Where the document's cleaned text is cached for re-chunking.

        Deliberately inside uploads_dir rather than beside the session JSON.
        Two reasons, and the first is the important one:

        - Cached text is a copy of the document. Keeping it in the directory
          reset() already deletes means it cannot outlive the PDF it came from,
          with no second cleanup path to remember. When the source document is
          confidential, an extra copy that nothing clears is the bug.
        - It inherits uploads_dir's session-id guard, so a crafted id cannot
          read or write outside the volume.

        Not kept on SessionState: that object is serialised to JSON on every
        stage transition, so a few hundred KB of text would be rewritten on
        each save, and lost on restart anyway.
        """
        return self.uploads_dir(session_id) / "cleaned.txt"

    def get_or_create(self, session_id: str | None) -> SessionState:
        """Return the live session, rehydrate it from disk, or start a new one."""
        if session_id:
            if session_id in self._live:
                return self._live[session_id]
            # No try/except around _path: a malformed id must raise here (see
            # _path's comment for the policy), not be treated as "no session
            # on disk" and quietly handed a brand-new id.
            path = self._path(session_id)
            if path.is_file():
                try:
                    state = SessionState.from_json(json.loads(path.read_text()))
                    if state.session_id != session_id or not state.session_id.isalnum():
                        raise ValueError("Persisted session id does not match its filename.")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # app-data is a named volume that survives image rebuilds,
                    # so a session file written under an older schema (a
                    # renamed field, Chunk gaining/losing one) can still be on
                    # disk after a later deploy changes the shape. Falling
                    # back to a fresh session turns that into "presenter
                    # starts over", not a 500 on every request until someone
                    # manually clears the volume mid-workshop.
                    pass
                else:
                    self._live[session_id] = state
                    return state

        state = SessionState(session_id=uuid.uuid4().hex, created_at=_now_iso())
        self._live[state.session_id] = state
        return state

    def save(self, state: SessionState) -> None:
        """Mirror a session to disk atomically.

        Written to a temp file and moved into place with `replace`, so a
        crash mid-write cannot leave a half-written session that fails to
        parse on reload. The temp name includes a fresh uuid, not just the
        session id: FastAPI runs sync endpoints in a threadpool, so an
        SSE-driven save and a status-poll save of the *same* session can race,
        and two saves sharing one `<sid>.json.tmp` path could `replace` torn
        content into place. This guards against a crash, not a power loss --
        there is no fsync before `replace`.
        """
        target = self._path(state.session_id)
        # Created here rather than in __init__ so importing this module never
        # writes to the filesystem -- see the constructor's comment.
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._live[state.session_id] = state
        temp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(state.to_disk(), indent=2))
        temp.replace(target)

    def reset(self, session_id: str) -> SessionState:
        """Discard a session's progress, keeping the same id.

        Goes through save(), so the cleared state is persisted, not just
        reassigned in `self._live` -- a second worker or a reloaded process
        must see the reset too, or it silently rehydrates the pre-reset file
        from the app-data volume on the next request.

        The uploaded document goes with it. Clearing the state alone would
        leave the PDF itself under uploads/<session_id>/, and app-data is a
        named volume that survives image rebuilds -- so the file would outlive
        every later reset and restart until someone cleared the volume by hand.
        When the source document is confidential, "Reset everything" quietly
        keeping a copy is the wrong failure mode.
        """
        fresh = SessionState(session_id=session_id, created_at=_now_iso())
        # save() first: it validates the id via _path, so a malformed one
        # raises before anything is removed from disk.
        self.save(fresh)
        shutil.rmtree(self.uploads_dir(session_id), ignore_errors=True)
        return fresh


store = SessionStore()
