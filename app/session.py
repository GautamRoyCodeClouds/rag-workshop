"""Per-session state and its on-disk mirror.

This is a single-presenter demo, so state lives in memory. It is also mirrored to
JSON, for one specific reason: a browser refresh mid-demo must not drop the
presenter back to step 1 in front of a room.

The step-unlock rule lives here rather than in the frontend, so the server is the
single authority on what is reachable.
"""

from __future__ import annotations

import json
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
        lookup LoadResult uses -- instead of keeping a second copy here. That
        exact piece of logic has already caused three separate bugs in this
        project; one shared implementation is the fix, not a careful second
        copy.
        """
        return _page_for_offset(self.page_offsets, offset)

    def to_json(self) -> dict:
        """The client-facing view.

        Chunk bodies are excluded: they stream over SSE during step 3, and
        shipping hundreds of them again in a status payload would be waste.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "upload": self.upload,
            "chunking": self.chunking,
            "embedding": self.embedding,
            "unlocked_step": self.unlocked_step(),
        }

    def to_disk(self) -> dict:
        """The full persisted view, chunk bodies included."""
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
        )


class SessionStore:
    """In-memory sessions with a JSON mirror on disk."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or settings.data_dir) / "sessions"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._live: dict[str, SessionState] = {}

    def _path(self, session_id: str) -> Path:
        # Session ids are generated uuid4 hex strings, never user input, so they
        # are safe as filenames. Validated anyway rather than trusted.
        if not session_id.isalnum():
            raise ValueError(f"Malformed session id: {session_id!r}")
        return self.data_dir / f"{session_id}.json"

    def get_or_create(self, session_id: str | None) -> SessionState:
        """Return the live session, rehydrate it from disk, or start a new one."""
        if session_id:
            if session_id in self._live:
                return self._live[session_id]
            try:
                path = self._path(session_id)
            except ValueError:
                path = None
            if path is not None and path.is_file():
                state = SessionState.from_json(json.loads(path.read_text()))
                self._live[state.session_id] = state
                return state

        state = SessionState(session_id=uuid.uuid4().hex, created_at=_now_iso())
        self._live[state.session_id] = state
        return state

    def save(self, state: SessionState) -> None:
        """Mirror a session to disk atomically.

        Written to a temporary file and moved into place, so a crash mid-write
        cannot leave a half-written session that fails to parse on reload.
        """
        self._live[state.session_id] = state
        target = self._path(state.session_id)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state.to_disk(), indent=2))
        temp.replace(target)

    def reset(self, session_id: str) -> SessionState:
        """Discard a session's progress, keeping the same id."""
        fresh = SessionState(session_id=session_id, created_at=_now_iso())
        self.save(fresh)
        return fresh


store = SessionStore()
