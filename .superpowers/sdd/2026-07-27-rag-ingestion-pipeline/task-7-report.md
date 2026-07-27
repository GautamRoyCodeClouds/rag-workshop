# Task 7 report — API routes and SSE job plumbing

## Implementation

- Added `app.jobs.Job`, `JobRegistry`, and `sse_format`.
  - Registry mutations and history snapshots are guarded by a `threading.RLock`.
  - A job captures its owning event loop when created. Publications from worker
    threads use `loop.call_soon_threadsafe`; `asyncio.Queue` is not treated as
    thread-safe.
  - SSE clients receive an atomic history snapshot plus a private live queue.
    The subscribe hand-off occurs under the same lock as publish, so a late
    client receives every event exactly once rather than replaying entries from
    the polling queue.
- Replaced `app.main` with the page, config, health, upload/local-load, chunk,
  embed, SSE, polling, collection, and reset routes.
  - Both chunk and embed routes are async and use `asyncio.create_task` only
    from their active request loop.
  - CPU-bound PDF loading, splitting, embedding-model loading, embedding, and
    vector writes use `asyncio.to_thread`.
  - Chunk split results are emitted only after the atomic splitter result is
    returned; the UI rendering streams individual chunk events without an
    artificial delay.
  - Upload, local load, jobs, and reset issue the session cookie. A malformed
    client session cookie gets a clear 400 response and a replacement cookie.
  - Requested collection drops now turn connection failures into 503 rather
    than claiming that reset succeeded.
- Added the Task 8 hand-off placeholders at `app/templates/index.html` and
  `app/static/.gitkeep`.

## Tests and RED/GREEN evidence

Tests in `tests/test_api.py` cover route behavior, uploads, server unlock
state, background chunk/embed jobs, collection/reset behavior, SSE formatting,
polling history, cross-thread publication, exact-once late-SSE replay,
malformed-session recovery, and local-load cookie persistence.

RED evidence (Docker Python 3.12):

```text
docker compose run --rm app pytest tests/test_api.py -v
ModuleNotFoundError: No module named 'app.jobs'
```

This was observed after adding the API tests and before adding `app/jobs.py` or
the replacement application routes. The new local-load-cookie regression was
also run with the cookie issuance deliberately removed and failed with:

```text
KeyError: 'set-cookie'
```

GREEN verification (Docker Python 3.12):

```text
docker compose run --rm app pytest tests/test_api.py -v
28 passed, 14 warnings

docker compose run --rm app pytest -m 'not slow' -v
140 passed, 2 deselected, 217 warnings
```

The warnings are known dependency noise: Chroma/Pydantic emits the
`model_fields` deprecation warning (213 full-suite occurrences), and Starlette
warns that per-request TestClient cookies are deprecated (one test warning).
No application test failed.

## Files

- `app/jobs.py`
- `app/main.py`
- `app/templates/index.html`
- `app/static/.gitkeep`
- `tests/test_api.py`
- `.superpowers/sdd/2026-07-27-rag-ingestion-pipeline/task-7-report.md`

## Self-review

- Confirmed every required route is registered and uses server-owned session
  state for unlocks.
- Confirmed `start_embedding` is async and its integration test observes a
  completed background task and per-batch progress event.
- Confirmed the late-subscriber event list contains the historical stage and
  terminal event once each, with no queued replay duplicate.
- Confirmed reset reports a forced collection-drop exception as 503.
- Confirmed no PDFs, generated graph output, or Task 8 CSS/JS were added.

## Concerns

Jobs are in-process by design, so a server restart abandons in-flight jobs;
session state remains persisted and the client can restart the affected step.
This is the intentionally small scope selected for the one-document workshop
demo, not a distributed task queue.

---

## Fix round 1/5 — resumability, ownership, and async boundaries

### Findings fixed

- **SSE/poll cursor contract:** every retained event receives a monotonically
  increasing `id`. SSE frames include a standard `id:` line when an id exists,
  while `sse_format({"type": "done"})` remains the original data-only frame.
  `Last-Event-ID` (or the `after` query cursor) replays only unseen SSE events;
  `/api/status/{job_id}?after=N` returns only later events and the current
  `cursor`.
- **Atomic polling snapshots:** `JobRegistry.snapshot()` reads terminal status,
  error, cursor, completion session view, and filtered events under one lock.
  `finish()` adds its terminal event inside that same lock, so a terminal
  snapshot always contains the terminal event.
- **Stale job ownership:** jobs carry a per-session in-process generation.
  Upload, local load, chunk start, embed start, and reset invalidate the old
  generation and cancel retained job tasks. An ownership check immediately
  before session mutation prevents a cancelled `to_thread` worker from
  resurrecting reset or newer-document state after its thread returns.
- **Task lifecycle:** registry stores each created task, consumes exceptions,
  cancels invalidated jobs, and FastAPI's lifespan drains active jobs on
  shutdown. The application uses a lifespan handler rather than deprecated
  `on_event` hooks.
- **Event-loop boundaries:** PDF parsing, Chroma collection lookup, record
  reading, requested collection drop, and existing embedding/vector work all
  run through `asyncio.to_thread`.
- **Server authority after completion:** chunk and embed summary events include
  `unlocked_step`; terminal job snapshots return the saved server session view.

### Regression tests

`tests/test_api.py` now proves:

- live subscription/reconnect replays only ids after the supplied cursor;
- switching from a partially consumed SSE stream to polling does not duplicate
  an event;
- a terminal snapshot includes its terminal event;
- a delayed chunk worker cannot restore state after reset or overwrite a newer
  upload;
- PDF parsing and collection lookup leave the event loop responsive while the
  worker is blocked;
- local-load cookie persistence is exercised by a subsequent cookie-backed
  request;
- embedding emits two genuine progress callbacks and returns completion unlock
  state.

### RED/GREEN evidence

After writing the new regression tests, they were run against the unmodified
`c60c367` production implementation in Docker Python 3.12:

```text
docker compose run --rm app pytest tests/test_api.py -v
11 failed, 25 passed
```

The failures directly showed missing completion `session`, SSE `id:` frames and
cursor filtering, `subscribe(after=...)`/`snapshot`, stale-session resurrection
after reset/new upload, and synchronous `_ingest_path`/`_collection` behavior.

After the implementation:

```text
docker compose run --rm app pytest tests/test_api.py -v
36 passed, 16 warnings

docker compose run --rm app pytest -m 'not slow' -v
148 passed, 2 deselected, 219 warnings
```

Warnings remain dependency/test-client noise: 218 Chroma/Pydantic
`model_fields` deprecations across the full suite and one Starlette
per-request-cookie deprecation. No application failures remain.

### Minor concern

Completed jobs intentionally retain history for reconnect/polling and are not
evicted yet. This remains appropriate for the single-document workshop process,
but a long-lived multi-user deployment would need TTL/size-based job eviction.
The required `Job.queue` remains available for the public interface; production
SSE uses per-subscriber queues so it does not duplicate replay frames.
