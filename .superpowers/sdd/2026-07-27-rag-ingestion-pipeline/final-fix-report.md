# Tasks 6–8 final fix wave

## Outcome

All seven final-review findings are addressed in one focused change set. The
accepted no-TTL job-history decision was left unchanged.

## Findings and fixes

### Important 1 — generation-safe mutations

- Added a per-session mutation lock and atomic generation claim to
  `JobRegistry`.
- Upload, local-load, chunk, embed, and reset now claim the session before
  their first slow await. Their canonical filesystem, session-state, Chroma,
  and persistence mutations happen under the session lock after a current
  generation/job check.
- Upload bytes are written to a UUID-named staging file in the session upload
  directory. Only the current generation may atomically replace `source.pdf`;
  every path cleans up its staging file.
- Embed creates its generation-owned job before awaiting collection
  acquisition. A failed preflight terminates that job rather than leaking a
  running history entry.
- Deterministic thread/event regressions cover old upload vs reset, old upload
  vs newer upload (including canonical file bytes), old local-load vs reset,
  and embed collection preflight vs reset. Existing delayed chunk regressions
  remain green.

### Important 2 — job-output ownership

- `/api/events/{job_id}` and `/api/status/{job_id}` resolve the request session
  before job lookup and return the same 404 for an absent or foreign job.
- Cross-client tests prove neither SSE chunk text nor polling status leaks.
- Both routes retain malformed-cookie handling: 400 plus a rotated session
  cookie, before ownership lookup.
- Cursor replay, SSE/poll handoff, and terminal exact-once tests remain green.

### Important 3 — async filesystem and persistence boundaries

- Upload directory creation and staged writes run through
  `asyncio.to_thread`.
- Every `SessionStore.save`/`reset` call reachable from an async route or
  background job runs through `asyncio.to_thread`, including JSON
  serialization, temp write, and atomic replace.
- A blocking-save responsiveness regression proves the event loop continues
  to schedule while persistence is in progress.

### Important 4 — fresh-document browser state

- Added one `resetDownstreamView()` that clears chunk cards/notes/count,
  embedding progress/status, record cards/count, and paging state after every
  successful new upload/local document and after reset.
- Added a client operation generation plus active follower cancellation.
  Superseded SSE events, polling completions, errors, and terminal callbacks
  are ignored.
- Supersession also restores both action buttons, including while the initial
  chunk/embed request is still pending.
- Collection-page requests also carry the operation generation, so a response
  already in flight cannot repopulate records after a new document reset.
- Successful re-chunking performs the embedding/record tier of the centralized
  reset because the server has invalidated the prior vectors at that point.
- The real `app.js` is executed in a Node VM with a minimal DOM to verify the
  reset, stale SSE suppression, stale collection-response suppression, and
  visual/accessibility progress updates. Document content rendering remains
  `textContent`-only.

### Important 5 — complete Chroma read translation

- `/api/collection` now translates both collection acquisition and
  `read_records` failures to the existing actionable 503 contract.
- An HTTP regression forces the read itself to fail and checks the 503 detail.

### Minor 1 — persisted session identity

- Rehydration accepts a session file only when its persisted `session_id` is
  valid and exactly matches the requested filename/session ID.
- Valid-but-mismatched and malformed persisted IDs are both covered and
  recover as fresh state without the persisted upload.

### Minor 2 — progressbar semantics

- The embedding bar now exposes `role=progressbar`,
  `aria-valuemin=0`, `aria-valuemax=100`, and `aria-valuenow`.
- `aria-valuenow` advances with the same computed percentage as the visual
  width, reaches 100 on success, and resets to 0 at operation/document reset.

## RED evidence

Tests were written before production changes and exercised against
`55fe64e` in the existing project image with the live workspace mounted.

- The initial focused regression run collected the new cases and failed for
  persisted-ID trust, missing progress semantics/browser reset, unhandled
  Chroma reads, foreign job visibility, malformed-cookie ownership order,
  delayed mutation overwrite, and synchronous persistence.
- Representative persisted-ID failure:
  `expected a fresh ID; got "differentvalidid"` from the mismatched file.
- The real client behavior harness against
  `55fe64e:app/static/app.js` exited `10` because successful document load did
  not clear the old chunk view.
- The self-review stale collection-response addition exited `16` before the
  generation check was added.
- The self-review embed-preflight test returned 503 but found a generation
  owner still in `running` state before terminal cleanup was added.
- Independent review extended the client harness; it exited `19` while
  superseded controls remained disabled before cancellation cleanup was added.
- The first host `pytest` attempt was not counted as RED evidence because the
  host interpreter lacked project dependencies. The running Compose container
  also held the prior image snapshot, so verification switched to mounting
  this exact workspace into the project image.

## GREEN and verification evidence

- Focused:
  `docker run --rm -v /home/gautam/Projects/test/class-rag:/workspace -w /workspace --entrypoint python class-rag-app -m pytest tests/test_session.py tests/test_api.py -q`
  — **72 passed, 1 skipped, 23 warnings in 5.72s**.
- Complete suite, including both marked slow tests:
  `docker run --rm -v /home/gautam/Projects/test/class-rag:/workspace -w /workspace --entrypoint python class-rag-app -m pytest -ra`
  — **170 passed, 1 skipped, 226 warnings in 9.23s**.
- Explicit slow subset:
  `... python -m pytest -m slow -q`
  — **2 passed, 169 deselected in 7.65s**.
- Browser behavior harness under host Node — exit 0.
- `node --check app/static/app.js` — exit 0.
- `git diff --check` — exit 0.

The one suite skip is the client behavior harness because the production
Python app image intentionally does not install Node.js. The identical harness
was run successfully with host Node. Warnings are existing dependency noise:
three Starlette per-request-cookie deprecations and 223 Chroma/Pydantic
`model_fields` deprecations. Chroma also logs its existing PostHog telemetry
signature error in tests; it does not fail the suite.

## Files

- `app/jobs.py`
- `app/main.py`
- `app/session.py`
- `app/static/app.js`
- `app/templates/index.html`
- `tests/test_api.py`
- `tests/test_session.py`
- `.superpowers/sdd/2026-07-27-rag-ingestion-pipeline/final-fix-report.md`

No PDF or external asset was added. `graphify-out/` remains an unrelated
untracked user artifact and is excluded.

## Self-review and concerns

- Self-review found and fixed two additional lifecycle edges: failed embed
  preflight job leakage and stale collection fetches after document reset.
- Independent read-only review found and fixed two frontend lifecycle edges:
  controls disabled by supersession and stale embedding/record output after a
  successful re-chunk. It reported no critical backend issue.
- The review's minor suggestion to claim only after request validation was not
  applied to upload: reading and sizing the upload requires an await, while the
  governing requirement explicitly requires ownership before the first await.
- Mutation locks are deliberately in-process, matching the existing
  in-process session/job architecture and single-app-process workshop
  deployment. Moving to multiple app workers would require shared session,
  generation, and locking storage as one coordinated architectural change.
- The accepted no-TTL job-history behavior remains the only known deferred
  lifecycle concern.
