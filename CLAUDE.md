# CLAUDE.md

Guidance for Claude Code (and any other agent) working in this repository.

## What this project is

A live teaching demo for a two-hour workshop on RAG, embeddings and vector
databases. It implements the **indexing half** of a RAG pipeline as a
single-page app where each step unlocks the next:

```
load PDF → clean → chunk (5 strategies) → embed → store in ChromaDB → inspect
```

It is a *teaching instrument*, not a product. That distinction drives most of
the rules below: the code is projected on a screen and read by attendees, so
clarity and honesty about what is really happening outrank cleverness.

Steps 1–5 (the ingestion wizard at `/`) stop at "the vectors are in the
database, here they are" — no retrieval, query, or LLM call happens there. A
separate, always-reachable `/chat` page runs the query half: `POST /api/chat`
retrieves against whatever is already in ChromaDB and optionally generates an
answer over Ollama. It does not depend on this browser's session having run
the ingestion pipeline at all — only on what some session, at some point, has
already embedded.

## Commands

```bash
# Bring the stack up (ChromaDB + app). App on :8080, Chroma on :8000.
docker compose up -d

# Tests. The `slow` marker means "loads the real ~90MB embedding model".
docker compose run --rm app pytest -q -m "not slow"
docker compose run --rm app pytest -q -m slow

# Verify the offline guarantee still holds (must succeed with no network):
docker run --rm --network none class-rag-app python -c \
  "from app.pipeline.embedder import build_embeddings; print(len(build_embeddings().embed_query('x')))"
```

**The single most important operational fact:** the image *bakes source in at
build time* — `app/`, `tests/` and `pytest.ini` are `COPY`'d, not bind-mounted.
**You must `docker compose build app` after every source change** or you are
testing stale code. This has wasted more time on this project than any other
single thing.

Current state: 172 tests collected, 171 passing, 1 skipped (it needs Node.js,
which the Python image does not have).

## Architecture

| File | Responsibility |
|---|---|
| `app/config.py` | Every tunable, each default annotated with the deck slide it came from |
| `app/pipeline/loader.py` | `PyPDFLoader` + visible cleaning; owns `page_offsets` and the shared `page_for_offset` |
| `app/pipeline/chunkers.py` | The five strategies, the `STRATEGIES` registry, and offset location |
| `app/pipeline/embedder.py` | Cached model load, `embed_batched` with genuine per-batch progress |
| `app/pipeline/store.py` | ChromaDB writes with delete-before-write, plus the record browser read |
| `app/pipeline/retriever.py` | Query-time retrieval: similarity/MMR ranking, the full `RetrievalTrace` |
| `app/pipeline/generator.py` | Optional Ollama generation: `probe`, prompt assembly, reasoning-stripped streaming |
| `app/session.py` | Per-session state, the server-side unlock rule, JSON mirror to disk |
| `app/jobs.py` | Job registry: generation-based invalidation, cursor-resumable SSE |
| `app/main.py` | FastAPI routes |
| `app/templates/`, `app/static/` | The progressive-unlock page and the chat page |

Data flow: `POST /api/upload` parses and persists → `POST /api/chunk` and
`POST /api/embed` return `202 {job_id}` and run as asyncio tasks → the client
follows `GET /api/events/{job_id}` (SSE) with `GET /api/status/{job_id}` as a
polling fallback → `GET /api/collection` paginates stored records.

Chat is a separate flow, not a sixth pipeline step: `POST /api/chat` runs
retrieval synchronously (the transparency panel must be populated before any
answer appears) and returns the full trace plus an answer in the same
response. When Ollama is configured and reachable that answer is `"kind":
"generated"` with a `job_id`, and the tokens stream over the same
`/api/events/{job_id}` machinery the ingestion pipeline uses; otherwise it is
`"extractive"` (assembled straight from the retrieved chunks) or `"unknown"`
(an honest "I don't know" — empty collection or nothing cleared the
threshold, never an error).

## Hard rules

### 1. No PDF is ever committed, ever

The workshop is demoed against a document that is private and internal to the
presenter's employer. `.gitignore` excludes `*.pdf` **absolutely, with no
allow-list**, and `.dockerignore` does the same so no document can enter a
build context.

- The presenter's document is bind-mounted **read-only** via a gitignored
  `docker-compose.override.yml` (see `docker-compose.override.yml.example`).
- It is **never `COPY`'d into an image** — that would leak it on any push.
- Tests generate synthetic PDFs at runtime with `reportlab`.
- **If you need a sample document, generate one.** Never reach for a file you
  find on the presenter's disk, and never name one in tracked files.

Remember the non-git channels too. Running the pipeline leaves the document's
text in the `app-data` and `chroma-data-v063` volumes in four places:

- `/data/uploads/<sid>/source.pdf` — the uploaded file itself
- `/data/uploads/<sid>/cleaned.txt` — the cached cleaned text (see below)
- `/data/sessions/<sid>.json` — chunk bodies, persisted for refresh recovery
- the Chroma collection — `documents` holds every chunk's text

The UI's "Reset everything" clears all four **for the current session**: it
drops the collection, clears the session state, and deletes that session's
upload directory. It does not touch *other* sessions, so before demoing on or
sharing a machine, clear the volumes outright with `docker compose down -v`.

Anything that adds a fifth copy must be deleted by `reset()` too. That is why
the cleaned-text cache lives inside `uploads/<sid>/` rather than beside the
session JSON: `reset()` already removes that whole directory, so the new file
is cleaned up by existing behaviour with no second code path to remember.
Never derive a deletion target from `state.pdf_path` — with the "Use local
document" shortcut it points at a read-only mount of a file *outside* the
volume, i.e. the presenter's own document.

### 2. Never fake progress

Progress reporting must correspond to real work.

- Embedding progress is genuinely per-batch — `embed_batched` invokes its
  callback after each batch completes.
- Chunk *rendering* streams, but splitting is atomic: the splitter returns all
  chunks at once, and the code says so where it streams them.
- **Never add an artificial delay, a simulated percentage, or a fake stage.**
  The workshop's whole thesis is that you should know what your pipeline is
  actually doing. Lying in the demo would undercut the talk.

### 3. Offline by default

The talk is presented with no network. The embedding model is baked into the
image, and `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set **after** the
bake step so no load attempts a freshness check. Without them an offline load
burns ~2 minutes on connection retries. Do not add anything that needs the
network at runtime.

### 4. Comments explain *why*, and must be true

This repo is open-sourced and read by attendees. Every non-obvious decision
carries a comment giving its reason, and config defaults name the deck slide
they came from so "why 700?" is one grep away.

A comment describing behaviour the code does not have is treated as a defect
here, on the same footing as a logic bug. Several have been caught and fixed —
including one that confidently cited a bug count that `git log` contradicted.
If you change behaviour, change the comment in the same edit.

## Documented deviations

These look like mistakes and are not. Do not "fix" them.

**Raw `chromadb` client instead of `langchain-chroma`.** `langchain-chroma`
computes embeddings internally, which would make both per-batch progress
reporting and the vector preview impossible — the two things step 4 and step 5
exist to show. LangChain still supplies the loader, the five splitters and the
embedding wrapper.

**The slide deck's code samples use LlamaIndex; this app uses LangChain.**
`rag-workshop.html` is left exactly as written. The presenter bridges the gap
verbally. Do not edit the deck to match the app.

**`unlocked_step()` never returns 3.** Steps 2 and 3 unlock together, so a
return of `2` already means both are reachable. This is deliberate and is
documented at the function.

**`ollama_base_url` backs `POST /api/chat`'s optional generation step.** When
set, and `probe()` reports the configured `ollama_model`
(`deepseek-r1:1.5b` by default) available, the chat streams a real generated
answer through the job registry. Left unset (the default) the chat still
works, falling back to an extractive answer assembled from the retrieved
chunks — the app stays offline-first with no network and no Ollama at all.
Semantic *chunking* is unrelated and still uses an *embeddings* model, not an
LLM.

## Testing standard

**Verify by mutation, not by green.** This project has a specific history: a
succession of real bugs each shipped with tests that passed *vacuously* — the
assertions held just as well when the code was broken. Examples that actually
happened here:

- A test asserting `page >= 1` passed against an implementation that always
  returned `1`.
- `vector_norm` could be replaced with `return 1.0` and the entire suite
  passed, because every fixture vector was already unit-length.
- A pagination test never checked *which* records came back, so hardcoding
  `offset=0` passed.
- `starts == sorted(starts)` held while an offset cursor was frozen, returning
  the same value 78 times.

So: after writing a test, **break the implementation deliberately, rebuild, and
confirm the test fails.** If it still passes, the test is worthless. Assert
specific values, not shapes or bounds.

Character-offset attribution is the highest-risk area in this codebase — it has
produced more bugs than anything else, and offsets become the page citations
shown to the room. `page_for_offset` lives in exactly one place
(`app/pipeline/loader.py`) so the loader and the session cannot drift apart.
Anything touching offsets deserves extra scrutiny.

## Conventions

- Python 3.12, `from __future__ import annotations`, dataclasses for state.
- Config is read from the environment **once at import**; tests build
  `Settings.from_env({...})` explicitly so they never depend on the ambient
  environment.
- Content-hash `doc_id` (sha256) makes re-ingestion idempotent; Chroma writes
  delete by `(doc_id, strategy)` first, because changing chunk size shrinks the
  chunk count and would otherwise orphan the previous run's tail.
- Chroma metadata rejects `None` — use `""` as the sentinel.
- The server is the single authority on which step is reachable. The frontend
  renders that decision; it never computes it.
