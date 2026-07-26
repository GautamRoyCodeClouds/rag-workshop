# RAG Ingestion Pipeline — Design

**Date:** 2026-07-27
**Status:** Approved
**Context:** Live demo app for the *RAG, Embeddings & Vector Databases* 2-hour workshop
(`rag-workshop.html`, 52 slides, 6 levels).

## Purpose

A single-page web app that walks an audience through document ingestion one step at a time:
PDF in, vectors in ChromaDB out. Each step unlocks only when the previous one completes, so the
presenter can pause, explain, and take questions between stages without the UI running ahead.

The app is a live counterpart to the deck. Every default it ships with is a number from a slide,
and every chunking strategy it offers is a row from the Level 3 table. Attendees clone the repo
afterwards, so readability is a requirement, not a nicety.

**Scope ends at the ChromaDB preview.** Retrieval, reranking, prompt assembly, and generation
(deck Levels 5–6) are a separate build. This spec leaves documented seams for them and no dead code.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | LangChain | Explicit requirement. The deck's code slides use LlamaIndex; the presenter bridges that verbally. Deck is not modified. |
| Embedding model | `all-MiniLM-L6-v2`, 384 dims | The self-host row of the deck's Level 2 model table. Runs on CPU, no API key. |
| Model delivery | Baked into the Docker image at build time | Presentation is assumed offline. Nothing downloads while a room watches. |
| Backend | FastAPI + asyncio background tasks + SSE | Two containers, no broker. Keeps the LangChain pipeline the most prominent code in the repo. |
| Frontend | Jinja2 template + vanilla JS | No build step. Matches the deck's visual language directly. |
| Sample document | `internal-document.pdf` (in repo) | Real, messy, and recognisable to the audience. |
| Vector store | ChromaDB in Docker, cosine, normalised vectors | Deck defaults slide. |
| Generation model | None in this build | Nothing in steps 1–9 needs an LLM. `deepseek-r1:1.5b` via Ollama is the documented choice for the future query build. |

### Rejected alternatives

- **Celery/RQ + Redis worker.** Four containers and a broker to fail live, for a one-document
  corpus with no throughput problem. It would also put task serialization between a reader and
  the pipeline code they came to read.
- **Streamlit.** Cannot wear the deck's design, fights progressive unlock with its rerun model,
  and tangles pipeline logic with UI logic in one file.
- **LlamaIndex to match the deck verbatim.** Contradicts the stated LangChain requirement.

### A note on semantic chunking

Semantic chunking uses an **embeddings** model, not a generative one. LangChain's `SemanticChunker`
embeds each sentence, computes cosine distance between consecutive sentences, and cuts where the
distance exceeds a percentile threshold. It reuses the same MiniLM instance as the embedding step.
No LLM is involved at any point in this build.

## Architecture

```
Browser ──HTTP + SSE──▶ app (FastAPI :8080) ──HTTP──▶ chromadb (:8000)
                          │                              │
                    /data/sessions/*.json          chroma-data volume
                    /data/uploads/*.pdf
                    /opt/hf  (MiniLM weights, baked in)
```

Two containers. `app` declares `depends_on: chromadb: condition: service_healthy`, so the
existing healthcheck gates startup.

### Repository layout

```
class-rag/
├── CLAUDE.md
├── README.md
├── docker-compose.yml          # extended, not replaced
├── .env.example
├── .dockerignore
├── rag-workshop.html           # the deck, unmodified
├── internal-document.pdf            # bundled sample; Dockerfile COPYs it to /app/samples/
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI routes, SSE endpoints
│   ├── config.py               # defaults, each citing its deck slide
│   ├── session.py              # state + JSON persistence for refresh-safety
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py           # PDF → text, boilerplate cleaning
│   │   ├── chunkers.py         # the five strategies
│   │   ├── embedder.py         # MiniLM wrapper, normalised output
│   │   └── store.py            # Chroma writes, metadata assembly
│   ├── static/{app.css, app.js}
│   └── templates/index.html
├── docs/superpowers/specs/
└── tests/
    ├── conftest.py
    ├── test_loader.py
    ├── test_chunkers.py
    └── test_store.py
```

Each module has one job and can be read on its own. `chunkers.py` is the file most attendees will
open first; it carries the highest comment density and every strategy's docstring quotes the deck's
verdict for that strategy.

## Component design

### `pipeline/loader.py`

`load_pdf(path) -> LoadResult`

Extracts text with LangChain's `PyPDFLoader`, then cleans it. Cleaning is a visible feature, not an
implementation detail — it demonstrates the deck's Level 2 gotcha #04, *"Embedding raw junk"*.

Cleaning operations, each returning a count for the UI:

- Strip U+200B zero-width spaces (the Google Docs export contains them on nearly every heading)
- Drop table-of-contents lines (dotted-leader / trailing-page-number pattern in the leading region)
- Drop repeated page headers and footers (lines recurring across a majority of pages)
- Normalise whitespace runs

`LoadResult` carries: `text`, `page_count`, `char_count`, `pages_without_text`,
`boilerplate_lines_removed`, `invisible_chars_removed`, `doc_id`, `page_offsets`.

`doc_id` is `sha256(cleaned_text)`, following the deck's Level 6 ingest slide — unchanged documents
produce the same id, making re-runs an idempotent overwrite rather than a duplicate insert.

**Page attribution.** Cleaning concatenates pages into one string, so a chunk that straddles a page
boundary has no single page number. `page_offsets` is a list of `(start_char, page_number)` pairs
built during extraction; a chunk is attributed to the page containing its **start** offset via
bisect. This keeps chunking free to cross page boundaries — which recursive and semantic strategies
must do — while still giving every chunk a citable page.

**Measured baseline for the bundled PDF:** 270 pages, 254,397 chars, 30,948 words, 1 page with no
text layer, 9 thin pages, ~1,834 sentences.

### `pipeline/chunkers.py`

One interface, five implementations:

```python
def chunk(text: str, size: int, overlap: int) -> list[Chunk]
```

| Strategy | Implementation | Deck verdict it demonstrates |
|---|---|---|
| Fixed size | `CharacterTextSplitter(separator="")` | "Baseline only. Splits mid sentence and mid word." |
| Recursive *(default)* | `RecursiveCharacterTextSplitter` with `["\n\n", "\n", ". ", " ", ""]` | "The right default. Respects natural boundaries." |
| Structure-aware | Regex split on this corpus's real heading markers (`❖`, `●`, numbered `1.2`, lettered `a)` / `i)` — 223 instances), then recursive within each section | "Best value when documents have real structure." |
| Semantic | `SemanticChunker` (langchain-experimental) with the local MiniLM embeddings | "Slow and costs embeddings up front. Sometimes worth it." |
| Parent document | 300-char children, 1500-char parents, parents held in a JSON docstore | "Best of both. Precise search, full context." |

Defaults: **size 700, overlap 100** — the deck's "Sensible defaults for version one" slide.

**How the sliders map per strategy.** Not every strategy consumes `size` and `overlap` the same
way, and the UI must not imply otherwise. The controls relabel themselves per selected card:

| Strategy | `size` | `overlap` | Extra control |
|---|---|---|---|
| Fixed size | chunk length in chars | overlap in chars | — |
| Recursive | target max chars | overlap in chars | — |
| Structure-aware | max chars *within* a section | overlap in chars | — |
| Semantic | **ignored** (breakpoints are data-driven) | **ignored** | breakpoint percentile, default 95 |
| Parent document | **child** chunk size | child overlap | parent size = 5 × child (grey, derived) |

For semantic, the size/overlap sliders are disabled with an inline note explaining that cut points
come from embedding distance, not a character budget — itself a teaching point.

**Parent-document caveat.** This is fundamentally a *retrieval* pattern. Within this scope the app
builds and displays the child→parent structure but cannot demonstrate the payoff (small chunk
matches, large chunk returned). The UI labels the card accordingly rather than implying capability
it does not yet have.

### `pipeline/embedder.py`

Wraps `HuggingFaceEmbeddings` around `all-MiniLM-L6-v2` loaded from the baked-in `/opt/hf` cache.
Vectors are L2-normalised at write time, so cosine similarity reduces to a dot product — the deck's
Level 2 instruction, and the reason the preview can display a norm of 1.0.

Batches of 64, emitting progress after each batch.

### `pipeline/store.py`

Writes to Chroma via `langchain-chroma`. Metadata per chunk, following the Level 6 ingest slide:

| Field | Type | Notes |
|---|---|---|
| `doc_id` | str | sha256 of cleaned source text; idempotency key |
| `source` | str | Original filename |
| `page` | int | 1-indexed source page |
| `chunk_index` | int | Ordinal within the document |
| `strategy` | str | One of the five strategy keys |
| `chunk_size` | int | Requested size |
| `overlap` | int | Requested overlap |
| `embed_model` | str | `all-MiniLM-L6-v2` — the deck instructs storing this alongside every vector "so you can tell what is stale" |
| `char_count` | int | Actual chunk length |
| `parent_id` | str \| null | Present only for the parent-document strategy |

Chunk ids are `f"{doc_id[:12]}-{strategy}-{chunk_index}"`, making re-ingestion of an unchanged
document an overwrite.

**Delete-before-write.** Ids alone are not sufficient for idempotency across *parameter* changes:
ingesting at size 700 yields 423 chunks, and re-ingesting the same document at size 1500 yields
roughly 200 — leaving chunks 200–422 from the first run orphaned in the collection, invisibly
polluting every later result. So each write first deletes all records matching
`{doc_id, strategy}`, then inserts. This is the behaviour `test_store.py` pins.

Because the presenter will demo several strategies against the same PDF in sequence, records are
keyed by strategy as well as document — comparing two strategies side by side in the collection
browser is intentional, and only a re-run of the *same* strategy replaces anything.

### `config.py` and `.env.example`

All tunables in one place, read from the environment with the deck's values as defaults. Every
entry carries a comment naming the slide it comes from, so a reader can trace a number back to
the teaching material.

| Variable | Default | Source |
|---|---|---|
| `CHROMA_HOST` | `chromadb` | Compose service name |
| `CHROMA_PORT` | `8000` | — |
| `CHROMA_COLLECTION` | `workshop` | — |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Level 2 model table, self-host row |
| `EMBED_DIMS` | `384` | Same |
| `EMBED_BATCH_SIZE` | `64` | — |
| `DEFAULT_CHUNK_SIZE` | `700` | Level 6 defaults slide |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Level 6 defaults slide |
| `DEFAULT_STRATEGY` | `recursive` | Level 3, "the right default" |
| `SEMANTIC_PERCENTILE` | `95` | LangChain default |
| `MAX_UPLOAD_MB` | `30` | Bundled sample is 14.1 MB |
| `SAMPLE_PDF_PATH` | `/app/samples/internal-document.pdf` | Bundled document |
| `HF_HOME` | `/opt/hf` | Baked-in model cache |
| `OLLAMA_BASE_URL` | *(unset)* | Reserved seam for the future query build; unused in this scope |

`OLLAMA_BASE_URL` is documented but read by nothing — it exists so the next build has an obvious
place to plug in, and CLAUDE.md says as much to prevent it looking like an oversight.

### `session.py`

Holds per-session state and mirrors each completed stage to `/data/sessions/{id}.json`, so a
mid-demo browser refresh rehydrates instead of resetting. Persisted shape:

```json
{
  "session_id": "s-0001",
  "created_at": "2026-07-27T10:30:00Z",
  "upload":    { "filename": "sample.pdf", "doc_id": "ab12cd34...", "page_count": 270 },
  "chunking":  { "strategy": "recursive", "size": 700, "overlap": 100, "chunk_count": 423 },
  "embedding": { "model": "all-MiniLM-L6-v2", "dims": 384, "vectors_written": 423 }
}
```

Timestamps are ISO 8601 UTC. Values above are synthetic examples.

## Data flow

| Step | Endpoint | Behaviour | Unlocks |
|---|---|---|---|
| 1. Upload | `POST /api/upload` (multipart) or `POST /api/use-sample` | Save, extract, clean. Returns page count, char count, and cleaning counts. | 2 |
| 2. Configure | client-side | Five strategy cards with deck verdicts; size/overlap sliders prefilled 700/100. | 3 |
| 3. Chunk | `POST /api/chunk` → `job_id`; `GET /api/events/{job_id}` (SSE) | asyncio task streams each chunk as it is cut. Preview pane appends index, char count, metadata badges. | 4 |
| 4. Embed | `POST /api/embed` → `job_id`; same SSE channel | Batched MiniLM encode, normalise, write to Chroma. Streams `done/total`. | 5 |
| 5. Inspect | `GET /api/collection?offset=&limit=` | Paginated record browser: id, text, full metadata, first 8 vector dims, and the vector norm. Raw-JSON toggle per record. | — |

Supporting endpoints: `GET /api/status/{job_id}` (SSE fallback), `POST /api/reset`
(clears session; optionally drops the collection).

The step-unlock rule is one client-side function: a step's controls are `disabled` until the
preceding step reports completion. State lives in the DOM plus the session JSON; no framework.

## Frontend

Reuses the deck's design tokens so the app reads as a continuation of the slides:
background `#0a0f1e`; accents cyan `#38e0cf`, amber `#ffb547`, pink `#ff5c8a`, violet `#8b7cff`,
green `#5ddb8b`; typefaces Bricolage Grotesque (display), Chivo (body), JetBrains Mono (code);
and the deck's `card`, `callout`, `demo-box`, and `pipe`-step component patterns.

Five stacked step sections, top to bottom, on one page. Locked steps are dimmed and
non-interactive. The chunk preview is a fixed-height scrollable pane so a 423-chunk run does not
push later steps off screen.

## Error handling

Every failure renders inside the step card that caused it. No stack traces on a projector.

| Failure | Handling |
|---|---|
| Non-PDF, or file >30MB | Rejected with a readable message before any processing |
| Encrypted / no text layer | Detected post-extraction: *"0 characters extracted — this looks like a scanned PDF; RAG needs a text layer."* A real gotcha, worth surfacing rather than swallowing |
| Chroma unreachable | Banner with retry. `depends_on` + healthcheck should prevent it; a mid-demo container death must not look like an app crash |
| SSE connection drops | JS falls back to polling `/api/status/{job_id}` |
| Semantic chunking latency | Elapsed-time counter plus an inline note that this is the expensive strategy (~1,834 sentences, under 10s on CPU for the bundled PDF) |
| Re-running the same PDF | Content-hash ids make it an overwrite, not 423 duplicate rows |

## Testing

`pytest`, run as `docker compose run --rm app pytest` so attendees need no local Python.
Tests document behaviour for readers rather than chasing a coverage percentage.

- **`test_loader.py`** — TOC lines removed; U+200B stripped; page count preserved; `doc_id` stable
  across repeated loads of identical input.
- **`test_chunkers.py`** — fixed-size splits mid-word (asserting the deck's criticism is literally
  true); recursive does not; structure-aware keeps `❖` sections intact; parent-document children all
  carry a resolvable `parent_id`; size/overlap respected within tolerance.
- **`test_store.py`** — write/read round-trip against an ephemeral Chroma client; ingesting the same
  document twice produces no duplicates; every record carries `embed_model`; stored vectors have
  norm ≈ 1.0.

## Documentation

**`CLAUDE.md`** — architecture map; the deck relationship, with each default traced to its slide;
the LangChain-vs-deck-LlamaIndex discrepancy recorded so it does not resurface as a bug report; the
Ollama seam for the future query build; conventions (high comment density in `pipeline/`, normal
elsewhere).

**`README.md`** — attendee quickstart: clone, `docker compose up`, open `localhost:8080`, click
*Use bundled document*. Plus a short tour of which file corresponds to which deck level.

## Out of scope

Retrieval, hybrid search, reranking, prompt assembly, generation, chat history, evaluation,
authentication, multi-user sessions, and access-control filtering. The deck covers these in Levels
5–6; they are the next build.

## Open item

The repository is not currently under version control, so this spec cannot be committed. Since the
project is destined for GitHub, `git init` plus an initial commit should happen before
implementation — pending confirmation.
