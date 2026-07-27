# Documents in, vectors out

A live, inspectable **RAG ingestion pipeline** built for a two-hour workshop on
retrieval-augmented generation, embeddings and vector databases.

Most RAG tutorials hand you a working pipeline and ask you to trust it. This one
makes you watch every step and shows you the numbers, because nearly every bad
RAG system fails during *indexing* — long before anyone writes a prompt.

```
┌─ 1 ─────────┐  ┌─ 2 ────────┐  ┌─ 3 ─────┐  ┌─ 4 ─────────┐  ┌─ 5 ─────────┐
│ Load a PDF  │→ │ Choose how │→ │ See the │→ │ Embed and   │→ │ Inspect     │
│ + clean it  │  │ to cut it  │  │ chunks  │  │ store       │  │ ChromaDB    │
└─────────────┘  └────────────┘  └─────────┘  └─────────────┘  └─────────────┘
```

Each step unlocks only when the previous one succeeds, so the dependency chain is
impossible to miss. **Bring your own PDF** — anything with a real text layer.

## Quick start

You need Docker and about 2 GB of disk (the embedding model is baked into the
image so the demo runs with no network).

```bash
git clone <this-repo> && cd class-rag
docker compose up -d          # first build takes a few minutes: model download
open http://localhost:8080
```

Upload a PDF and work down the page. To stop and remove everything, including
the stored vectors and any uploaded file:

```bash
docker compose down -v
```

## What each step actually demonstrates

**Step 1 — Load and clean.** Extracts text with `pypdf` via LangChain's
`PyPDFLoader`, then reports what cleaning removed: boilerplate headers and
footers, and invisible characters. The counts are shown because raw junk embeds
perfectly well and then quietly pollutes every search result. If your PDF is a
scan with no text layer, this step tells you rather than silently producing
nothing.

**Step 2 — Choose a chunking strategy.** Five strategies, all real, with the
trade-offs stated up front:

| Strategy | Verdict |
|---|---|
| **Fixed size** | Baseline only. Splits mid sentence and mid word. |
| **Recursive** | The right default. Respects natural boundaries. |
| **Structure aware** | Best value when documents have real structure. |
| **Semantic** | Slow and costs embeddings up front. Sometimes worth it. |
| **Parent document** | Best of both. Precise search, full context. |

Size and overlap sliders enable and disable themselves per strategy — semantic
chunking ignores both, because its cut points come from embedding distance
between neighbouring sentences, not character counts. Controls that would do
nothing are visibly disabled instead of silently ignored.

**Step 3 — Read the chunks.** Every chunk with its index, character count, and
the source page its start offset falls on. This is the step where a room
discovers what a bad chunk actually looks like. Try fixed-size at 200
characters and watch sentences get guillotined.

**Step 4 — Embed and store.** Loads `all-MiniLM-L6-v2` (384 dimensions, CPU, no
API key) and embeds in batches. The progress bar advances **per completed
batch** — it is real work, not a timer. Vectors go to ChromaDB with metadata
including the model name, so you can tell later which records are stale.

**Step 5 — Look inside the database.** Paginate the stored records: the id, the
text, the first few vector components, the dimensionality, the L2 norm, and the
full metadata. The norm reads `1.0` on every record — that is normalisation,
visible rather than asserted.

## Bring your own document

Drag any text-layer PDF onto step 1, up to 30 MB by default.

No sample document ships with this repository, and `*.pdf` is gitignored
outright so none can be committed by accident. If you want a permanent
presenter shortcut for one local file, copy the example override:

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# edit it to point at your file, then:
docker compose up -d
```

That mounts your document **read-only** and never copies it into the image. The
override file is gitignored.

## Configuration

Every default comes from the workshop slides. Copy `.env.example` to `.env` to
change any of them.

| Variable | Default | What it does |
|---|---|---|
| `DEFAULT_CHUNK_SIZE` | `700` | Starting chunk size in characters |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Characters shared between neighbours |
| `DEFAULT_STRATEGY` | `recursive` | Strategy selected on load |
| `SEMANTIC_PERCENTILE` | `95` | Breakpoint threshold for semantic chunking |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBED_DIMS` | `384` | Vector dimensionality |
| `EMBED_BATCH_SIZE` | `64` | Chunks per batch |
| `MAX_UPLOAD_MB` | `30` | Upload size cap |
| `CHROMA_HOST` / `CHROMA_PORT` | `chromadb` / `8000` | Vector store location |
| `CHROMA_COLLECTION` | `workshop` | Collection name |

## Tech stack

- **FastAPI** — routes, plus Server-Sent Events for progress with a polling
  fallback for when SSE is blocked by a proxy
- **LangChain** — `PyPDFLoader`, the five text splitters, and the
  `HuggingFaceEmbeddings` wrapper
- **ChromaDB** — the vector store, talked to through the raw `chromadb` client
  rather than `langchain-chroma`, so the app can report per-batch progress and
  show you the actual vectors
- **sentence-transformers** — `all-MiniLM-L6-v2`, baked into the image at build
  time so nothing downloads at runtime
- Vanilla HTML, CSS and JavaScript — no build step, nothing to install to read
  the frontend

## Running the tests

```bash
docker compose run --rm app pytest -q -m "not slow"   # fast: no model load
docker compose run --rm app pytest -q -m slow         # loads the real model
```

Test PDFs are generated at runtime with `reportlab`; none is committed.

If you change any source file, rebuild before testing — the image copies source
in at build time rather than mounting it:

```bash
docker compose build app
```

## Project layout

```
app/
  config.py            every tunable, annotated with the slide it came from
  main.py              FastAPI routes
  jobs.py              background job registry + resumable SSE
  session.py           per-session state and the step-unlock rule
  pipeline/
    loader.py          PDF text extraction and cleaning
    chunkers.py        the five chunking strategies
    embedder.py        batched embedding with real progress
    store.py           ChromaDB writes and reads
  templates/, static/  the single-page frontend
tests/                 172 tests
rag-workshop.html      the workshop slide deck
```

## Scope

This is the **indexing half** of RAG: everything up to "the vectors are in the
database." There is no retrieval, no reranking, no prompt and no LLM here —
those are the second half of the workshop, and mixing them in would blur the
point that indexing quality decides everything downstream.

One deliberate inconsistency: the slide deck's code samples use LlamaIndex while
this app uses LangChain. The concepts are identical and the deck is left as
presented.
