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

## Intended use

This is teaching software. It is built to be read, projected, and argued with
during a two-hour session, and it is deliberately small enough to hold in your
head. That shape is a series of trade-offs, and they are not ones to carry into
a deployment:

- **No authentication or authorisation.** Every endpoint is open, including the
  one that drops the collection.
- **Single process, single presenter.** Job and session state live in memory,
  so running more than one worker breaks progress reporting and step unlocking.
- **The collection is shared.** Step 5 lists every record in the database,
  regardless of which session wrote it.
- **The upload limit is advisory.** The request body is fully received before
  its size can be checked, and there is no rate limiting.
- **Nothing is hardened for an untrusted network.** The session cookie is not
  `Secure`, there is no CSRF protection, and ChromaDB runs without credentials.

Each is a sensible simplification for a demo on a laptop and a liability
anywhere else. The durable part is the pipeline itself — loading, cleaning,
chunking, embedding, storing, and the decisions inside each stage. The web app
around it is scaffolding for making those decisions visible.

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

## The other half: `/chat`

Indexing is only worth judging by what retrieval can then find, so `/chat` runs
the query side and shows its working. Two columns: the conversation on the left,
an inspector on the right.

The inspector is the point. For every question it shows the query as a vector,
then each stage with its timing — `embed_query → search → rank → filter →
assemble` — then **every candidate the search returned**, not just the winners:
its cosine distance *and* the similarity derived from it, its MMR score where
relevant, its page citation, and for the losers, which stage eliminated it
(`not_top_k`, `below_threshold`, `mmr_redundant`).

Three controls change the outcome live:

- **top-k** — how many chunks reach the answer
- **MMR** with a λ slider — maximal marginal relevance, trading relevance against
  diversity. At λ=1 it collapses to plain similarity; at λ=0 it maximises
  variety. Watch the selected set change.
- **minimum similarity** — and when nothing clears it, the answer is "I don't
  know", with the rejected candidates still listed and scored

That last state is the most useful thing on the page. A RAG system that
confidently answers from nothing is the failure everyone ships; here it is a
first-class, visible outcome rather than an error.

`/chat` is always reachable and does not depend on this browser having run the
pipeline — only on something being in ChromaDB. Open it with an empty collection
and it tells you so.

**Answers are extractive.** No LLM runs anywhere in this app: the answer *is*
the retrieved chunks with their citations, which keeps the line between "the
retriever found this text" and "a model wrote this" visible. When nothing
clears the similarity threshold the answer is an honest "I don't know" instead
of a guess.

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
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question on `/chat` |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Below this similarity, the answer is "I don't know" |
| `RETRIEVAL_MMR_LAMBDA` | `0.5` | MMR relevance/diversity balance, 0–1 |
| `RETRIEVAL_POOL_MULTIPLIER` | `4` | Candidates fetched per requested chunk, so MMR has runners-up |

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
    retriever.py       query-time search, MMR, thresholds, the full trace
  templates/, static/  the ingestion page and the chat page
tests/                 245 tests
rag-workshop.html      the workshop slide deck
```

## Contributing

Bug reports and pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first — it is short, and it covers the three
rules that are not negotiable here (no PDF is ever committed, progress is never
faked, comments must be true) plus how this project tests. The short version of
that last part: a green suite is not evidence, so break your implementation on
purpose and confirm your test fails before trusting it.

## Licence

[MIT](LICENSE) — use it, fork it, teach from it. Attribution appreciated but
not required.

The slide deck (`rag-workshop.html`) is covered by the same licence. Its code
samples use LlamaIndex while this app uses LangChain; see Scope below.

Third-party components keep their own licences, all permissive: FastAPI and
LangChain (MIT), ChromaDB, sentence-transformers and transformers (Apache-2.0),
uvicorn and Jinja2 (BSD). The embedding model,
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
is Apache-2.0 and is downloaded into the Docker image at build time rather than
redistributed here.

## Scope

The five-step page at `/` is the **indexing half** of RAG: everything up to "the
vectors are in the database." It deliberately stops there — no retrieval, no
query, no LLM — because mixing the halves would blur the point that indexing
quality decides everything downstream.

`/chat` is the other half, kept on its own page for the same reason. It has no
reranking and no agentic retrieval: one query, one search, one answer, with every
intermediate value on screen.

One deliberate inconsistency: the slide deck's code samples use LlamaIndex while
this app uses LangChain. The concepts are identical and the deck is left as
presented.
