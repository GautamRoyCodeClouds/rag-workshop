# RAG Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-page web app that ingests a PDF through five selectable chunking strategies into ChromaDB, unlocking one step at a time so a presenter can narrate each stage live.

**Architecture:** Two Docker containers — a FastAPI app and ChromaDB. The app serves one Jinja2 page; pipeline work runs in asyncio tasks and reports progress over Server-Sent Events. LangChain supplies the PDF loader, the five text splitters, and the embedding wrapper; the raw `chromadb` client handles writes and vector reads so batch progress and the vector preview can be shown honestly.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, Jinja2, LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-experimental`), `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims), ChromaDB, pytest + reportlab.

**Spec:** `docs/superpowers/specs/2026-07-27-rag-ingestion-pipeline-design.md`

## Global Constraints

- **Python 3.12** in the container. No local Python required to run or test.
- **LangChain, not LlamaIndex.** The deck's code slides use LlamaIndex; that discrepancy is intentional and recorded in CLAUDE.md. Do not "fix" the deck.
- **No PDF may be committed.** `.gitignore` excludes `*.pdf` absolutely, with no allow-list. Tests generate PDFs at runtime.
- **Model baked into the image** at build time via `HF_HOME=/opt/hf`. Nothing downloads at runtime.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, L2-normalised at write time.
- **Deck defaults, verbatim:** chunk size `700`, overlap `100`, strategy `recursive`, metric `cosine`.
- **Comment density:** high in `app/pipeline/` (attendees read these first — every strategy docstring quotes the deck's verdict); normal elsewhere.
- **No dead code.** `OLLAMA_BASE_URL` is the one documented exception: declared in config and CLAUDE.md as a seam for the future query build, read by nothing.
- **Chroma metadata cannot hold `None`.** Use `""` for an absent `parent_id`.
- **Honesty about progress:** embedding progress is genuinely incremental (per batch). Chunking is atomic inside LangChain's splitters — the *rendering* of chunks streams, the splitting does not. Never insert artificial delays to fake progress.
- **Deviation from the spec, deliberate:** the spec said writes go through `langchain-chroma`. They go through the raw `chromadb` client instead, because that library computes embeddings internally, which would prevent both per-batch progress reporting and the vector preview. LangChain still supplies the loader, all five splitters, and the embedding wrapper. Recorded in CLAUDE.md.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/config.py` | All tunables, read from env, each default annotated with its deck slide |
| `app/pipeline/loader.py` | PDF → cleaned text; cleaning counts; page-offset map |
| `app/pipeline/chunkers.py` | The five strategies behind one `chunk()` interface |
| `app/pipeline/embedder.py` | MiniLM wrapper; batched embedding with progress callback |
| `app/pipeline/store.py` | Chroma writes (delete-before-write) and record reads |
| `app/session.py` | Per-session state + JSON persistence for refresh-safety |
| `app/jobs.py` | Job registry and SSE event queues |
| `app/main.py` | FastAPI routes, SSE endpoints, page render |
| `app/templates/index.html` | The five stacked step sections |
| `app/static/app.css` | Deck design tokens |
| `app/static/app.js` | Step unlock, SSE consumption, polling fallback |
| `tests/conftest.py` | Runtime-generated PDF fixtures (reportlab) |

Task order below is dependency order. Each task ends with something independently runnable.

---

## Task 1: Docker foundation, config, and health check

**Files:**
- Create: `app/__init__.py`, `app/config.py`, `app/main.py`, `app/requirements.txt`, `requirements-dev.txt`, `app/Dockerfile`, `.dockerignore`, `.env.example`, `docker-compose.override.yml.example`
- Modify: `docker-compose.yml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `app.config.Settings` (frozen dataclass) and `app.config.settings` (module-level instance). Fields: `chroma_host: str`, `chroma_port: int`, `chroma_collection: str`, `embed_model: str`, `embed_dims: int`, `embed_batch_size: int`, `default_chunk_size: int`, `default_chunk_overlap: int`, `default_strategy: str`, `semantic_percentile: int`, `max_upload_mb: int`, `data_dir: Path`, `local_pdf_path: str`, `ollama_base_url: str`. Classmethod `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings`. Properties `local_pdf -> Path | None` and `max_upload_bytes -> int`. Also produces `app.main.app` (FastAPI instance) and route `GET /api/health`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Config defaults are the deck's numbers. If one of these fails, either the
deck changed or someone drifted from it -- check the slide first.
"""
from pathlib import Path

from app.config import Settings


def test_defaults_match_the_deck():
    s = Settings.from_env({})
    # "Sensible defaults for version one" slide (Level 6)
    assert s.default_chunk_size == 700
    assert s.default_chunk_overlap == 100
    assert s.default_strategy == "recursive"
    # Level 2 model table, self-host row
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.embed_dims == 384


def test_env_overrides_are_typed():
    s = Settings.from_env({"DEFAULT_CHUNK_SIZE": "1500", "CHROMA_PORT": "9000"})
    assert s.default_chunk_size == 1500
    assert s.chroma_port == 9000
    assert isinstance(s.chroma_port, int)


def test_blank_env_value_falls_back_to_default():
    assert Settings.from_env({"DEFAULT_CHUNK_SIZE": ""}).default_chunk_size == 700


def test_local_pdf_is_none_when_unset():
    assert Settings.from_env({}).local_pdf is None


def test_local_pdf_is_none_when_path_missing():
    assert Settings.from_env({"LOCAL_PDF_PATH": "/nope/absent.pdf"}).local_pdf is None


def test_local_pdf_resolves_when_file_exists(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n")
    assert Settings.from_env({"LOCAL_PDF_PATH": str(p)}).local_pdf == Path(p)


def test_max_upload_bytes_derives_from_megabytes():
    assert Settings.from_env({"MAX_UPLOAD_MB": "2"}).max_upload_bytes == 2 * 1024 * 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Write the config module**

Create `app/__init__.py` as an empty file.

Create `app/config.py`:

```python
"""Central configuration for the ingestion demo.

Every default here is a number taken from the workshop deck, and the comment
beside it names the slide. That traceability is deliberate: when an attendee
asks "why 700?", the answer is one grep away.

Values are read from the environment once at import. Tests build Settings
directly via from_env() so they never depend on the ambient environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # --- ChromaDB -----------------------------------------------------------
    chroma_host: str = "chromadb"          # Compose service name
    chroma_port: int = 8000
    chroma_collection: str = "workshop"

    # --- Embeddings ---------------------------------------------------------
    # Level 2, "Which embedding model, and does it matter": the self-host row.
    # 384 dims, runs on CPU, needs no API key -- which is why it works offline.
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dims: int = 384
    embed_batch_size: int = 64

    # --- Chunking -----------------------------------------------------------
    # Level 6, "Sensible defaults for version one".
    default_chunk_size: int = 700
    default_chunk_overlap: int = 100
    default_strategy: str = "recursive"     # Level 3: "the right default"
    semantic_percentile: int = 95           # LangChain SemanticChunker default

    # --- Upload -------------------------------------------------------------
    max_upload_mb: int = 30
    data_dir: Path = Path("/data")

    # Optional presenter convenience. When this resolves to a real file the UI
    # offers a "Use local document" button, so the presenter can skip a large
    # file picker mid-talk. Unset for everyone else, and the button vanishes.
    local_pdf_path: str = ""

    # Reserved seam for the future query build (deck Levels 5-6). Read by
    # nothing in this scope -- see CLAUDE.md. deepseek-r1:1.5b is the intended
    # generation model.
    ollama_base_url: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from a mapping, defaulting to os.environ."""
        e = os.environ if env is None else env

        def text(key: str, default: str) -> str:
            value = e.get(key)
            return default if value is None or value == "" else value

        def number(key: str, default: int) -> int:
            value = e.get(key)
            return default if value is None or value == "" else int(value)

        return cls(
            chroma_host=text("CHROMA_HOST", cls.chroma_host),
            chroma_port=number("CHROMA_PORT", cls.chroma_port),
            chroma_collection=text("CHROMA_COLLECTION", cls.chroma_collection),
            embed_model=text("EMBED_MODEL", cls.embed_model),
            embed_dims=number("EMBED_DIMS", cls.embed_dims),
            embed_batch_size=number("EMBED_BATCH_SIZE", cls.embed_batch_size),
            default_chunk_size=number("DEFAULT_CHUNK_SIZE", cls.default_chunk_size),
            default_chunk_overlap=number(
                "DEFAULT_CHUNK_OVERLAP", cls.default_chunk_overlap
            ),
            default_strategy=text("DEFAULT_STRATEGY", cls.default_strategy),
            semantic_percentile=number("SEMANTIC_PERCENTILE", cls.semantic_percentile),
            max_upload_mb=number("MAX_UPLOAD_MB", cls.max_upload_mb),
            data_dir=Path(text("DATA_DIR", str(cls.data_dir))),
            local_pdf_path=text("LOCAL_PDF_PATH", cls.local_pdf_path),
            ollama_base_url=text("OLLAMA_BASE_URL", cls.ollama_base_url),
        )

    @property
    def local_pdf(self) -> Path | None:
        """The presenter's local document, or None when absent or unset."""
        if not self.local_pdf_path:
            return None
        path = Path(self.local_pdf_path)
        return path if path.is_file() else None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings.from_env()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Write the requirements files**

Create `app/requirements.txt`:

```
# Bounded rather than exactly pinned: the versions that resolve at build time
# are frozen into requirements.lock.txt in Step 9 for reproducibility.
fastapi>=0.115,<0.116
uvicorn[standard]>=0.34,<0.35
jinja2>=3.1,<4
python-multipart>=0.0.20,<0.1

# LangChain: PDF loader, the five splitters, and the embedding wrapper.
langchain-community>=0.3.14,<0.4
langchain-text-splitters>=0.3.5,<0.4
langchain-huggingface>=0.1.2,<0.2
langchain-experimental>=0.3.4,<0.4

# Embeddings + vector store
sentence-transformers>=3.3,<4
chromadb>=0.6,<0.7
pypdf>=5.1,<6
numpy>=1.26,<3
```

Create `requirements-dev.txt`:

```
pytest>=8.3,<9
httpx>=0.28,<0.29        # FastAPI TestClient dependency
reportlab>=4.2,<5        # generates test PDFs at runtime; none is committed
```

- [ ] **Step 6: Write the minimal app with a health check**

Create `app/main.py`:

```python
"""FastAPI application.

Later tasks add the pipeline routes. This task establishes the app object plus
a health check, so `docker compose up` can be validated before any pipeline
code exists.
"""

from __future__ import annotations

import chromadb
from fastapi import FastAPI

from app.config import settings

app = FastAPI(title="RAG Ingestion Pipeline", docs_url="/api/docs")


@app.get("/api/health")
def health() -> dict:
    """Report liveness and whether Chroma is reachable.

    A Chroma failure is reported as degraded rather than raised: the UI shows a
    retry banner, which beats a stack trace on a projector.
    """
    chroma_ok, detail = False, ""
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        chroma_ok = True
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        detail = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if chroma_ok else "degraded",
        "chroma": {"reachable": chroma_ok, "detail": detail},
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
    }
```

- [ ] **Step 7: Write the Dockerfile**

Create `app/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# HF_HOME is where model weights land. Setting it *before* the download step is
# what makes the model part of the image rather than a runtime fetch -- the
# workshop is presented offline, so nothing may download live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/srv

WORKDIR /srv

# build-essential is needed by some wheels; purged after install so the image
# does not carry a compiler around.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt \
 && apt-get purge -y --auto-remove build-essential

# Bake the embedding model into the image. ~90MB, downloaded once at build.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('model cached')"

COPY app/ /srv/app/
COPY tests/ /srv/tests/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Create `.dockerignore`:

```
# No PDF may enter the build context -- the presenter's document is internal
# and must never end up in an image layer.
*.pdf

.git/
.gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ruff_cache/
data/
docs/
.impeccable/
docker-compose.override.yml
rag-workshop.html
```

- [ ] **Step 8: Extend Compose and add the example override**

Replace `docker-compose.yml` with:

```yaml
services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/data
    environment:
      # Persist collections to the mounted volume instead of memory
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: /data
      ANONYMIZED_TELEMETRY: "FALSE"
    healthcheck:
      # The image is slim (no curl), so probe the port with bash's /dev/tcp.
      # The /dev/tcp syntax requires a slash between host and port, not a colon.
      test: ["CMD", "/bin/bash", "-c", "cat < /dev/null > /dev/tcp/localhost/8000"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 20s

  app:
    build:
      context: .
      dockerfile: app/Dockerfile
    container_name: rag-app
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - app-data:/data
    environment:
      CHROMA_HOST: chromadb
      CHROMA_PORT: "8000"
      DATA_DIR: /data
    depends_on:
      chromadb:
        condition: service_healthy

volumes:
  chroma-data:
  app-data:
```

Create `docker-compose.override.yml.example`:

```yaml
# Presenter-only. Copy to docker-compose.override.yml (gitignored) and point it
# at a local PDF. Compose merges override files automatically, and their absence
# is not an error -- so attendees are unaffected.
#
# The mount is read-only, and the document is NEVER copied into the image:
# baking it into a layer would leak it on any image push or share.
services:
  app:
    volumes:
      - ./your-document.pdf:/srv/samples/local.pdf:ro
    environment:
      LOCAL_PDF_PATH: /srv/samples/local.pdf
```

Create `.env.example`:

```bash
# Copy to .env to override any default. Every value here is the shipped
# default; the comment names the deck slide it came from.

CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION=workshop

# Level 2, model table, self-host row: 384 dims, CPU, no API key
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIMS=384
EMBED_BATCH_SIZE=64

# Level 6, "Sensible defaults for version one"
DEFAULT_CHUNK_SIZE=700
DEFAULT_CHUNK_OVERLAP=100
DEFAULT_STRATEGY=recursive
SEMANTIC_PERCENTILE=95

MAX_UPLOAD_MB=30
DATA_DIR=/data

# Presenter convenience; see docker-compose.override.yml.example
# LOCAL_PDF_PATH=/srv/samples/local.pdf

# Reserved for the future query build. Unused in this scope.
# OLLAMA_BASE_URL=http://host.docker.internal:11434
```

- [ ] **Step 9: Build, verify health, prove offline, freeze the lockfile**

Run:

```bash
docker compose build app
docker compose up -d
sleep 15
curl -s localhost:8080/api/health
```

Expected: `{"status":"ok","chroma":{"reachable":true,"detail":""},"embed_model":"sentence-transformers/all-MiniLM-L6-v2","embed_dims":384}`

Prove the model is baked in rather than fetched at runtime:

```bash
docker compose run --rm --network none app python -c \
  "from sentence_transformers import SentenceTransformer as S; \
   m = S('sentence-transformers/all-MiniLM-L6-v2'); \
   print('offline load OK, dims =', m.get_sentence_embedding_dimension())"
```

Expected: `offline load OK, dims = 384`. This is the offline presentation requirement, enforced rather than assumed.

Freeze resolved versions and run the tests in-container:

```bash
docker compose run --rm app pip freeze > requirements.lock.txt
docker compose run --rm app pytest tests/test_config.py -v
```

Expected: 7 passed

- [ ] **Step 10: Commit**

```bash
git add app/__init__.py app/config.py app/main.py app/requirements.txt \
        app/Dockerfile requirements-dev.txt requirements.lock.txt \
        .dockerignore .env.example docker-compose.yml \
        docker-compose.override.yml.example tests/test_config.py
git commit -m "feat: Docker foundation, config from deck defaults, health check

Two-container stack. The embedding model is baked into the image at build time
and verified loadable with --network none, so the offline presentation
requirement is enforced by a check rather than a hope."
```

---

## Task 2: PDF loader and text cleaning

**Files:**
- Create: `app/pipeline/__init__.py`, `app/pipeline/loader.py`, `tests/conftest.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: nothing from Task 1 at runtime (independent module).
- Produces:
  - `CleanResult` dataclass: `pages: list[str]`, `boilerplate_lines_removed: int`, `invisible_chars_removed: int`
  - `clean_pages(raw_pages: list[str]) -> CleanResult` — pure, no I/O
  - `LoadResult` dataclass: `text: str`, `page_count: int`, `char_count: int`, `pages_without_text: int`, `boilerplate_lines_removed: int`, `invisible_chars_removed: int`, `doc_id: str`, `page_offsets: list[tuple[int, int]]`, method `page_for_offset(offset: int) -> int`
  - `load_pdf(path: str | Path) -> LoadResult`
  - `EmptyDocumentError(ValueError)`
- Fixtures produced for later tasks: `structured_pdf`, `flat_pdf`, `dirty_pages`.

**Why cleaning is split from loading:** `clean_pages` is pure, so assertions about zero-width characters are exact. Whether U+200B survives a reportlab→pypdf round-trip is a property of those libraries, not of our code; testing it through a generated PDF would test the wrong thing.

> **Correction applied during execution (commit `bc5785d`).** The `load_pdf` code
> below contains a real bug, left here so the fix stays legible rather than being
> silently rewritten. It accumulates `page_offsets` assuming a separator between
> every page, then builds the text with `separator.join(parts).strip()`. When a
> leading page cleans to `""` — a logo-only cover, an all-boilerplate title page —
> the joined string *begins* with the separator and `.strip()` removes it,
> shifting every later page's true start left while the recorded offsets still
> assume it is there. One empty leading page makes every citation in the document
> point a page too early, and the fixture's blank page is *last*, so no test
> caught it.
>
> The shipped implementation extracts a `_join_pages()` helper that measures the
> leading whitespace `.strip()` removes and subtracts it from every offset, so
> text and offsets are correct by construction. Three further corrections landed
> in the same commit: the TOC regex now requires a genuine leader (3+ dots, an
> ellipsis, or a 6+ character gap) rather than any two whitespace-or-dot
> characters, which had been deleting lines like `"All rights reserved.  2024"`;
> boilerplate detection now considers only lines within 3 of a page edge, where
> running headers actually live, instead of ranking purely by frequency; and the
> page-attribution tests now pin an interior page, an exact boundary, an empty
> leading page, and an offset past the end.
>
> **`app/pipeline/loader.py` is the source of truth — read it, not the block below.**

- [ ] **Step 1: Write the fixtures**

Create `tests/conftest.py`:

```python
"""Test fixtures.

No PDF is committed to this repository -- the presenter's document is internal.
Every PDF the suite uses is generated here at runtime with reportlab, which also
makes these fixtures executable documentation of what "dirty input" means.
"""

from __future__ import annotations

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FOOTER = "Internal Handbook -- Confidential"

TOC_ENTRIES = [
    "Introduction .......................... 1",
    "Getting Started ....................... 2",
    "Companies Management .................. 3",
    "People Management ..................... 4",
    "Reporting ............................. 5",
]

SECTIONS = [
    ("1. Introduction", [
        "This handbook describes the platform and its day to day operation.",
        "Each section covers one area of the product in practical terms.",
    ]),
    ("2. Companies Management", [
        "Companies are the top level record in the system.",
        "Every company holds an address, operating hours and custom fields.",
    ]),
    ("3. People Management", [
        "People belong to one or more companies as contacts.",
        "A person record holds an address, notes and custom fields.",
    ]),
]


def _draw_lines(pdf: canvas.Canvas, lines: list[str], start_y: int = 750) -> None:
    y = start_y
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 16


@pytest.fixture(scope="session")
def structured_pdf(tmp_path_factory) -> str:
    """A PDF with a TOC, numbered headings, a repeated footer, and a blank page.

    Deliberately messy, mirroring what real exported documents look like.
    """
    path = tmp_path_factory.mktemp("pdfs") / "structured.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)

    # Page 1: title + table of contents (retrieval poison; the loader strips it)
    _draw_lines(pdf, ["Sample Handbook", ""] + TOC_ENTRIES)
    pdf.drawString(72, 40, FOOTER)
    pdf.showPage()

    # Pages 2-4: content, each carrying the same footer
    for heading, body in SECTIONS:
        _draw_lines(pdf, [heading, ""] + body)
        pdf.drawString(72, 40, FOOTER)
        pdf.showPage()

    # Final page: intentionally blank -- no text layer at all
    pdf.showPage()
    pdf.save()
    return str(path)


@pytest.fixture(scope="session")
def flat_pdf(tmp_path_factory) -> str:
    """A PDF with no headings whatsoever.

    Used to prove structure-aware chunking degrades to recursive rather than
    returning one section containing the whole document.
    """
    path = tmp_path_factory.mktemp("pdfs") / "flat.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    sentence = "The quick brown fox jumps over the lazy dog and keeps running."
    _draw_lines(pdf, [sentence] * 30)
    pdf.save()
    return str(path)


@pytest.fixture
def dirty_pages() -> list[str]:
    """Raw page text as an extractor hands it over, including U+200B.

    Exactly three zero-width spaces, marked below, so the count assertion in
    test_loader.py is exact.
    """
    return [
        "Sample Handbook\n"
        "Introduction .......................... 1\n"
        "Getting Started ....................... 2\n"
        "Companies Management .................. 3\n" + FOOTER,
        "1.​ Introduction\n"                       # zero-width #1
        "This handbook describes    the platform.\n"
        "\n\n\n"
        "It covers day to day operation.\n" + FOOTER,
        "2.​ Companies​ Management\n"          # zero-width #2 and #3
        "Companies are the top level record.\n" + FOOTER,
        "3. People Management\nPeople belong to companies.\n" + FOOTER,
        "",
    ]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_loader.py`:

```python
"""Loader tests.

clean_pages is tested against strings so the zero-width assertions are exact.
load_pdf is tested against a generated PDF for file-level properties.
"""

import pytest

from app.pipeline.loader import EmptyDocumentError, clean_pages, load_pdf


class TestCleanPages:
    def test_strips_zero_width_characters(self, dirty_pages):
        result = clean_pages(dirty_pages)
        assert result.invisible_chars_removed == 3
        assert "​" not in "".join(result.pages)

    def test_removes_table_of_contents_lines(self, dirty_pages):
        joined = "\n".join(clean_pages(dirty_pages).pages)
        assert "Companies Management .................. 3" not in joined
        # ...but the real heading of the same name survives
        assert "Companies Management" in joined

    def test_removes_the_repeated_footer(self, dirty_pages):
        result = clean_pages(dirty_pages)
        assert "Confidential" not in "\n".join(result.pages)
        assert result.boilerplate_lines_removed >= 4

    def test_squashes_whitespace_runs(self, dirty_pages):
        joined = "\n".join(clean_pages(dirty_pages).pages)
        assert "    " not in joined
        assert "\n\n\n" not in joined

    def test_keeps_body_text(self, dirty_pages):
        assert "top level record" in "\n".join(clean_pages(dirty_pages).pages)

    def test_short_documents_have_no_boilerplate(self):
        # With two pages, "appears on most pages" is not evidence of a running
        # header -- it may simply be a two-page document that repeats a line.
        result = clean_pages(["Alpha line\nBody one", "Alpha line\nBody two"])
        assert result.boilerplate_lines_removed == 0

    def test_page_count_is_preserved(self, dirty_pages):
        assert len(clean_pages(dirty_pages).pages) == len(dirty_pages)


class TestLoadPdf:
    def test_reports_page_count_and_blank_pages(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.page_count == 5
        assert result.pages_without_text == 1

    def test_doc_id_is_stable_across_identical_loads(self, structured_pdf):
        assert load_pdf(structured_pdf).doc_id == load_pdf(structured_pdf).doc_id

    def test_doc_id_differs_between_documents(self, structured_pdf, flat_pdf):
        assert load_pdf(structured_pdf).doc_id != load_pdf(flat_pdf).doc_id

    def test_char_count_matches_text_length(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.char_count == len(result.text)

    def test_page_attribution_starts_at_one_and_advances(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.page_for_offset(0) == 1
        assert result.page_for_offset(len(result.text) - 1) >= 1

    def test_offset_before_the_first_page_clamps(self, structured_pdf):
        assert load_pdf(structured_pdf).page_for_offset(-5) == 1

    def test_rejects_a_document_with_no_text_layer(self, tmp_path):
        # A PDF with pages but no extractable text is the scanned-document case.
        # It must fail loudly, not silently produce an empty collection.
        from reportlab.pdfgen import canvas

        path = tmp_path / "scanned.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.showPage()
        pdf.save()

        with pytest.raises(EmptyDocumentError, match="text layer"):
            load_pdf(path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 4: Write the loader**

Create `app/pipeline/__init__.py` as an empty file.

Create `app/pipeline/loader.py`:

```python
"""PDF loading and text cleaning -- step 1 of the pipeline.

Level 2 of the deck lists "Embedding raw junk" as one of four ways people break
RAG: navigation menus, cookie banners and page footers embed perfectly well and
then pollute every result set. So cleaning here is a visible feature, not a
hidden detail -- every removal is counted and reported to the UI.

Two entry points, deliberately separated:

  clean_pages()  pure, no I/O -- the interesting logic, exactly testable
  load_pdf()     extracts with LangChain's PyPDFLoader, then delegates
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# Zero-width and byte-order marks. Documents exported from Google Docs are
# littered with U+200B: invisible on screen, yet it burns tokens and can split a
# word in the middle as far as a tokeniser is concerned.
_INVISIBLE_RE = re.compile("[​‌‍﻿]")

# A table-of-contents line: text, a run of dots or spaces, then a trailing page
# number. "Companies Management .......... 16"
_TOC_LINE_RE = re.compile(r"^\s*\S.*?[\s.…]{2,}\d{1,4}\s*$")

# Lines this long are prose, not running headers, whatever their frequency.
_MAX_BOILERPLATE_LEN = 120

# Below this many pages, "appears on most pages" is not evidence of boilerplate.
_MIN_PAGES_FOR_BOILERPLATE = 4

# Fraction of pages a line must appear on before it counts as boilerplate.
_BOILERPLATE_PAGE_FRACTION = 0.5

# Only strip TOC lines from the front, where a TOC actually lives. A mid-document
# line that happens to end in a number is probably data.
_TOC_LEADING_FRACTION = 0.15


class EmptyDocumentError(ValueError):
    """Raised when a PDF yields no usable text.

    Almost always a scanned document: the pages are images, so there is no text
    layer to extract. Worth failing loudly -- it is a real RAG gotcha, and a
    silently empty collection is far more confusing than an error.
    """


@dataclass
class CleanResult:
    pages: list[str]
    boilerplate_lines_removed: int
    invisible_chars_removed: int


@dataclass
class LoadResult:
    text: str
    page_count: int
    char_count: int
    pages_without_text: int
    boilerplate_lines_removed: int
    invisible_chars_removed: int
    doc_id: str
    # (start_char, page_number) pairs, ascending by start_char.
    page_offsets: list[tuple[int, int]]

    def page_for_offset(self, offset: int) -> int:
        """Which source page does this character offset fall on?

        Cleaning concatenates every page into one string, so a chunk can straddle
        a page boundary. Chunks are attributed to the page containing their
        *start* offset: that keeps splitters free to cross boundaries (recursive
        and semantic must) while still giving every chunk something citable.
        """
        if not self.page_offsets:
            return 1
        starts = [start for start, _ in self.page_offsets]
        index = max(bisect.bisect_right(starts, offset) - 1, 0)
        return self.page_offsets[index][1]


def _find_boilerplate(pages: list[str]) -> set[str]:
    """Lines recurring across most pages -- running headers and footers."""
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        # Count each distinct line once per page, so a line repeated many times
        # on a single page does not masquerade as a running header.
        counts.update({
            line.strip()
            for line in page.splitlines()
            if line.strip() and len(line.strip()) <= _MAX_BOILERPLATE_LEN
        })

    threshold = len(pages) * _BOILERPLATE_PAGE_FRACTION
    return {line for line, count in counts.items() if count >= threshold}


def _squash_whitespace(text: str) -> str:
    """Collapse run-on spaces and blank lines left behind by extraction."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def clean_pages(raw_pages: list[str]) -> CleanResult:
    """Strip the parts of a document that should never reach an embedding model.

    Returns cleaned pages plus counts, because the UI displays them: watching
    "removed 412 boilerplate lines, stripped 1,847 invisible characters" appear
    on screen is what turns the deck's gotcha into something the room has seen.
    """
    invisible_removed = sum(len(_INVISIBLE_RE.findall(page)) for page in raw_pages)
    pages = [_INVISIBLE_RE.sub("", page) for page in raw_pages]

    boilerplate = _find_boilerplate(pages)
    toc_cutoff = max(1, int(len(pages) * _TOC_LEADING_FRACTION))

    lines_removed = 0
    kept_pages: list[str] = []
    for page_index, page in enumerate(pages):
        kept_lines: list[str] = []
        for line in page.splitlines():
            stripped = line.strip()
            if stripped and stripped in boilerplate:
                lines_removed += 1
                continue
            if page_index < toc_cutoff and _TOC_LINE_RE.match(line):
                lines_removed += 1
                continue
            kept_lines.append(line)
        kept_pages.append(_squash_whitespace("\n".join(kept_lines)))

    return CleanResult(
        pages=kept_pages,
        boilerplate_lines_removed=lines_removed,
        invisible_chars_removed=invisible_removed,
    )


def load_pdf(path: str | Path) -> LoadResult:
    """Extract and clean a PDF, returning text plus everything the UI reports."""
    documents = PyPDFLoader(str(path)).load()
    raw_pages = [doc.page_content or "" for doc in documents]

    cleaned = clean_pages(raw_pages)
    pages_without_text = sum(1 for page in cleaned.pages if not page.strip())

    # Assemble one string, recording where each page starts so chunks produced
    # downstream can be attributed back to a page number.
    separator = "\n\n"
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_index, page in enumerate(cleaned.pages):
        offsets.append((cursor, page_index + 1))
        parts.append(page)
        cursor += len(page) + len(separator)

    text = separator.join(parts).strip()
    if not text:
        raise EmptyDocumentError(
            f"0 characters extracted from {Path(path).name}. This looks like a "
            "scanned PDF -- the pages are images, so there is no text layer. RAG "
            "needs extractable text; run OCR over the document first."
        )

    return LoadResult(
        text=text,
        page_count=len(raw_pages),
        char_count=len(text),
        pages_without_text=pages_without_text,
        boilerplate_lines_removed=cleaned.boilerplate_lines_removed,
        invisible_chars_removed=cleaned.invisible_chars_removed,
        doc_id=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        page_offsets=offsets,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_loader.py -v`
Expected: PASS, 14 passed

If `test_reports_page_count_and_blank_pages` disagrees on `pages_without_text`, check whether reportlab emitted a trailing page for the final `showPage()`; adjust the fixture rather than loosening the assertion.

- [ ] **Step 6: Commit**

```bash
git add app/pipeline/__init__.py app/pipeline/loader.py tests/conftest.py tests/test_loader.py
git commit -m "feat: PDF loader with visible boilerplate cleaning

Cleaning lives in a pure clean_pages() so the zero-width and TOC assertions are
exact; load_pdf() handles extraction. Every removal is counted because the UI
reports it -- the deck's 'embedding raw junk' gotcha becomes something the room
watches happen.

Scanned PDFs raise EmptyDocumentError instead of yielding a silently empty
collection. Fixtures generate PDFs at runtime; none is committed."
```

---

## Task 3: The five chunking strategies

**Files:**
- Create: `app/pipeline/chunkers.py`
- Test: `tests/test_chunkers.py`

**Interfaces:**
- Consumes: nothing at runtime; `embeddings` is injected so tests pass a fake and never load a model.
- Produces:
  - `Chunk` dataclass: `index: int`, `text: str`, `start: int`, `strategy: str`, `parent_id: str`, `parent_text: str` (`""` when not applicable — Chroma metadata rejects `None`)
  - `ChunkResult` dataclass: `chunks: list[Chunk]`, `strategy: str`, `notes: list[str]`, `sections_detected: int`, `fell_back: bool`
  - `STRATEGIES: dict[str, StrategyInfo]` where `StrategyInfo` has `key`, `label`, `verdict`, `uses_size`, `uses_overlap`, `extra_control` — the UI renders its five cards straight from this
  - `chunk(text: str, *, strategy: str, size: int, overlap: int, embeddings=None, percentile: int = 95) -> ChunkResult`
  - `UnknownStrategyError(ValueError)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunkers.py`:

```python
"""Chunker tests.

Several of these assert that a strategy is *bad* in the specific way the deck
says it is. That is deliberate: the demo's teaching value depends on fixed-size
really shredding words, so it is worth a test.
"""

import pytest

from app.pipeline.chunkers import (
    STRATEGIES,
    UnknownStrategyError,
    chunk,
)

PROSE = (
    "Companies are the top level record in the system. "
    "Every company holds an address, operating hours and custom fields. "
    "People belong to one or more companies as contacts. "
    "A person record holds an address, notes and custom fields. "
) * 12

STRUCTURED = "\n".join(
    f"{n}. Section {n}\n" + ("Body text for this section. " * 14)
    for n in range(1, 7)
)


class FakeEmbeddings:
    """Deterministic stand-in for a real model.

    Sentences are embedded by length parity, which gives SemanticChunker a real
    distance signal to find breakpoints in without loading 90MB of weights.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if len(t) % 2 else [0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class TestRegistry:
    def test_exposes_exactly_the_decks_five_strategies(self):
        assert set(STRATEGIES) == {
            "fixed", "recursive", "structure", "semantic", "parent",
        }

    def test_every_strategy_carries_the_decks_verdict(self):
        for info in STRATEGIES.values():
            assert info.verdict, f"{info.key} has no verdict text"
            assert info.label

    def test_semantic_declares_it_ignores_size_and_overlap(self):
        # The UI disables those sliders based on these flags. If they were wrong
        # the room would see controls that silently do nothing.
        info = STRATEGIES["semantic"]
        assert info.uses_size is False
        assert info.uses_overlap is False
        assert info.extra_control == "percentile"

    def test_recursive_is_the_default_and_uses_both_sliders(self):
        info = STRATEGIES["recursive"]
        assert info.uses_size and info.uses_overlap


class TestFixedSize:
    def test_splits_words_in_half(self):
        """The deck calls this 'baseline only, splits mid sentence and mid word'.

        Asserting it proves the criticism on screen rather than just claiming it.
        """
        result = chunk(PROSE, strategy="fixed", size=120, overlap=0)
        # A chunk boundary that lands inside a word: chunk ends with a letter and
        # the next begins with one, with no whitespace between them.
        boundaries = [
            (a.text[-1], b.text[0])
            for a, b in zip(result.chunks, result.chunks[1:])
        ]
        assert any(x.isalpha() and y.isalpha() for x, y in boundaries)

    def test_respects_the_requested_size(self):
        result = chunk(PROSE, strategy="fixed", size=200, overlap=0)
        assert all(len(c.text) <= 200 for c in result.chunks)


class TestRecursive:
    def test_does_not_split_words(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        for a, b in zip(result.chunks, result.chunks[1:]):
            assert not (a.text[-1].isalpha() and b.text[0].isalpha())

    def test_indexes_are_sequential_from_zero(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        assert [c.index for c in result.chunks] == list(range(len(result.chunks)))

    def test_start_offsets_are_non_decreasing(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        starts = [c.start for c in result.chunks]
        assert starts == sorted(starts)

    def test_start_offsets_point_at_the_real_text(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        for c in result.chunks:
            assert PROSE[c.start:c.start + len(c.text)] == c.text


class TestStructureAware:
    def test_detects_numbered_sections(self):
        result = chunk(STRUCTURED, strategy="structure", size=700, overlap=100)
        assert result.sections_detected == 6
        assert result.fell_back is False

    def test_keeps_a_section_heading_with_its_body(self):
        result = chunk(STRUCTURED, strategy="structure", size=700, overlap=100)
        first = result.chunks[0].text
        assert first.startswith("1. Section 1")
        assert "Body text" in first

    def test_falls_back_to_recursive_without_structure(self):
        """An honest empty result beats one section holding the whole document."""
        result = chunk(PROSE, strategy="structure", size=200, overlap=20)
        assert result.fell_back is True
        assert result.sections_detected == 0
        assert any("recursive" in note.lower() for note in result.notes)
        assert len(result.chunks) > 1


class TestSemantic:
    def test_produces_chunks_without_size_or_overlap(self):
        result = chunk(
            STRUCTURED, strategy="semantic", size=700, overlap=100,
            embeddings=FakeEmbeddings(), percentile=50,
        )
        assert len(result.chunks) >= 1
        assert all(c.text.strip() for c in result.chunks)

    def test_requires_an_embeddings_object(self):
        with pytest.raises(ValueError, match="embeddings"):
            chunk(PROSE, strategy="semantic", size=700, overlap=100)

    def test_notes_that_the_sliders_do_not_apply(self):
        result = chunk(
            STRUCTURED, strategy="semantic", size=700, overlap=100,
            embeddings=FakeEmbeddings(), percentile=50,
        )
        assert any("embedding distance" in note.lower() for note in result.notes)


class TestParentDocument:
    def test_every_child_resolves_to_a_parent(self):
        result = chunk(PROSE, strategy="parent", size=200, overlap=20)
        assert result.chunks
        for c in result.chunks:
            assert c.parent_id
            assert c.parent_text
            assert c.text in c.parent_text

    def test_parents_are_larger_than_children(self):
        result = chunk(PROSE, strategy="parent", size=200, overlap=20)
        assert all(len(c.parent_text) > len(c.text) for c in result.chunks)

    def test_children_respect_the_child_size(self):
        result = chunk(PROSE, strategy="parent", size=200, overlap=20)
        assert all(len(c.text) <= 200 for c in result.chunks)

    def test_more_than_one_parent_for_long_input(self):
        result = chunk(PROSE, strategy="parent", size=200, overlap=20)
        assert len({c.parent_id for c in result.chunks}) > 1


class TestDispatch:
    def test_unknown_strategy_names_the_valid_options(self):
        with pytest.raises(UnknownStrategyError, match="recursive"):
            chunk(PROSE, strategy="nonsense", size=700, overlap=100)

    def test_parent_id_is_empty_string_not_none(self):
        # Chroma metadata rejects None, so absent values must be "".
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        assert all(c.parent_id == "" for c in result.chunks)

    def test_every_chunk_records_its_strategy(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        assert all(c.strategy == "recursive" for c in result.chunks)

    def test_empty_text_yields_no_chunks(self):
        assert chunk("", strategy="recursive", size=700, overlap=100).chunks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_chunkers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.chunkers'`

- [ ] **Step 3: Write the chunkers**

Create `app/pipeline/chunkers.py`:

```python
"""The five chunking strategies from Level 3 of the deck.

    "Five strategies, in order of effort"

      Fixed size       Every N characters, blindly
      Recursive        Try paragraph, then sentence, then word, until it fits
      Structure aware  Split on headings, HTML tags, code blocks
      Semantic         Embed sentences, cut where meaning shifts
      Parent document  Index small chunks, return their larger parent

This is the file most people open first, so it is written to be read: one
function per strategy, each docstring quoting the deck's verdict, and no
cleverness that needs unpicking.

All five go through chunk(), which returns a ChunkResult carrying the chunks
plus any notes the UI should show the room.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    index: int
    text: str
    start: int              # character offset into the source text
    strategy: str
    # Empty strings rather than None: Chroma metadata rejects null values.
    parent_id: str = ""
    parent_text: str = ""


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    strategy: str
    notes: list[str] = field(default_factory=list)
    sections_detected: int = 0
    fell_back: bool = False


@dataclass(frozen=True)
class StrategyInfo:
    """Everything the UI needs to render one strategy card.

    uses_size / uses_overlap drive whether the sliders are enabled. Getting
    these wrong would show the room controls that silently do nothing.
    """

    key: str
    label: str
    verdict: str            # quoted from the deck's Level 3 table
    uses_size: bool
    uses_overlap: bool
    extra_control: str = ""


class UnknownStrategyError(ValueError):
    """Raised when an unrecognised strategy key is requested."""


STRATEGIES: dict[str, StrategyInfo] = {
    "fixed": StrategyInfo(
        key="fixed",
        label="Fixed size",
        verdict="Baseline only. Splits mid sentence and mid word.",
        uses_size=True,
        uses_overlap=True,
    ),
    "recursive": StrategyInfo(
        key="recursive",
        label="Recursive",
        verdict="The right default. Respects natural boundaries.",
        uses_size=True,
        uses_overlap=True,
    ),
    "structure": StrategyInfo(
        key="structure",
        label="Structure aware",
        verdict="Best value when documents have real structure.",
        uses_size=True,
        uses_overlap=True,
    ),
    "semantic": StrategyInfo(
        key="semantic",
        label="Semantic",
        verdict="Slow and costs embeddings up front. Sometimes worth it.",
        uses_size=False,
        uses_overlap=False,
        extra_control="percentile",
    ),
    "parent": StrategyInfo(
        key="parent",
        label="Parent document",
        verdict="Best of both. Precise search, full context.",
        uses_size=True,
        uses_overlap=True,
    ),
}

# --------------------------------------------------------------------------
# Heading detection for the structure-aware strategy
# --------------------------------------------------------------------------

# Tried in order; the first pattern matching at least _MIN_SECTIONS times wins.
# Attendees upload arbitrary PDFs, so this cannot be tuned to one document.
_HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("markdown", re.compile(r"^#{1,6}[ \t]+\S", re.M)),
    ("numbered", re.compile(r"^[ \t]*\d+(?:\.\d+)*\.?[ \t]+\S", re.M)),
    ("symbol", re.compile(r"^[ \t]*[❖●▪◆■][ \t]*\S", re.M)),
    ("lettered", re.compile(r"^[ \t]*(?:[a-z]|[ivxIVX]{1,4})[).][ \t]+\S", re.M)),
    ("caps", re.compile(r"^[A-Z][A-Z0-9 &/,'\-]{6,80}$", re.M)),
]

# Fewer matches than this is noise, not structure.
_MIN_SECTIONS = 3

# Parents are this multiple of the child size in the parent-document strategy.
_PARENT_SIZE_MULTIPLIER = 5

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=_RECURSIVE_SEPARATORS,
    )


def _locate(text: str, needle: str, cursor: int) -> int:
    """Find where a chunk sits in the source text.

    LangChain's splitters return strings, not offsets, and we need offsets to
    attribute chunks to pages. Searching forward from a cursor keeps this linear
    and picks the right occurrence when text repeats. Overlap means a chunk can
    begin slightly behind the cursor, hence the small rewind.
    """
    if not needle:
        return cursor
    found = text.find(needle, max(0, cursor - len(needle)))
    if found == -1:
        found = text.find(needle)
    return cursor if found == -1 else found


def _assemble(pieces: list[str], text: str, strategy: str) -> list[Chunk]:
    """Turn a list of chunk strings into Chunks with indexes and offsets."""
    chunks: list[Chunk] = []
    cursor = 0
    for index, piece in enumerate(p for p in pieces if p.strip()):
        start = _locate(text, piece, cursor)
        chunks.append(Chunk(index=index, text=piece, start=start, strategy=strategy))
        cursor = start + 1
    # Re-index after filtering blanks so indexes stay contiguous.
    for position, chunk_obj in enumerate(chunks):
        chunk_obj.index = position
    return chunks


# --------------------------------------------------------------------------
# The five strategies
# --------------------------------------------------------------------------


def _chunk_fixed(text: str, size: int, overlap: int) -> ChunkResult:
    """Cut every N characters, blindly.

    Deck verdict: "Baseline only. Splits mid sentence and mid word."

    separator="" makes CharacterTextSplitter split on individual characters and
    then merge up to chunk_size, which is precisely the blind cut the deck warns
    about. It is here to be visibly worse than the others.
    """
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=overlap)
    return ChunkResult(
        chunks=_assemble(splitter.split_text(text), text, "fixed"),
        strategy="fixed",
        notes=["Cuts on character count alone, ignoring word and sentence boundaries."],
    )


def _chunk_recursive(text: str, size: int, overlap: int) -> ChunkResult:
    """Try paragraph, then sentence, then word, until the chunk fits.

    Deck verdict: "The right default. Respects natural boundaries."
    """
    pieces = _recursive_splitter(size, overlap).split_text(text)
    return ChunkResult(
        chunks=_assemble(pieces, text, "recursive"),
        strategy="recursive",
        notes=["Falls back through paragraph, line, sentence, word, character."],
    )


def _detect_sections(text: str) -> tuple[str, list[int]]:
    """Find heading offsets using the first pattern that matches often enough.

    Returns (pattern_name, offsets). An empty offsets list means no structure
    was found, which is a legitimate answer for a flat document.
    """
    for name, pattern in _HEADING_PATTERNS:
        offsets = [m.start() for m in pattern.finditer(text)]
        if len(offsets) >= _MIN_SECTIONS:
            return name, offsets
    return "", []


def _chunk_structure(text: str, size: int, overlap: int) -> ChunkResult:
    """Split on the document's own headings, then recursively within sections.

    Deck verdict: "Best value when documents have real structure."

    When no structure is found this degrades to recursive and says so. That
    honesty matters: one section containing the whole document would look like
    success while quietly ruining retrieval.
    """
    pattern_name, offsets = _detect_sections(text)

    if not offsets:
        fallback = _chunk_recursive(text, size, overlap)
        return ChunkResult(
            chunks=[Chunk(**{**c.__dict__, "strategy": "structure"}) for c in fallback.chunks],
            strategy="structure",
            notes=["No document structure detected; fell back to recursive splitting."],
            sections_detected=0,
            fell_back=True,
        )

    # Slice the document at heading offsets, keeping each heading with its body.
    bounds = offsets + [len(text)]
    sections = [text[bounds[i]:bounds[i + 1]] for i in range(len(offsets))]

    splitter = _recursive_splitter(size, overlap)
    pieces: list[str] = []
    for section in sections:
        pieces.extend(splitter.split_text(section))

    return ChunkResult(
        chunks=_assemble(pieces, text, "structure"),
        strategy="structure",
        notes=[f"Detected {len(offsets)} sections using the {pattern_name} heading pattern."],
        sections_detected=len(offsets),
    )


def _chunk_semantic(text: str, embeddings, percentile: int) -> ChunkResult:
    """Embed sentences and cut where meaning shifts.

    Deck verdict: "Slow and costs embeddings up front. Sometimes worth it."

    Note for the room: this uses an *embeddings* model, not an LLM. Sentences are
    embedded, consecutive pairs compared by cosine distance, and a cut made where
    the distance exceeds the chosen percentile. Nothing generates text.

    Because breakpoints are data-driven, chunk size and overlap do not apply.
    """
    if embeddings is None:
        raise ValueError(
            "Semantic chunking needs an embeddings object -- it works by "
            "comparing sentence embeddings, so there is nothing to compare without one."
        )

    # Imported lazily: langchain_experimental pulls a heavy dependency tree, and
    # only this one strategy needs it.
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )
    return ChunkResult(
        chunks=_assemble(splitter.split_text(text), text, "semantic"),
        strategy="semantic",
        notes=[
            "Cut points come from embedding distance between neighbouring "
            f"sentences at the {percentile}th percentile, not a character budget.",
            "Uses the embedding model, not an LLM. No text is generated.",
        ],
    )


def _chunk_parent(text: str, size: int, overlap: int) -> ChunkResult:
    """Index small children, keep their larger parent for later retrieval.

    Deck verdict: "Best of both. Precise search, full context."

    Scope note: this is really a *retrieval* pattern. Here we build and display
    the child-to-parent structure; the payoff (a small chunk matches, the large
    parent is returned) arrives with the query build. The UI says so rather than
    implying a capability that is not wired up yet.
    """
    parent_size = size * _PARENT_SIZE_MULTIPLIER
    parents = _recursive_splitter(parent_size, 0).split_text(text)
    child_splitter = _recursive_splitter(size, overlap)

    chunks: list[Chunk] = []
    cursor = 0
    index = 0
    for parent_number, parent in enumerate(parents):
        parent_id = f"p{parent_number:04d}"
        for child in child_splitter.split_text(parent):
            if not child.strip():
                continue
            start = _locate(text, child, cursor)
            chunks.append(
                Chunk(
                    index=index,
                    text=child,
                    start=start,
                    strategy="parent",
                    parent_id=parent_id,
                    parent_text=parent,
                )
            )
            cursor = start + 1
            index += 1

    return ChunkResult(
        chunks=chunks,
        strategy="parent",
        notes=[
            f"{len(parents)} parents of ~{parent_size} chars, each split into "
            f"~{size}-char children. Children are what get embedded.",
            "The retrieval payoff needs the query build; only the structure is shown here.",
        ],
    )


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def chunk(
    text: str,
    *,
    strategy: str,
    size: int,
    overlap: int,
    embeddings=None,
    percentile: int = 95,
) -> ChunkResult:
    """Split text using one of the five strategies.

    embeddings is injected rather than constructed here, so tests can pass a
    lightweight fake and never load 90MB of model weights.
    """
    if strategy not in STRATEGIES:
        raise UnknownStrategyError(
            f"Unknown strategy {strategy!r}. Valid options: {', '.join(sorted(STRATEGIES))}."
        )

    if not text.strip():
        return ChunkResult(chunks=[], strategy=strategy, notes=["Document is empty."])

    if strategy == "fixed":
        return _chunk_fixed(text, size, overlap)
    if strategy == "recursive":
        return _chunk_recursive(text, size, overlap)
    if strategy == "structure":
        return _chunk_structure(text, size, overlap)
    if strategy == "semantic":
        return _chunk_semantic(text, embeddings, percentile)
    return _chunk_parent(text, size, overlap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_chunkers.py -v`
Expected: PASS, 24 passed

If `test_produces_chunks_without_size_or_overlap` errors inside `SemanticChunker`, the fake's vectors are being rejected for having 2 dimensions. Widen `FakeEmbeddings` to return 8-dimensional vectors (`[1.0] + [0.0] * 7` / `[0.0] * 7 + [1.0]`) rather than changing the production code.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/chunkers.py tests/test_chunkers.py
git commit -m "feat: the five chunking strategies from deck Level 3

One function per strategy, each docstring quoting the deck's verdict. Tests
assert fixed-size really does split words in half, so the criticism the slide
makes is demonstrable rather than merely claimed.

Structure-aware tries five heading patterns in order and degrades to recursive
with a visible note when a document has none -- an honest fallback beats one
section holding the whole document. STRATEGIES drives the UI cards, including
which sliders apply, so semantic cannot show controls that do nothing."
```

---

## Task 4: Embedder

**Files:**
- Create: `app/pipeline/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Consumes: `app.config.settings` (Task 1).
- Produces:
  - `build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings` — cached per process
  - `embed_batched(embeddings, texts: list[str], batch_size: int, on_progress: Callable[[int, int], None] | None = None) -> list[list[float]]`
  - `vector_norm(vector: list[float]) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_embedder.py`:

```python
"""Embedder tests.

build_embeddings loads a real 90MB model, so only one test touches it and it is
marked slow. Everything else uses a fake, because batching and progress
reporting are our logic, not the model's.
"""

import pytest

from app.pipeline.embedder import build_embeddings, embed_batched, vector_norm


class CountingEmbeddings:
    """Records how it was called so batching can be asserted."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_batches_at_the_requested_size():
    fake = CountingEmbeddings()
    embed_batched(fake, [f"t{i}" for i in range(10)], batch_size=4)
    assert fake.batch_sizes == [4, 4, 2]


def test_returns_one_vector_per_text_in_order():
    vectors = embed_batched(CountingEmbeddings(), ["a", "b", "c"], batch_size=2)
    assert len(vectors) == 3


def test_progress_is_reported_per_batch_and_reaches_the_total():
    seen: list[tuple[int, int]] = []
    embed_batched(
        CountingEmbeddings(), [f"t{i}" for i in range(10)],
        batch_size=4, on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(4, 10), (8, 10), (10, 10)]


def test_empty_input_makes_no_calls():
    fake = CountingEmbeddings()
    assert embed_batched(fake, [], batch_size=4) == []
    assert fake.batch_sizes == []


def test_vector_norm_of_a_unit_vector_is_one():
    assert vector_norm([0.6, 0.8]) == pytest.approx(1.0)


@pytest.mark.slow
def test_real_model_returns_normalised_384_dim_vectors():
    """The deck says normalise once at write time so cosine becomes a dot
    product. This is the test that proves we actually did."""
    embeddings = build_embeddings()
    vectors = embed_batched(embeddings, ["annual leave policy"], batch_size=8)
    assert len(vectors[0]) == 384
    assert vector_norm(vectors[0]) == pytest.approx(1.0, abs=1e-3)
```

Register the marker — create `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: loads the real embedding model (deselect with -m "not slow")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_embedder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.embedder'`

- [ ] **Step 3: Write the embedder**

Create `app/pipeline/embedder.py`:

```python
"""Turning text into vectors -- step 2 of the pipeline.

Level 2 of the deck: an embedding model is a function, text in and a fixed-length
list of floats out, trained so that related text lands in nearby positions.

Two deck instructions are enforced here:

  - Normalise once at write time, so cosine similarity becomes a dot product
    (gotcha #02: unnormalised vectors let long documents dominate).
  - Pin the model name in config, never as a default argument (the indexing and
    querying paths must use byte-identical models).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=2)
def build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Load the embedding model, cached for the process lifetime.

    Weights are baked into the image under HF_HOME, so this never reaches the
    network -- the workshop is presented offline. First call costs a few seconds
    of load time; subsequent calls are free thanks to the cache.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name or settings.embed_model,
        model_kwargs={"device": "cpu"},
        # normalize_embeddings is the deck's "normalise once at write time".
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_batched(
    embeddings,
    texts: list[str],
    batch_size: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """Embed texts in batches, reporting progress after each one.

    Progress here is genuine: each callback fires after a batch has actually been
    encoded. That matters for the demo -- a progress bar that moves because of a
    timer teaches the room nothing.
    """
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(embeddings.embed_documents(batch))
        if on_progress is not None:
            on_progress(min(start + len(batch), total), total)
    return vectors


def vector_norm(vector: list[float]) -> float:
    """Euclidean length of a vector.

    Shown in the collection preview: seeing 1.000 next to every record is how the
    room confirms normalisation happened rather than taking it on trust.
    """
    return math.sqrt(sum(component * component for component in vector))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_embedder.py -v`
Expected: PASS, 6 passed (the slow test loads the real model; allow ~20s)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/embedder.py tests/test_embedder.py pytest.ini
git commit -m "feat: batched embedding with genuine per-batch progress

Vectors are L2-normalised at write time, per the deck's instruction that cosine
then reduces to a dot product -- and a test asserts the norm really is 1.0
rather than trusting the flag. Progress callbacks fire after real work, never
on a timer."
```

---

## Task 5: Chroma store with delete-before-write

**Files:**
- Create: `app/pipeline/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Chunk` (Task 3), `app.config.settings` (Task 1).
- Produces:
  - `get_client(host=None, port=None)`, `get_collection(client, name=None)`
  - `write_chunks(collection, *, chunks, vectors, doc_id, source, size, overlap, embed_model, page_for_offset) -> int`
  - `read_records(collection, offset=0, limit=25, preview_dims=8) -> dict`
  - `count_records(collection) -> int`, `drop_collection(client, name=None) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
"""Store tests, run against an in-process ephemeral Chroma client.

No server needed, so the suite stays fast and hermetic while still exercising
the real Chroma API rather than a mock of it.
"""

import chromadb
import pytest

from app.pipeline.chunkers import Chunk
from app.pipeline.store import (
    count_records,
    get_collection,
    read_records,
    write_chunks,
)

DOC_ID = "a" * 64
OTHER_DOC_ID = "b" * 64


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return get_collection(client, "test-collection")


def make_chunks(count: int, strategy: str = "recursive") -> list[Chunk]:
    return [
        Chunk(index=i, text=f"chunk number {i}", start=i * 100, strategy=strategy)
        for i in range(count)
    ]


def make_vectors(count: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in range(count)]


def write(collection, chunks, *, doc_id=DOC_ID, size=700, overlap=100):
    return write_chunks(
        collection,
        chunks=chunks,
        vectors=make_vectors(len(chunks)),
        doc_id=doc_id,
        source="handbook.pdf",
        size=size,
        overlap=overlap,
        embed_model="all-MiniLM-L6-v2",
        page_for_offset=lambda offset: offset // 100 + 1,
    )


class TestWrite:
    def test_writes_every_chunk(self, collection):
        assert write(collection, make_chunks(5)) == 5
        assert count_records(collection) == 5

    def test_reingesting_the_same_document_does_not_duplicate(self, collection):
        write(collection, make_chunks(5))
        write(collection, make_chunks(5))
        assert count_records(collection) == 5

    def test_shrinking_the_chunk_count_leaves_no_orphans(self, collection):
        """The bug this exists to prevent.

        Ingest at size 700 -> 5 chunks. Re-ingest at 1500 -> 2 chunks. Without
        delete-before-write, chunks 2-4 from the first run survive as orphans and
        silently pollute every later result.
        """
        write(collection, make_chunks(5), size=700)
        write(collection, make_chunks(2), size=1500)
        assert count_records(collection) == 2

    def test_a_different_strategy_coexists(self, collection):
        # Comparing two strategies side by side is intentional, so only a re-run
        # of the *same* strategy replaces anything.
        write(collection, make_chunks(3, "recursive"))
        write(collection, make_chunks(4, "fixed"))
        assert count_records(collection) == 7

    def test_a_different_document_coexists(self, collection):
        write(collection, make_chunks(3), doc_id=DOC_ID)
        write(collection, make_chunks(3), doc_id=OTHER_DOC_ID)
        assert count_records(collection) == 6

    def test_writing_no_chunks_is_a_no_op(self, collection):
        assert write(collection, []) == 0
        assert count_records(collection) == 0


class TestMetadata:
    def test_every_record_records_the_embedding_model(self, collection):
        # The deck: store the model name alongside every vector so you can tell
        # what is stale.
        write(collection, make_chunks(3))
        for meta in read_records(collection)["records"]:
            assert meta["metadata"]["embed_model"] == "all-MiniLM-L6-v2"

    def test_carries_the_full_metadata_set(self, collection):
        write(collection, make_chunks(1))
        meta = read_records(collection)["records"][0]["metadata"]
        for key in (
            "doc_id", "source", "page", "chunk_index", "strategy",
            "chunk_size", "overlap", "embed_model", "char_count", "parent_id",
        ):
            assert key in meta, f"missing metadata key {key}"

    def test_page_is_derived_from_the_offset(self, collection):
        write(collection, make_chunks(3))
        pages = sorted(r["metadata"]["page"] for r in read_records(collection)["records"])
        assert pages == [1, 2, 3]

    def test_absent_parent_id_is_an_empty_string(self, collection):
        # Chroma rejects None in metadata; "" is the sentinel.
        write(collection, make_chunks(1))
        assert read_records(collection)["records"][0]["metadata"]["parent_id"] == ""


class TestRead:
    def test_returns_a_vector_preview_and_norm(self, collection):
        write(collection, make_chunks(1))
        record = read_records(collection, preview_dims=2)["records"][0]
        assert len(record["vector_preview"]) == 2
        assert record["vector_norm"] == pytest.approx(1.0)
        assert record["dims"] == 3

    def test_paginates(self, collection):
        write(collection, make_chunks(10))
        page = read_records(collection, offset=4, limit=3)
        assert len(page["records"]) == 3
        assert page["total"] == 10

    def test_includes_the_document_text(self, collection):
        write(collection, make_chunks(1))
        assert read_records(collection)["records"][0]["text"] == "chunk number 0"

    def test_empty_collection_reads_cleanly(self, collection):
        page = read_records(collection)
        assert page["records"] == []
        assert page["total"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.store'`

- [ ] **Step 3: Write the store**

Create `app/pipeline/store.py`:

```python
"""Writing vectors to ChromaDB -- step 3 of the pipeline.

Level 4 of the deck: "One record, four fields" -- an id, a vector, the original
text, and metadata. That is the whole data model, and it is what this module
writes.

We use the raw chromadb client rather than langchain-chroma, deliberately.
langchain-chroma computes embeddings internally, which would rule out both
per-batch progress reporting and showing the room the actual vectors. LangChain
still supplies the loader, the splitters and the embedding wrapper. Recorded in
CLAUDE.md so the choice is not mistaken for an oversight.
"""

from __future__ import annotations

from collections.abc import Callable

import chromadb

from app.config import settings
from app.pipeline.chunkers import Chunk
from app.pipeline.embedder import vector_norm


def get_client(host: str | None = None, port: int | None = None):
    """Connect to the Chroma server defined in config."""
    return chromadb.HttpClient(
        host=host or settings.chroma_host,
        port=port or settings.chroma_port,
    )


def get_collection(client, name: str | None = None):
    """Fetch or create the workshop collection.

    hnsw:space=cosine matches the deck's defaults slide. Since vectors are
    normalised at write time, cosine and dot product agree -- but stating cosine
    explicitly documents the intent.
    """
    return client.get_or_create_collection(
        name=name or settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def write_chunks(
    collection,
    *,
    chunks: list[Chunk],
    vectors: list[list[float]],
    doc_id: str,
    source: str,
    size: int,
    overlap: int,
    embed_model: str,
    page_for_offset: Callable[[int], int],
) -> int:
    """Write chunks and their vectors, replacing any previous run.

    Content-hash ids make a same-parameters re-run a clean overwrite. That alone
    is not enough, though: ingesting at size 700 yields far more chunks than at
    1500, so the tail of the earlier run would survive as orphans and quietly
    pollute every later result. Hence the delete first, scoped to this
    (doc_id, strategy) pair so a *different* strategy can sit alongside for
    comparison.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(
            f"{len(chunks)} chunks but {len(vectors)} vectors -- these must match."
        )

    strategy = chunks[0].strategy

    collection.delete(where={"$and": [{"doc_id": doc_id}, {"strategy": strategy}]})

    collection.add(
        ids=[f"{doc_id[:12]}-{strategy}-{c.index}" for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": doc_id,
                "source": source,
                "page": page_for_offset(c.start),
                "chunk_index": c.index,
                "strategy": c.strategy,
                "chunk_size": size,
                "overlap": overlap,
                # The deck: store the model name alongside every vector so you
                # can tell what is stale when you change models.
                "embed_model": embed_model,
                "char_count": len(c.text),
                # "" not None -- Chroma metadata rejects null values.
                "parent_id": c.parent_id,
            }
            for c in chunks
        ],
    )
    return len(chunks)


def count_records(collection) -> int:
    return collection.count()


def read_records(
    collection,
    offset: int = 0,
    limit: int = 25,
    preview_dims: int = 8,
) -> dict:
    """Read a page of stored records for the collection browser.

    Returns a vector preview and its norm rather than 384 floats: the point is
    for the room to see that a record really is numbers, and that the norm is
    1.0, without a wall of digits.
    """
    total = collection.count()
    if total == 0:
        return {"records": [], "total": 0, "offset": offset, "limit": limit}

    result = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=limit,
        offset=offset,
    )

    records = []
    for position, record_id in enumerate(result["ids"]):
        vector = list(result["embeddings"][position])
        records.append({
            "id": record_id,
            "text": result["documents"][position],
            "metadata": result["metadatas"][position],
            "dims": len(vector),
            "vector_preview": [round(v, 4) for v in vector[:preview_dims]],
            "vector_norm": round(vector_norm(vector), 4),
        })

    return {"records": records, "total": total, "offset": offset, "limit": limit}


def drop_collection(client, name: str | None = None) -> None:
    """Delete the collection outright, for the UI's Reset control."""
    try:
        client.delete_collection(name or settings.chroma_collection)
    except Exception:  # noqa: BLE001 - already absent is success for a reset
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_store.py -v`
Expected: PASS, 14 passed

If `read_records` raises on `result["embeddings"]` being `None`, this Chroma version needs `include=["documents", "metadatas", "embeddings"]` passed as an `IncludeEnum` — check `collection.get.__doc__` in the installed version and adapt the include list, keeping the returned shape identical.

- [ ] **Step 5: Run the whole suite and commit**

```bash
docker compose run --rm app pytest -v -m "not slow"
docker compose run --rm app pytest -v -m slow
git add app/pipeline/store.py tests/test_store.py
git commit -m "feat: Chroma writes with delete-before-write idempotency

Content-hash ids make a same-parameters re-run an overwrite, but changing chunk
size shrinks the chunk count and would leave the earlier run's tail behind as
orphans. Records are deleted per (doc_id, strategy) before writing, with a test
pinning exactly that scenario.

Records are keyed by strategy too, so two strategies can be compared side by
side in the browser while a re-run of one replaces only itself."
```

---

## Task 6: Session state with refresh-safe persistence

**Files:**
- Create: `app/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: `app.config.settings` (Task 1), `Chunk` (Task 3).
- Produces:
  - `SessionState` dataclass: `session_id: str`, `created_at: str`, `upload: dict | None`, `chunking: dict | None`, `embedding: dict | None`, `chunks: list[Chunk]`, `pdf_path: str`, `page_offsets: list[tuple[int, int]]`; methods `unlocked_step() -> int`, `page_for_offset(offset: int) -> int`, `to_json() -> dict`
  - `SessionStore` class with `get_or_create(session_id: str | None) -> SessionState`, `save(state) -> None`, `reset(session_id: str) -> SessionState`
  - `SessionState.from_json(data: dict) -> SessionState`

- [ ] **Step 1: Write the failing test**

Create `tests/test_session.py`:

```python
"""Session tests.

The persistence requirement is not academic: a browser refresh mid-demo must not
send the presenter back to step 1 in front of a room.
"""

from app.pipeline.chunkers import Chunk
from app.session import SessionState, SessionStore


def make_store(tmp_path) -> SessionStore:
    return SessionStore(data_dir=tmp_path)


class TestUnlocking:
    def test_a_fresh_session_only_unlocks_upload(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        assert state.unlocked_step() == 1

    def test_upload_unlocks_configuration(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf", "doc_id": "x" * 64, "page_count": 3}
        assert state.unlocked_step() == 2

    def test_chunking_unlocks_embedding(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf"}
        state.chunking = {"strategy": "recursive", "chunk_count": 12}
        assert state.unlocked_step() == 4

    def test_embedding_unlocks_the_browser(self, tmp_path):
        state = make_store(tmp_path).get_or_create(None)
        state.upload = {"filename": "a.pdf"}
        state.chunking = {"strategy": "recursive", "chunk_count": 12}
        state.embedding = {"vectors_written": 12}
        assert state.unlocked_step() == 5


class TestPersistence:
    def test_a_saved_session_survives_a_new_store(self, tmp_path):
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

    def test_page_offsets_round_trip_as_tuples(self, tmp_path):
        # JSON turns tuples into lists; page_for_offset must still work.
        store = make_store(tmp_path)
        state = store.get_or_create(None)
        state.page_offsets = [(0, 1), (500, 2), (900, 3)]
        store.save(state)

        rehydrated = make_store(tmp_path).get_or_create(state.session_id)
        assert rehydrated.page_for_offset(600) == 2
        assert rehydrated.page_for_offset(0) == 1

    def test_an_unknown_session_id_yields_a_fresh_session(self, tmp_path):
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

    def test_to_json_omits_chunk_text_bodies(self, tmp_path):
        """The client gets counts and metadata, not 423 chunk bodies twice."""
        state = make_store(tmp_path).get_or_create(None)
        state.chunks = [Chunk(index=0, text="x" * 700, start=0, strategy="fixed")]
        assert "x" * 700 not in str(state.to_json())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.session'`

- [ ] **Step 3: Write the session module**

Create `app/session.py`:

```python
"""Per-session state and its on-disk mirror.

This is a single-presenter demo, so state lives in memory. It is also mirrored to
JSON, for one specific reason: a browser refresh mid-demo must not drop the
presenter back to step 1 in front of a room.

The step-unlock rule lives here rather than in the frontend, so the server is the
single authority on what is reachable.
"""

from __future__ import annotations

import bisect
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.pipeline.chunkers import Chunk


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
        together: once a document is loaded, choosing a strategy and running it
        are one interaction.
        """
        if self.embedding:
            return 5
        if self.chunking:
            return 4
        if self.upload:
            return 2
        return 1

    def page_for_offset(self, offset: int) -> int:
        """Page number for a character offset -- see loader.page_for_offset.

        Duplicated here because the offsets survive a JSON round-trip while the
        LoadResult object does not.
        """
        if not self.page_offsets:
            return 1
        starts = [start for start, _ in self.page_offsets]
        index = max(bisect.bisect_right(starts, offset) - 1, 0)
        return self.page_offsets[index][1]

    def to_json(self) -> dict:
        """The client-facing view.

        Chunk bodies are excluded: they stream over SSE during step 3, and
        shipping 423 of them again in a status payload would be waste.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_session.py -v`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
git add app/session.py tests/test_session.py
git commit -m "feat: session state with refresh-safe JSON persistence

A browser refresh mid-demo must not drop the presenter back to step 1 in front
of a room, so each completed stage is mirrored to disk and rehydrated on reload.
Writes go through a temp file and rename so a crash cannot leave an unparseable
session behind.

The unlock rule lives server-side, making the server the single authority on
what is reachable rather than trusting the DOM."
```

---

## Task 7: API routes and SSE job plumbing

**Files:**
- Create: `app/jobs.py`
- Modify: `app/main.py` (replace wholesale — the health route is retained inside)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  - `app.jobs.Job` dataclass: `job_id`, `queue`, `status`, `events`, `error`
  - `app.jobs.JobRegistry` with `create() -> Job`, `get(job_id) -> Job | None`, `publish(job, event: dict) -> None` (thread-safe), `finish(job, status, error="") -> None`
  - `app.jobs.sse_format(event: dict) -> str`
  - Routes: `GET /`, `GET /api/health`, `GET /api/config`, `POST /api/upload`, `POST /api/use-local`, `POST /api/chunk`, `POST /api/embed`, `GET /api/events/{job_id}`, `GET /api/status/{job_id}`, `GET /api/collection`, `POST /api/reset`

**On progress honesty:** LangChain's splitters have no streaming API — they return every chunk at once. So step 3 emits a `stage` event while splitting runs in a worker thread, then emits one `chunk` event per result as fast as the client consumes them. The *rendering* streams; the splitting does not. Embedding progress is genuinely incremental, one event per encoded batch. No artificial delays anywhere.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
"""API tests using FastAPI's TestClient.

Chroma is stubbed with an ephemeral in-process client, so these run without a
server. The SSE endpoints are tested by draining the job queue directly, which
is far more reliable than parsing a streaming response under test.
"""

import json
import time

import chromadb
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.jobs import JobRegistry, sse_format


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point sessions at tmp_path and Chroma at an ephemeral client."""
    from app.session import SessionStore

    monkeypatch.setattr(main_module, "store", SessionStore(data_dir=tmp_path))
    client = chromadb.EphemeralClient()
    monkeypatch.setattr(main_module, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(main_module.settings, "data_dir", tmp_path, raising=False)
    yield


@pytest.fixture
def client():
    return TestClient(main_module.app)


class TestPage:
    def test_the_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_the_page_lists_all_five_strategies(self, client):
        body = client.get("/").text
        for label in ("Fixed size", "Recursive", "Structure aware", "Semantic", "Parent document"):
            assert label in body

    def test_config_exposes_the_deck_defaults(self, client):
        data = client.get("/api/config").json()
        assert data["default_chunk_size"] == 700
        assert data["default_chunk_overlap"] == 100
        assert data["default_strategy"] == "recursive"
        assert len(data["strategies"]) == 5

    def test_config_reports_whether_a_local_document_exists(self, client):
        # Unset in tests, so the button must not be offered.
        assert client.get("/api/config").json()["has_local_pdf"] is False


class TestUpload:
    def test_rejects_a_non_pdf(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower()

    def test_rejects_a_file_over_the_size_limit(self, client, monkeypatch):
        monkeypatch.setattr(main_module.settings, "max_upload_mb", 0, raising=False)
        response = client.post(
            "/api/upload",
            files={"file": ("big.pdf", b"%PDF-1.4" + b"x" * 2048, "application/pdf")},
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_accepts_a_real_pdf_and_unlocks_step_two(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("handbook.pdf", handle.read(), "application/pdf")},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["upload"]["page_count"] == 5
        assert body["unlocked_step"] == 2
        assert "boilerplate_lines_removed" in body["upload"]

    def test_reports_a_scanned_pdf_clearly(self, client, tmp_path):
        from reportlab.pdfgen import canvas

        path = tmp_path / "scanned.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.showPage()
        pdf.save()

        with open(path, "rb") as handle:
            response = client.post(
                "/api/upload",
                files={"file": ("scanned.pdf", handle.read(), "application/pdf")},
            )
        assert response.status_code == 400
        assert "text layer" in response.json()["detail"]

    def test_use_local_is_unavailable_when_unconfigured(self, client):
        assert client.post("/api/use-local").status_code == 404


class TestChunkAndEmbed:
    def upload(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            return client.post(
                "/api/upload",
                files={"file": ("handbook.pdf", handle.read(), "application/pdf")},
            ).json()

    def test_chunking_requires_an_upload_first(self, client):
        response = client.post("/api/chunk", json={"strategy": "recursive"})
        assert response.status_code == 409
        assert "upload" in response.json()["detail"].lower()

    def test_chunking_rejects_an_unknown_strategy(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post("/api/chunk", json={"strategy": "nonsense"})
        assert response.status_code == 400

    def test_chunking_returns_a_job_id(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post(
            "/api/chunk",
            json={"strategy": "recursive", "size": 200, "overlap": 20},
        )
        assert response.status_code == 202
        assert response.json()["job_id"]

    def test_status_reports_a_finished_chunk_job(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        job_id = client.post(
            "/api/chunk", json={"strategy": "recursive", "size": 200, "overlap": 20}
        ).json()["job_id"]

        # The route uses a bare asyncio.create_task, which TestClient does NOT
        # guarantee to drain before returning. Poll until terminal rather than
        # assuming, so this cannot pass locally and flake in CI.
        status = None
        for _ in range(100):
            status = client.get(f"/api/status/{job_id}").json()
            if status["status"] != "running":
                break
            time.sleep(0.05)

        assert status["status"] == "done", f"job did not finish: {status}"
        assert any(e["type"] == "chunk" for e in status["events"])
        assert any(e["type"] == "done" for e in status["events"])

    def test_embedding_requires_chunking_first(self, client, structured_pdf):
        self.upload(client, structured_pdf)
        response = client.post("/api/embed")
        assert response.status_code == 409

    def test_status_for_an_unknown_job_is_404(self, client):
        assert client.get("/api/status/nope").status_code == 404


class TestCollectionAndReset:
    def test_the_collection_reads_empty(self, client):
        body = client.get("/api/collection").json()
        assert body["total"] == 0
        assert body["records"] == []

    def test_reset_returns_a_clean_session(self, client, structured_pdf):
        with open(structured_pdf, "rb") as handle:
            client.post(
                "/api/upload",
                files={"file": ("h.pdf", handle.read(), "application/pdf")},
            )
        body = client.post("/api/reset", json={"drop_collection": True}).json()
        assert body["unlocked_step"] == 1
        assert body["upload"] is None


class TestSseFormatting:
    def test_formats_as_a_data_line_pair(self):
        assert sse_format({"type": "done"}) == 'data: {"type": "done"}\n\n'

    def test_payload_is_valid_json(self):
        raw = sse_format({"type": "chunk", "index": 3})
        assert json.loads(raw.removeprefix("data: ").strip())["index"] == 3


class TestJobRegistry:
    def test_publish_records_events_for_the_polling_fallback(self):
        registry = JobRegistry()
        job = registry.create()
        registry.publish(job, {"type": "chunk", "index": 0})
        assert job.events == [{"type": "chunk", "index": 0}]

    def test_finish_sets_the_terminal_status(self):
        registry = JobRegistry()
        job = registry.create()
        registry.finish(job, "error", "boom")
        assert job.status == "error"
        assert job.error == "boom"

    def test_unknown_job_lookup_returns_none(self):
        assert JobRegistry().get("absent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm app pytest tests/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'JobRegistry' from 'app.jobs'`

- [ ] **Step 3: Write the job registry**

Create `app/jobs.py`:

```python
"""Background jobs and their event streams.

Deliberately small: an asyncio task per job and a queue per job. The spec
rejected Celery and Redis for this, because four containers and a broker are a
lot of failure surface for a one-document corpus, and task serialisation would
sit between a reader and the pipeline code they came to read.

Every job keeps its events in a list as well as a queue, so a dropped SSE
connection can fall back to polling without losing history.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    job_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    status: str = "running"          # running | done | error
    events: list[dict] = field(default_factory=list)
    error: str = ""


def sse_format(event: dict) -> str:
    """Render one event as a Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


class JobRegistry:
    """Tracks in-flight jobs for one process."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        job = Job(job_id=uuid.uuid4().hex)
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def publish(self, job: Job, event: dict) -> None:
        """Record an event and push it to any listening SSE stream.

        Safe to call from a worker thread: put_nowait on an unbounded queue does
        not block, and the history list append is atomic under the GIL. Pipeline
        work runs via asyncio.to_thread so it cannot stall the event loop, which
        would otherwise leave SSE frames unflushed.
        """
        job.events.append(event)
        job.queue.put_nowait(event)

    def finish(self, job: Job, status: str, error: str = "") -> None:
        job.status = status
        job.error = error
        self.publish(job, {"type": status, "error": error} if error else {"type": status})
        # Sentinel so an SSE consumer knows to close rather than hang.
        job.queue.put_nowait(None)


registry = JobRegistry()
```

- [ ] **Step 4: Write the routes**

Replace `app/main.py` entirely:

```python
"""FastAPI application: routes for the five-step ingestion page.

Route shape follows the steps. Each mutating route updates the session and
returns the session view, so the client always learns which step is unlocked from
the server rather than deciding for itself.

Long-running work goes through app.jobs: the route returns a job id immediately
and the client subscribes to /api/events/{job_id}.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import chromadb
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.jobs import registry, sse_format
from app.pipeline import store as vector_store
from app.pipeline.chunkers import STRATEGIES, UnknownStrategyError, chunk
from app.pipeline.embedder import build_embeddings, embed_batched
from app.pipeline.loader import EmptyDocumentError, load_pdf
from app.pipeline.store import get_client
from app.session import store

BASE_DIR = Path(__file__).parent

app = FastAPI(title="RAG Ingestion Pipeline", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

SESSION_COOKIE = "rag_session"


def _session(request: Request):
    return store.get_or_create(request.cookies.get(SESSION_COOKIE))


def _with_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")


def _collection():
    """Fetch the working collection, converting connection failure into a 503."""
    try:
        return vector_store.get_collection(get_client())
    except Exception as exc:  # noqa: BLE001 - becomes a retry banner in the UI
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB is unreachable ({type(exc).__name__}). "
            "Check that the chromadb container is running, then retry.",
        ) from exc


# ---------------------------------------------------------------- page & config


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    state = _session(request)
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "strategies": list(STRATEGIES.values()),
            "settings": settings,
            "state": state.to_json(),
            "has_local_pdf": settings.local_pdf is not None,
        },
    )
    _with_session_cookie(response, state.session_id)
    return response


@app.get("/api/config")
def config() -> dict:
    """Defaults and strategy metadata, so the client hardcodes nothing."""
    return {
        "default_chunk_size": settings.default_chunk_size,
        "default_chunk_overlap": settings.default_chunk_overlap,
        "default_strategy": settings.default_strategy,
        "semantic_percentile": settings.semantic_percentile,
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
        "max_upload_mb": settings.max_upload_mb,
        "has_local_pdf": settings.local_pdf is not None,
        "strategies": [
            {
                "key": info.key,
                "label": info.label,
                "verdict": info.verdict,
                "uses_size": info.uses_size,
                "uses_overlap": info.uses_overlap,
                "extra_control": info.extra_control,
            }
            for info in STRATEGIES.values()
        ],
    }


@app.get("/api/health")
def health() -> dict:
    chroma_ok, detail = False, ""
    try:
        chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        ).heartbeat()
        chroma_ok = True
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "status": "ok" if chroma_ok else "degraded",
        "chroma": {"reachable": chroma_ok, "detail": detail},
        "embed_model": settings.embed_model,
        "embed_dims": settings.embed_dims,
    }


# ------------------------------------------------------------------ step 1: load


def _ingest_path(state, path: Path, display_name: str) -> dict:
    """Load a PDF into the session, or raise a 400 the UI can display."""
    try:
        result = load_pdf(path)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - corrupt or encrypted PDF
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {display_name}: {type(exc).__name__}. "
            "If the file is password protected, remove the protection first.",
        ) from exc

    state.pdf_path = str(path)
    state.page_offsets = result.page_offsets
    state.upload = {
        "filename": display_name,
        "doc_id": result.doc_id,
        "page_count": result.page_count,
        "char_count": result.char_count,
        "pages_without_text": result.pages_without_text,
        "boilerplate_lines_removed": result.boilerplate_lines_removed,
        "invisible_chars_removed": result.invisible_chars_removed,
    }
    # A new document invalidates everything downstream.
    state.chunking = None
    state.embedding = None
    state.chunks = []
    store.save(state)
    return state.to_json()


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> Response:
    state = _session(request)

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. RAG needs extractable text, "
            "and this pipeline reads it from a PDF text layer.",
        )

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large: {len(payload) / 1_048_576:.1f} MB exceeds "
            f"the {settings.max_upload_mb} MB limit.",
        )

    uploads = Path(settings.data_dir) / "uploads" / state.session_id
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "source.pdf"
    target.write_bytes(payload)

    body = _ingest_path(state, target, file.filename or "document.pdf")
    response = Response(
        content=__import__("json").dumps(body), media_type="application/json"
    )
    _with_session_cookie(response, state.session_id)
    return response


@app.post("/api/use-local")
def use_local(request: Request) -> dict:
    """Load the presenter's bind-mounted document.

    Only available when LOCAL_PDF_PATH resolves, which it does not for attendees
    -- so the UI never shows the button and this route reports 404 if called.
    """
    local = settings.local_pdf
    if local is None:
        raise HTTPException(
            status_code=404,
            detail="No local document is configured. See docker-compose.override.yml.example.",
        )
    return _ingest_path(_session(request), local, local.name)


# -------------------------------------------------------------- step 2/3: chunk


@app.post("/api/chunk")
async def start_chunking(request: Request) -> Response:
    state = _session(request)
    if not state.upload:
        raise HTTPException(status_code=409, detail="Upload a document first.")

    body = await request.json() if await request.body() else {}
    strategy = body.get("strategy", settings.default_strategy)
    size = int(body.get("size", settings.default_chunk_size))
    overlap = int(body.get("overlap", settings.default_chunk_overlap))
    percentile = int(body.get("percentile", settings.semantic_percentile))

    if strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {strategy!r}. Valid: {', '.join(sorted(STRATEGIES))}.",
        )

    job = registry.create()

    async def run() -> None:
        try:
            registry.publish(job, {"type": "stage", "message": f"Splitting with {strategy}..."})
            text = load_pdf(state.pdf_path).text

            # Semantic chunking needs the model; the others do not, so it is only
            # loaded when actually required.
            embeddings = build_embeddings() if strategy == "semantic" else None

            # to_thread keeps the event loop free, so SSE frames keep flushing
            # while a CPU-bound splitter runs.
            result = await asyncio.to_thread(
                chunk,
                text,
                strategy=strategy,
                size=size,
                overlap=overlap,
                embeddings=embeddings,
                percentile=percentile,
            )

            for note in result.notes:
                registry.publish(job, {"type": "note", "message": note})

            # LangChain splitters return everything at once, so this streams the
            # rendering, not the splitting. No artificial delay is added.
            for piece in result.chunks:
                registry.publish(job, {
                    "type": "chunk",
                    "index": piece.index,
                    "text": piece.text,
                    "char_count": len(piece.text),
                    "page": state.page_for_offset(piece.start),
                    "parent_id": piece.parent_id,
                })

            state.chunks = result.chunks
            state.chunking = {
                "strategy": strategy,
                "size": size,
                "overlap": overlap,
                "percentile": percentile,
                "chunk_count": len(result.chunks),
                "sections_detected": result.sections_detected,
                "fell_back": result.fell_back,
                "notes": result.notes,
            }
            state.embedding = None
            store.save(state)

            registry.publish(job, {
                "type": "summary",
                "chunk_count": len(result.chunks),
                "sections_detected": result.sections_detected,
                "fell_back": result.fell_back,
            })
            registry.finish(job, "done")
        except UnknownStrategyError as exc:
            registry.finish(job, "error", str(exc))
        except Exception as exc:  # noqa: BLE001 - reported in the step card
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    asyncio.create_task(run())
    response = Response(
        content=__import__("json").dumps({"job_id": job.job_id}),
        media_type="application/json",
        status_code=202,
    )
    _with_session_cookie(response, state.session_id)
    return response


# ------------------------------------------------------------------ step 4: embed


@app.post("/api/embed")
def start_embedding(request: Request) -> Response:
    state = _session(request)
    if not state.chunks:
        raise HTTPException(status_code=409, detail="Chunk the document first.")

    collection = _collection()
    job = registry.create()

    async def run() -> None:
        try:
            registry.publish(job, {
                "type": "stage",
                "message": f"Loading {settings.embed_model}...",
            })
            embeddings = await asyncio.to_thread(build_embeddings)

            texts = [piece.text for piece in state.chunks]
            loop = asyncio.get_running_loop()

            def on_progress(done: int, total: int) -> None:
                # Called from the worker thread; hop back to the loop thread.
                loop.call_soon_threadsafe(
                    registry.publish,
                    job,
                    {"type": "embedded", "done": done, "total": total},
                )

            vectors = await asyncio.to_thread(
                embed_batched,
                embeddings,
                texts,
                settings.embed_batch_size,
                on_progress,
            )

            registry.publish(job, {"type": "stage", "message": "Writing to ChromaDB..."})
            written = await asyncio.to_thread(
                vector_store.write_chunks,
                collection,
                chunks=state.chunks,
                vectors=vectors,
                doc_id=state.upload["doc_id"],
                source=state.upload["filename"],
                size=state.chunking["size"],
                overlap=state.chunking["overlap"],
                embed_model=settings.embed_model,
                page_for_offset=state.page_for_offset,
            )

            state.embedding = {
                "model": settings.embed_model,
                "dims": settings.embed_dims,
                "vectors_written": written,
            }
            store.save(state)

            registry.publish(job, {"type": "summary", "vectors_written": written})
            registry.finish(job, "done")
        except Exception as exc:  # noqa: BLE001 - reported in the step card
            registry.finish(job, "error", f"{type(exc).__name__}: {exc}")

    asyncio.create_task(run())
    response = Response(
        content=__import__("json").dumps({"job_id": job.job_id}),
        media_type="application/json",
        status_code=202,
    )
    _with_session_cookie(response, state.session_id)
    return response


# ------------------------------------------------------------- progress channels


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    """Stream a job's events as Server-Sent Events."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")

    async def stream():
        # Replay history first, so a client that connects late sees everything.
        for event in list(job.events):
            yield sse_format(event)
        if job.status != "running":
            return
        while True:
            event = await job.queue.get()
            if event is None:      # finish() sentinel
                break
            yield sse_format(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/status/{job_id}")
def job_status(job_id: str) -> dict:
    """Polling fallback for when SSE cannot be established."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "events": job.events,
    }


# ----------------------------------------------------------- step 5: inspect


@app.get("/api/collection")
def collection_records(offset: int = 0, limit: int = 25) -> dict:
    return vector_store.read_records(_collection(), offset=offset, limit=min(limit, 100))


@app.post("/api/reset")
async def reset(request: Request) -> Response:
    state = _session(request)
    body = await request.json() if await request.body() else {}
    if body.get("drop_collection"):
        try:
            vector_store.drop_collection(get_client())
        except Exception:  # noqa: BLE001 - reset should not fail on a dead Chroma
            pass
    fresh = store.reset(state.session_id)
    response = Response(
        content=__import__("json").dumps(fresh.to_json()), media_type="application/json"
    )
    _with_session_cookie(response, fresh.session_id)
    return response
```

- [ ] **Step 5: Run test to verify it passes**

Run: `docker compose run --rm app pytest tests/test_api.py -v`
Expected: PASS, 21 passed

This task depends on `app/templates/index.html` and `app/static/` existing. Create placeholders so the mount and render succeed, and Task 8 fills them in:

```bash
mkdir -p app/templates app/static
printf '<h1>RAG Ingestion</h1>\n{%% for s in strategies %%}<div>{{ s.label }}</div>{%% endfor %%}\n' > app/templates/index.html
touch app/static/.gitkeep
```

- [ ] **Step 6: Commit**

```bash
git add app/jobs.py app/main.py app/templates/index.html app/static/.gitkeep tests/test_api.py
git commit -m "feat: API routes with SSE progress and a polling fallback

One asyncio task and one queue per job -- no broker, per the spec. Jobs keep an
event history alongside the queue, so a dropped SSE connection falls back to
polling /api/status without losing anything, and a late subscriber gets a replay.

CPU-bound work runs through asyncio.to_thread so the event loop keeps flushing
frames, and the embedder's thread hops back via call_soon_threadsafe to publish.
Chunk streaming is honest: LangChain splitters return everything at once, so the
rendering streams while the split does not. No artificial delays."
```

---

## Task 8: The progressive-unlock frontend

**Files:**
- Create (replacing the Task 7 placeholder): `app/templates/index.html`, `app/static/app.css`, `app/static/app.js`

**Interfaces:**
- Consumes: `GET /api/config`, and every route from Task 7. The template receives `strategies`, `settings`, `state`, `has_local_pdf`.
- Produces: no Python interface. Verified by the Task 7 API tests plus the manual checklist in Step 5.

**Design intent:** the page should read as a continuation of the deck, using its tokens — `#0a0f1e` background, cyan `#38e0cf` / amber `#ffb547` / pink `#ff5c8a` / violet `#8b7cff` / green `#5ddb8b`, Bricolage Grotesque + Chivo + JetBrains Mono, and its `card` / `callout` / `pipe`-step patterns. Fonts are referenced by family name only with system fallbacks; **no webfont is fetched**, because the presentation is offline.

- [ ] **Step 1: Write the stylesheet**

Create `app/static/app.css`:

```css
/* Design tokens lifted from rag-workshop.html so the app reads as slide 53
   rather than a different product. Fonts are named with system fallbacks and
   never fetched: the workshop is presented offline. */
:root {
  --bg: #0a0f1e;
  --panel: rgba(255,255,255,.045);
  --panel2: rgba(255,255,255,.075);
  --ink: #e9eef9;
  --muted: #94a3c4;
  --cyan: #38e0cf;
  --amber: #ffb547;
  --pink: #ff5c8a;
  --violet: #8b7cff;
  --green: #5ddb8b;
  --line: rgba(255,255,255,.11);
  --r: 14px;
  --sans: 'Chivo', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --display: 'Bricolage Grotesque', 'Chivo', ui-sans-serif, system-ui, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
  line-height: 1.5;
  padding: clamp(1rem, 3vw, 2.5rem);
  max-width: 1100px;
  margin: 0 auto;
}

h1, h2, h3 { font-family: var(--display); letter-spacing: -.02em; }
h1 { font-size: clamp(1.6rem, 4vw, 2.6rem); font-weight: 800; line-height: 1.05; }
h2 { font-size: clamp(1.05rem, 2vw, 1.5rem); font-weight: 700; }
.hl { color: var(--cyan); font-style: normal; }
.muted { color: var(--muted); }
.mono { font-family: var(--mono); }

header { margin-bottom: 2rem; }
header p { color: var(--muted); max-width: 62ch; margin-top: .6rem; }

.eyebrow {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--cyan);
  display: flex;
  align-items: center;
  gap: .6rem;
  margin-bottom: .5rem;
}
.eyebrow::after { content: ""; height: 1px; flex: 1; background: linear-gradient(90deg, var(--line), transparent); }

/* --- steps ------------------------------------------------------------- */
.step {
  border: 1px solid var(--line);
  border-radius: var(--r);
  background: var(--panel);
  padding: clamp(.9rem, 2vw, 1.5rem);
  margin-bottom: 1.2rem;
  transition: opacity .25s, border-color .25s;
}
.step[data-locked="true"] {
  opacity: .38;
  pointer-events: none;   /* belt and braces; the server is the authority */
}
.step[data-active="true"] { border-color: color-mix(in srgb, var(--cyan) 45%, transparent); }
.step-head { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap; }
.step-num {
  font-family: var(--mono);
  font-size: .72rem;
  color: var(--cyan);
  border: 1px solid color-mix(in srgb, var(--cyan) 40%, transparent);
  border-radius: 999px;
  padding: .1rem .5rem;
}
.step-body { margin-top: 1rem; display: flex; flex-direction: column; gap: .9rem; }

/* --- controls ---------------------------------------------------------- */
button {
  font-family: var(--mono);
  font-size: .78rem;
  background: var(--panel2);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .45rem .85rem;
  cursor: pointer;
  transition: .15s;
}
button:hover:not(:disabled) { border-color: var(--cyan); color: var(--cyan); }
button.pri { background: var(--cyan); color: #04121a; border-color: var(--cyan); font-weight: 700; }
button.pri:hover:not(:disabled) { filter: brightness(1.1); color: #04121a; }
button:disabled { opacity: .45; cursor: not-allowed; }
.ctl { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center; }

input[type=range] { accent-color: var(--cyan); width: 100%; }
label.rng { display: flex; flex-direction: column; gap: .2rem; font-family: var(--mono); font-size: .74rem; color: var(--muted); }
label.rng b { color: var(--cyan); }
label.rng[data-disabled="true"] { opacity: .4; }
.sliders { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

/* --- strategy cards ---------------------------------------------------- */
.strategies { display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); }
.strategy {
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: .35rem;
  padding: .7rem .8rem;
  height: 100%;
}
.strategy[aria-pressed="true"] {
  border-color: var(--cyan);
  background: color-mix(in srgb, var(--cyan) 14%, transparent);
}
.strategy b { font-family: var(--display); font-size: .95rem; color: var(--ink); }
.strategy span { font-size: .72rem; color: var(--muted); line-height: 1.4; font-family: var(--sans); }

/* --- previews ---------------------------------------------------------- */
.preview {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #070b16;
  max-height: 44vh;
  overflow-y: auto;
  padding: .6rem;
  font-family: var(--mono);
  font-size: .72rem;
}
.chunk { border-bottom: 1px solid var(--line); padding: .5rem .2rem; }
.chunk:last-child { border-bottom: 0; }
.chunk-meta { color: var(--cyan); font-size: .66rem; margin-bottom: .25rem; display: flex; gap: .6rem; flex-wrap: wrap; }
.chunk-text { color: #d3ddf2; white-space: pre-wrap; word-break: break-word; }

.record { border-bottom: 1px solid var(--line); padding: .6rem .2rem; }
.record:last-child { border-bottom: 0; }
.record-id { color: var(--amber); font-size: .66rem; }
.record-vec { color: var(--violet); font-size: .66rem; margin-top: .3rem; }
.record details summary { color: var(--muted); cursor: pointer; font-size: .66rem; margin-top: .3rem; }
.record pre { color: var(--muted); font-size: .64rem; white-space: pre-wrap; margin-top: .3rem; }

/* --- stats, notes, errors --------------------------------------------- */
.stats { display: flex; gap: 1.2rem; flex-wrap: wrap; font-family: var(--mono); font-size: .74rem; }
.stats div { display: flex; flex-direction: column; }
.stats b { color: var(--cyan); font-size: 1.05rem; }
.stats span { color: var(--muted); font-size: .66rem; }

.callout { border: 1px solid color-mix(in srgb, var(--c, var(--cyan)) 32%, transparent);
  background: color-mix(in srgb, var(--c, var(--cyan)) 7%, transparent);
  border-radius: 10px; padding: .55rem .8rem; font-size: .78rem; }
.callout.warn { --c: var(--amber); }
.callout.err  { --c: var(--pink); }
.callout.ok   { --c: var(--green); }
.callout[hidden] { display: none; }

.bar { height: 8px; background: var(--panel2); border-radius: 4px; overflow: hidden; }
.bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--violet), var(--cyan)); transition: width .2s ease; }

.drop {
  border: 1px dashed color-mix(in srgb, var(--amber) 55%, transparent);
  background: color-mix(in srgb, var(--amber) 7%, transparent);
  border-radius: var(--r); padding: 1.4rem; text-align: center; font-size: .82rem;
}
.drop.over { border-color: var(--cyan); background: color-mix(in srgb, var(--cyan) 10%, transparent); }

@media (max-width: 620px) { .sliders { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { transition-duration: .01ms !important; } }
```

- [ ] **Step 2: Write the template**

Create `app/templates/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Ingestion Pipeline</title>
<link rel="stylesheet" href="/static/app.css">
</head>
<body>

<header>
  <div class="eyebrow">Levels 2 &amp; 3 &middot; Live</div>
  <h1>Documents in, <em class="hl">vectors out.</em></h1>
  <p>
    The indexing half of the pipeline, one step at a time. Each step unlocks when
    the one above it finishes. Defaults are the deck's numbers:
    {{ settings.default_chunk_size }}-character chunks,
    {{ settings.default_chunk_overlap }} overlap, recursive splitting.
  </p>
</header>

<div class="callout err" id="banner" hidden></div>

<!-- ---------------------------------------------------------- step 1 -->
<section class="step" id="step-1" data-locked="false" data-active="true">
  <div class="step-head">
    <span class="step-num">STEP 1</span>
    <h2>Load a document</h2>
  </div>
  <div class="step-body">
    <div class="drop" id="drop">
      <p>Drag a PDF here, or <button type="button" id="pick">choose a file</button></p>
      <p class="muted mono" style="font-size:.7rem;margin-top:.4rem">
        Needs a text layer &mdash; scanned PDFs have none. Max {{ settings.max_upload_mb }} MB.
      </p>
      <input type="file" id="file" accept="application/pdf" hidden>
    </div>

    {% if has_local_pdf %}
    <div class="ctl">
      <button type="button" id="use-local">Use local document</button>
      <span class="muted mono" style="font-size:.7rem">Presenter shortcut &mdash; skips the file picker.</span>
    </div>
    {% endif %}

    <div class="stats" id="upload-stats" hidden></div>
    <div class="callout warn" id="clean-note" hidden></div>
  </div>
</section>

<!-- ---------------------------------------------------------- step 2 -->
<section class="step" id="step-2" data-locked="true">
  <div class="step-head">
    <span class="step-num">STEP 2</span>
    <h2>Choose how to cut it</h2>
  </div>
  <div class="step-body">
    <div class="strategies" id="strategies">
      {% for s in strategies %}
      <button type="button" class="strategy" data-key="{{ s.key }}"
              data-uses-size="{{ 'true' if s.uses_size else 'false' }}"
              data-uses-overlap="{{ 'true' if s.uses_overlap else 'false' }}"
              data-extra="{{ s.extra_control }}"
              aria-pressed="{{ 'true' if s.key == settings.default_strategy else 'false' }}">
        <b>{{ s.label }}</b>
        <span>{{ s.verdict }}</span>
      </button>
      {% endfor %}
    </div>

    <div class="sliders">
      <label class="rng" id="size-label">
        Chunk size <b id="size-out">{{ settings.default_chunk_size }}</b> chars
        <input type="range" id="size" min="100" max="3000" step="50"
               value="{{ settings.default_chunk_size }}">
      </label>
      <label class="rng" id="overlap-label">
        Overlap <b id="overlap-out">{{ settings.default_chunk_overlap }}</b> chars
        <input type="range" id="overlap" min="0" max="500" step="10"
               value="{{ settings.default_chunk_overlap }}">
      </label>
      <label class="rng" id="percentile-label" hidden>
        Breakpoint percentile <b id="percentile-out">{{ settings.semantic_percentile }}</b>
        <input type="range" id="percentile" min="50" max="99" step="1"
               value="{{ settings.semantic_percentile }}">
      </label>
    </div>

    <div class="callout" id="strategy-note"></div>
    <div class="ctl"><button type="button" class="pri" id="run-chunk">Start chunking</button></div>
  </div>
</section>

<!-- ---------------------------------------------------------- step 3 -->
<section class="step" id="step-3" data-locked="true">
  <div class="step-head">
    <span class="step-num">STEP 3</span>
    <h2>The chunks</h2>
    <span class="muted mono" style="font-size:.72rem" id="chunk-count"></span>
  </div>
  <div class="step-body">
    <div class="callout warn" id="chunk-notes" hidden></div>
    <div class="preview" id="chunk-preview"></div>
  </div>
</section>

<!-- ---------------------------------------------------------- step 4 -->
<section class="step" id="step-4" data-locked="true">
  <div class="step-head">
    <span class="step-num">STEP 4</span>
    <h2>Embed and store</h2>
  </div>
  <div class="step-body">
    <p class="muted" style="font-size:.82rem">
      <span class="mono">{{ settings.embed_model }}</span> &middot;
      {{ settings.embed_dims }} dimensions &middot; normalised, so cosine is a dot product.
      Runs on CPU, in this container, with no network.
    </p>
    <div class="ctl"><button type="button" class="pri" id="run-embed">Start embedding</button></div>
    <div class="bar"><i id="embed-bar"></i></div>
    <div class="muted mono" style="font-size:.72rem" id="embed-status"></div>
  </div>
</section>

<!-- ---------------------------------------------------------- step 5 -->
<section class="step" id="step-5" data-locked="true">
  <div class="step-head">
    <span class="step-num">STEP 5</span>
    <h2>What ChromaDB is holding</h2>
    <span class="muted mono" style="font-size:.72rem" id="record-count"></span>
  </div>
  <div class="step-body">
    <div class="preview" id="records"></div>
    <div class="ctl">
      <button type="button" id="prev-page">&larr; Previous</button>
      <button type="button" id="next-page">Next &rarr;</button>
      <button type="button" id="reset">Reset everything</button>
    </div>
  </div>
</section>

<script>
  // Server-rendered state, so a refresh resumes where the presenter left off.
  window.__STATE__ = {{ state | tojson }};
</script>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write the client script**

Create `app/static/app.js`:

```javascript
/* Progressive-unlock client.
 *
 * No framework and no build step: the interesting code in this repository is the
 * Python pipeline, and a bundler between a reader and that code would be a tax.
 *
 * Two things worth knowing:
 *  - The server decides which step is unlocked (state.unlocked_step). This file
 *    only reflects that decision, so the DOM is never the authority.
 *  - Progress arrives over SSE, with a polling fallback if EventSource fails.
 *    A flaky projector setup should not kill the preview.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { unlocked: 1, strategy: null, chunkCount: 0, offset: 0, limit: 25 };

/* ------------------------------------------------------------------ helpers */

function banner(message) {
  const el = $('banner');
  el.textContent = message;
  el.hidden = !message;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch { /* non-JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function applyUnlock(step) {
  state.unlocked = step;
  for (let n = 1; n <= 5; n++) {
    const section = $(`step-${n}`);
    // Steps 2 and 3 unlock together: picking a strategy and running it is one
    // interaction, so step 3 is reachable whenever step 2 is.
    const reachable = n <= step || (n === 3 && step >= 2);
    section.dataset.locked = reachable ? 'false' : 'true';
    section.dataset.active = n === step ? 'true' : 'false';
  }
}

/* Subscribe to a job, preferring SSE and degrading to polling. */
function follow(jobId, onEvent, onDone) {
  let settled = false;
  const finish = (error) => {
    if (settled) return;
    settled = true;
    onDone(error);
  };

  let source;
  try {
    source = new EventSource(`/api/events/${jobId}`);
  } catch {
    return poll(jobId, onEvent, finish);
  }

  source.onmessage = (message) => {
    const event = JSON.parse(message.data);
    onEvent(event);
    if (event.type === 'done' || event.type === 'error') {
      source.close();
      finish(event.type === 'error' ? event.error : null);
    }
  };

  source.onerror = () => {
    // Could be a genuine failure or a closed stream after completion. Polling
    // resolves which, and replays anything missed.
    source.close();
    if (!settled) poll(jobId, onEvent, finish);
  };
}

/* Polling fallback. Job event history is replayed, so nothing is lost. */
function poll(jobId, onEvent, finish) {
  let seen = 0;
  const tick = async () => {
    let status;
    try {
      status = await api(`/api/status/${jobId}`);
    } catch (error) {
      return finish(error.message);
    }
    status.events.slice(seen).forEach(onEvent);
    seen = status.events.length;
    if (status.status === 'running') return setTimeout(tick, 400);
    finish(status.status === 'error' ? status.error : null);
  };
  tick();
}

/* ------------------------------------------------------------- step 1: load */

function renderUpload(upload) {
  $('upload-stats').hidden = false;
  $('upload-stats').innerHTML = [
    ['Pages', upload.page_count],
    ['Characters', upload.char_count.toLocaleString()],
    ['Blank pages', upload.pages_without_text],
  ].map(([label, value]) => `<div><b>${value}</b><span>${label}</span></div>`).join('');

  // The deck's "embedding raw junk" gotcha, as a number the room can see.
  const removed = upload.boilerplate_lines_removed;
  const invisible = upload.invisible_chars_removed;
  if (removed || invisible) {
    $('clean-note').hidden = false;
    $('clean-note').innerHTML =
      `Cleaned before embedding: removed <b>${removed}</b> boilerplate lines ` +
      `(running headers, footers, table-of-contents entries) and stripped ` +
      `<b>${invisible}</b> invisible characters. All of it would have embedded ` +
      `perfectly well and polluted every result.`;
  }
}

async function loadDocument(request) {
  banner('');
  try {
    const body = await request();
    renderUpload(body.upload);
    applyUnlock(body.unlocked_step);
  } catch (error) {
    banner(error.message);
  }
}

function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  return loadDocument(() => api('/api/upload', { method: 'POST', body: form }));
}

$('pick').onclick = () => $('file').click();
$('file').onchange = (event) => {
  if (event.target.files[0]) uploadFile(event.target.files[0]);
};

const drop = $('drop');
drop.addEventListener('dragover', (event) => {
  event.preventDefault();
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('over');
  if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]);
});

if ($('use-local')) {
  $('use-local').onclick = () =>
    loadDocument(() => api('/api/use-local', { method: 'POST' }));
}

/* -------------------------------------------------- step 2: pick a strategy */

function selectStrategy(button) {
  document.querySelectorAll('.strategy').forEach((el) =>
    el.setAttribute('aria-pressed', String(el === button)));
  state.strategy = button.dataset.key;

  // Relabel the controls, because the sliders do not mean the same thing for
  // every strategy -- and disabled controls must look disabled.
  const usesSize = button.dataset.usesSize === 'true';
  const usesOverlap = button.dataset.usesOverlap === 'true';
  $('size').disabled = !usesSize;
  $('overlap').disabled = !usesOverlap;
  $('size-label').dataset.disabled = String(!usesSize);
  $('overlap-label').dataset.disabled = String(!usesOverlap);
  $('percentile-label').hidden = button.dataset.extra !== 'percentile';

  const notes = {
    fixed: 'Watch the boundaries: this cuts on character count alone and will slice words in half.',
    recursive: 'Tries paragraph, then line, then sentence, then word. The deck\'s recommended default.',
    structure: 'Splits on the document\'s own headings. If it finds none, it says so and falls back to recursive.',
    semantic: 'Size and overlap do not apply: cut points come from embedding distance between neighbouring sentences. Slower, because every sentence gets embedded first.',
    parent: 'Small children get embedded, larger parents are kept alongside. The retrieval payoff needs the query build; here you see the structure.',
  };
  $('strategy-note').textContent = notes[state.strategy] || '';
}

document.querySelectorAll('.strategy').forEach((button) => {
  button.onclick = () => selectStrategy(button);
});

[['size', 'size-out'], ['overlap', 'overlap-out'], ['percentile', 'percentile-out']]
  .forEach(([input, output]) => {
    $(input).oninput = () => { $(output).textContent = $(input).value; };
  });

/* ------------------------------------------------- step 3: run the chunking */

$('run-chunk').onclick = async () => {
  banner('');
  $('run-chunk').disabled = true;
  $('chunk-preview').innerHTML = '';
  $('chunk-notes').hidden = true;
  $('chunk-notes').innerHTML = '';
  state.chunkCount = 0;

  const notes = [];
  try {
    const { job_id } = await api('/api/chunk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        strategy: state.strategy,
        size: Number($('size').value),
        overlap: Number($('overlap').value),
        percentile: Number($('percentile').value),
      }),
    });

    follow(job_id, (event) => {
      if (event.type === 'chunk') {
        state.chunkCount += 1;
        const chunk = document.createElement('div');
        chunk.className = 'chunk';
        const parent = event.parent_id ? ` · parent ${event.parent_id}` : '';
        chunk.innerHTML =
          `<div class="chunk-meta"><span>#${event.index}</span>` +
          `<span>${event.char_count} chars</span><span>page ${event.page}</span>` +
          `<span>${parent}</span></div>` +
          `<div class="chunk-text"></div>`;
        // textContent, not innerHTML: chunk text is document content and must
        // never be interpreted as markup.
        chunk.querySelector('.chunk-text').textContent = event.text;
        $('chunk-preview').appendChild(chunk);
        $('chunk-count').textContent = `${state.chunkCount} chunks`;
      } else if (event.type === 'note') {
        notes.push(event.message);
      } else if (event.type === 'stage') {
        $('chunk-count').textContent = event.message;
      }
    }, (error) => {
      $('run-chunk').disabled = false;
      if (error) return banner(error);
      if (notes.length) {
        $('chunk-notes').hidden = false;
        $('chunk-notes').innerHTML = notes.map((n) => `<div>${n}</div>`).join('');
      }
      $('chunk-count').textContent = `${state.chunkCount} chunks`;
      applyUnlock(4);
    });
  } catch (error) {
    $('run-chunk').disabled = false;
    banner(error.message);
  }
};

/* ------------------------------------------------------- step 4: embeddings */

$('run-embed').onclick = async () => {
  banner('');
  $('run-embed').disabled = true;
  const started = Date.now();

  try {
    const { job_id } = await api('/api/embed', { method: 'POST' });

    follow(job_id, (event) => {
      if (event.type === 'embedded') {
        const pct = Math.round((event.done / event.total) * 100);
        $('embed-bar').style.width = `${pct}%`;
        const seconds = ((Date.now() - started) / 1000).toFixed(1);
        $('embed-status').textContent =
          `${event.done} / ${event.total} chunks embedded · ${seconds}s`;
      } else if (event.type === 'stage') {
        $('embed-status').textContent = event.message;
      } else if (event.type === 'summary') {
        $('embed-status').textContent =
          `${event.vectors_written} vectors written to ChromaDB`;
      }
    }, async (error) => {
      $('run-embed').disabled = false;
      if (error) return banner(error);
      $('embed-bar').style.width = '100%';
      applyUnlock(5);
      state.offset = 0;
      await loadRecords();
    });
  } catch (error) {
    $('run-embed').disabled = false;
    banner(error.message);
  }
};

/* -------------------------------------------------- step 5: browse the store */

async function loadRecords() {
  try {
    const page = await api(`/api/collection?offset=${state.offset}&limit=${state.limit}`);
    $('record-count').textContent =
      `${page.total} records · showing ${page.offset + 1}-${page.offset + page.records.length}`;

    $('records').innerHTML = '';
    page.records.forEach((record) => {
      const el = document.createElement('div');
      el.className = 'record';
      el.innerHTML =
        `<div class="record-id"></div><div class="chunk-text"></div>` +
        `<div class="record-vec"></div>` +
        `<details><summary>metadata</summary><pre></pre></details>`;
      el.querySelector('.record-id').textContent = record.id;
      el.querySelector('.chunk-text').textContent = record.text;
      // The norm is displayed so the room can confirm normalisation happened
      // rather than taking the flag on trust.
      el.querySelector('.record-vec').textContent =
        `[${record.vector_preview.join(', ')}, ...] ` +
        `${record.dims} dims · norm ${record.vector_norm}`;
      el.querySelector('pre').textContent = JSON.stringify(record.metadata, null, 2);
      $('records').appendChild(el);
    });

    $('prev-page').disabled = page.offset === 0;
    $('next-page').disabled = page.offset + page.records.length >= page.total;
  } catch (error) {
    banner(error.message);
  }
}

$('prev-page').onclick = () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadRecords();
};
$('next-page').onclick = () => {
  state.offset += state.limit;
  loadRecords();
};

$('reset').onclick = async () => {
  if (!confirm('Clear this session and drop the ChromaDB collection?')) return;
  try {
    const body = await api('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drop_collection: true }),
    });
    $('chunk-preview').innerHTML = '';
    $('records').innerHTML = '';
    $('upload-stats').hidden = true;
    $('clean-note').hidden = true;
    $('embed-bar').style.width = '0';
    $('embed-status').textContent = '';
    $('chunk-count').textContent = '';
    $('record-count').textContent = '';
    applyUnlock(body.unlocked_step);
  } catch (error) {
    banner(error.message);
  }
};

/* ----------------------------------------------------------------- start-up */

(function init() {
  const served = window.__STATE__ || {};
  selectStrategy(document.querySelector('.strategy[aria-pressed="true"]'));
  if (served.upload) renderUpload(served.upload);
  applyUnlock(served.unlocked_step || 1);
  if ((served.unlocked_step || 1) >= 5) loadRecords();
})();
```

- [ ] **Step 4: Confirm the API tests still pass against the real template**

Run: `docker compose run --rm app pytest tests/test_api.py -v`
Expected: PASS, 21 passed — `test_the_page_lists_all_five_strategies` now checks the real template rather than the placeholder.

- [ ] **Step 5: Manual verification checklist**

```bash
docker compose up -d --build
```

Open `http://localhost:8080` and confirm each item:

- [ ] Steps 2–5 are visibly dimmed and unclickable on load
- [ ] Dragging a PDF onto step 1 shows page count, characters, and the cleaning counts
- [ ] Step 2 unlocks; **Recursive** is preselected
- [ ] Selecting **Semantic** disables and dims the size and overlap sliders, and reveals the percentile slider
- [ ] *Start chunking* fills the preview; each chunk shows index, char count, and page
- [ ] Selecting **Fixed size** at ~120 chars visibly slices words in half
- [ ] Selecting **Structure aware** on a heading-free PDF shows the fallback note
- [ ] Step 4's progress bar advances in stages, and elapsed seconds increase
- [ ] Step 5 lists records with a vector preview and `norm 1.0`
- [ ] Pagination buttons disable correctly at both ends
- [ ] Reloading the browser mid-demo keeps the completed steps unlocked
- [ ] *Reset everything* returns the page to step 1 only

- [ ] **Step 6: Commit**

```bash
git add app/templates/index.html app/static/app.css app/static/app.js
git commit -m "feat: progressive-unlock frontend using the deck's design tokens

No framework and no build step, so nothing sits between a reader and the Python
pipeline. Fonts are named with system fallbacks and never fetched, because the
workshop is presented offline.

Unlock state comes from the server, so the DOM is never the authority. Strategy
cards relabel the sliders per strategy -- semantic disables size and overlap
rather than showing controls that silently do nothing. Chunk and record text is
set via textContent, never innerHTML, since document content must not be
interpreted as markup."
```

---

## Task 9: CLAUDE.md and README.md

**Files:**
- Create: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: the finished repository.
- Produces: no code interface.

- [ ] **Step 1: Write CLAUDE.md**

Create `CLAUDE.md`:

```markdown
# CLAUDE.md

Context for working on this repository.

## What this is

A teaching app built for a two-hour workshop, *RAG, Embeddings & Vector
Databases*. The slide deck is `rag-workshop.html` (52 slides, 6 levels), and this
app is the live counterpart to its indexing half: a PDF goes in, vectors land in
ChromaDB, and the audience watches each stage happen.

It is optimised for **being read aloud and cloned by attendees**, not for
production throughput. When a change would make the code faster but harder to
follow, prefer legibility and say why in a comment.

## Scope

Implemented: load → clean → chunk → embed → store → inspect (deck Levels 2–3).

Not implemented, and deliberately so: retrieval, hybrid search, reranking, prompt
assembly, generation, chat, evaluation, auth, multi-user sessions, access-control
filtering. These are deck Levels 5–6 and a separate build.

## Architecture

    Browser --HTTP + SSE--> app (FastAPI :8080) --HTTP--> chromadb (:8000)
                              |                            |
                        /data/sessions/*.json        chroma-data volume
                        /data/uploads/*.pdf
                        /opt/hf  (MiniLM weights, baked in)

| Path | Responsibility |
|---|---|
| `app/config.py` | Every tunable, each default annotated with its deck slide |
| `app/pipeline/loader.py` | PDF → cleaned text, cleaning counts, page-offset map |
| `app/pipeline/chunkers.py` | The five strategies. **Read this first** |
| `app/pipeline/embedder.py` | MiniLM wrapper, batched with progress |
| `app/pipeline/store.py` | Chroma writes and record reads |
| `app/session.py` | Session state + JSON mirror; owns the unlock rule |
| `app/jobs.py` | Job registry and SSE queues |
| `app/main.py` | Routes |

## Decisions that look like bugs but are not

**The deck's code slides use LlamaIndex; this app uses LangChain.** Intentional.
LangChain was a hard requirement; the deck was left unmodified. The presenter
bridges the gap verbally. Do not "fix" either side.

**Writes go through the raw `chromadb` client, not `langchain-chroma`.**
`langchain-chroma` computes embeddings internally, which would make both
per-batch progress reporting and the vector preview impossible. LangChain still
supplies the loader, all five splitters, and the embedding wrapper.

**`OLLAMA_BASE_URL` is read by nothing.** It is a documented seam for the future
query build, where `deepseek-r1:1.5b` is the intended generation model. Nothing
in ingestion needs an LLM. It is the only dead config entry, and it is on purpose.

**Semantic chunking uses an embeddings model, not an LLM.** `SemanticChunker`
embeds sentences and cuts where cosine distance between neighbours exceeds a
percentile. No text is generated. Expect to explain this more than once.

**Chunk streaming is partly cosmetic, and honestly so.** LangChain's splitters
return every chunk at once, so step 3 streams the *rendering*, not the splitting.
Embedding progress is genuinely per batch. **Never add artificial delays to make
progress look smoother** — a teaching tool that fakes its own telemetry is worse
than one with a jumpy progress bar.

**`parent_id` is `""` rather than `None`.** Chroma metadata rejects null values.

**Delete-before-write is load-bearing.** Content-hash ids make a same-parameters
re-run an overwrite, but shrinking the chunk count would orphan the earlier run's
tail. Records are deleted per `(doc_id, strategy)` before writing. Two
*different* strategies are meant to coexist for comparison.

## The private document

The presenter demos against an internal PDF that must not be published.

- `.gitignore` excludes `*.pdf` **absolutely, with no allow-list**. Do not add
  one — tests generate PDFs at runtime instead.
- The document is **never `COPY`'d into the image**; that would leak it on any
  image push. It is bind-mounted read-only via `docker-compose.override.yml`
  (gitignored). See `docker-compose.override.yml.example`.
- `.dockerignore` also excludes `*.pdf`, so no document reaches the build context.
- `README.md` must never reference it.

If you are asked to add a sample document, generate a synthetic one — do not
reach for the presenter's file.

## Where the defaults come from

| Setting | Value | Slide |
|---|---|---|
| Chunk size | 700 | Level 6, "Sensible defaults for version one" |
| Overlap | 100 | Same |
| Strategy | recursive | Level 3, "the right default" |
| Embedding model | all-MiniLM-L6-v2, 384d | Level 2 model table, self-host row |
| Metric | cosine, normalised | Level 6 defaults |

Changing one of these means changing the slide too, or the demo starts
contradicting the teaching.

## Commands

    docker compose up -d --build          # start; open localhost:8080
    docker compose run --rm app pytest -v # full suite
    docker compose run --rm app pytest -m "not slow"   # skip the real-model test
    docker compose down                   # stop, keep vectors
    docker compose down -v                # stop, delete vectors

The model is baked into the image. To verify that still holds after touching the
Dockerfile:

    docker compose run --rm --network none app python -c \
      "from sentence_transformers import SentenceTransformer as S; \
       print(S('sentence-transformers/all-MiniLM-L6-v2').get_sentence_embedding_dimension())"

Expected: `384`. If this fails, the offline presentation requirement is broken.

## Conventions

- **High comment density in `app/pipeline/`.** Attendees read those four files
  first. Every strategy docstring quotes the deck's verdict for that strategy.
- Normal comment density elsewhere.
- Tests document behaviour rather than chase coverage. Several deliberately
  assert that a strategy is *bad* in the way the deck says it is.
- No PDF, ever, in a commit.

## Design documents

- Spec: `docs/superpowers/specs/2026-07-27-rag-ingestion-pipeline-design.md`
- Plan: `docs/superpowers/plans/2026-07-27-rag-ingestion-pipeline.md`
```

- [ ] **Step 2: Write README.md**

Create `README.md`:

```markdown
# RAG Ingestion Pipeline

A step-by-step, watchable version of the indexing half of a RAG system. Load a
PDF, choose how to cut it, watch the chunks appear, embed them locally, and
inspect exactly what landed in the vector database.

Built as the live demo for a two-hour workshop on RAG, embeddings and vector
databases. The slides are in `rag-workshop.html` — open it in a browser.

## Quick start

You need Docker and about 2GB of disk.

    git clone <this-repo>
    cd class-rag
    docker compose up -d --build

The first build takes a few minutes: it downloads the embedding model and bakes
it into the image, so nothing needs the network afterwards.

Then open <http://localhost:8080> and drag in a PDF.

## Bring your own PDF

No document ships with this repository. Any PDF with a **text layer** works —
anything you can select text in.

A good demo document has:

- real headings (so structure-aware chunking has something to find)
- a table of contents (so you can watch it get stripped before embedding)
- more than a few pages (so chunk counts are interesting)

**Scanned PDFs will not work.** Their pages are images, so there is no text to
extract; the app tells you this rather than silently producing an empty
collection. Run OCR over such a document first.

## The five steps

1. **Load** — extract text, then clean it. Running headers, footers, and
   table-of-contents lines are removed and counted, because all of them embed
   perfectly well and then pollute every search result.
2. **Choose a strategy** — fixed size, recursive, structure-aware, semantic, or
   parent-document. Each card carries the verdict from the workshop's Level 3
   table. Defaults are 700-character chunks with 100 overlap.
3. **Chunk** — chunks stream into a scrollable preview with index, character
   count, and source page. Try fixed size at ~120 characters to see words get
   sliced in half; that failure is the point.
4. **Embed** — `all-MiniLM-L6-v2` runs on CPU inside the container. 384
   dimensions, normalised, no API key, no network.
5. **Inspect** — browse the ChromaDB records: id, text, full metadata, the first
   eight vector dimensions, and the vector norm.

## Which file matches which part of the workshop

| Workshop level | Code |
|---|---|
| Level 2 — Embeddings | `app/pipeline/embedder.py` |
| Level 3 — Chunking | `app/pipeline/chunkers.py` |
| Level 4 — Vector DBs | `app/pipeline/store.py` |
| Cleaning gotcha (Level 2) | `app/pipeline/loader.py` |

`app/pipeline/chunkers.py` is the one to read first. Each strategy is a single
function whose docstring quotes what the slides say about it.

## Tests

    docker compose run --rm app pytest -v

Test PDFs are generated at runtime, so the suite is self-contained. One test
loads the real embedding model and is marked `slow`:

    docker compose run --rm app pytest -m "not slow"

## Configuration

Copy `.env.example` to `.env` to change anything. Every default is a number from
the workshop, and each is commented with the slide it came from.

## Stopping

    docker compose down       # keeps your vectors
    docker compose down -v    # deletes them

## Not included

Retrieval, reranking, and generation — the querying half. This app deliberately
stops where the vectors land. The workshop covers the rest in Levels 5 and 6.

## Licence

MIT.
```

- [ ] **Step 3: Verify docs match reality**

```bash
# Every command in the READMEs should exist and every path should resolve.
grep -oE 'app/[a-z_/]+\.(py|css|js|html)' README.md CLAUDE.md | cut -d: -f2 | sort -u | \
  while read -r path; do [ -e "$path" ] || echo "MISSING: $path"; done
```

Expected: no output.

```bash
docker compose down -v && docker compose up -d --build && sleep 20
curl -s localhost:8080/api/health | grep -q '"status":"ok"' && echo "quickstart works"
```

Expected: `quickstart works`

- [ ] **Step 4: Confirm no PDF is tracked**

```bash
git ls-files | grep -i '\.pdf$' && echo "FAIL: a PDF is tracked" || echo "OK: no PDF tracked"
```

Expected: `OK: no PDF tracked`

- [ ] **Step 5: Run the full suite one last time**

```bash
docker compose run --rm app pytest -v
```

Expected: 96 passed (7 config + 14 loader + 24 chunkers + 6 embedder + 14 store + 10 session + 21 api)

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: CLAUDE.md and attendee-facing README

CLAUDE.md records the decisions that look like bugs but are not -- the
LangChain/LlamaIndex split with the deck, the raw chromadb client, the
deliberately unused Ollama seam, and why chunk streaming is partly cosmetic --
so none of them get 'fixed' later.

README never references the internal document: attendees bring their own PDF,
and the constraints on it (needs a text layer) are stated with the reason."
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: decisions and config → Task 1; private-document handling → Tasks 1 (`.dockerignore`, override example) and 9 (CLAUDE.md); loader with cleaning and page attribution → Task 2; the five strategies, slider semantics, heading detection, parent caveat → Task 3; embedder and normalisation → Task 4; metadata schema and delete-before-write → Task 5; session persistence and unlock rule → Task 6; data flow, endpoints, SSE and its fallback → Task 7; frontend tokens and step layout → Task 8; documentation → Task 9. The spec's error-handling table is covered across Tasks 2 (scanned PDF), 5 (idempotency), 7 (size/type limits, Chroma 503, unknown strategy), and 8 (banner, polling fallback).

**Interface consistency.** `page_for_offset` has the same name and signature in `LoadResult` and `SessionState`. `Chunk` field names (`index`, `text`, `start`, `strategy`, `parent_id`, `parent_text`) are identical across Tasks 3, 5, 6 and 7. `write_chunks` is called in Task 7 with exactly the keyword arguments Task 5 defines. `STRATEGIES` keys (`fixed`, `recursive`, `structure`, `semantic`, `parent`) match the template's `data-key` values and the note lookup in `app.js`. `registry.publish`/`finish` signatures match their use in `main.py`.

**Two fixes applied during review:**

1. Task 7's tests originally imported `store` from `app.session` while monkeypatching `main_module.store`; the fixture now patches the attribute on `main_module`, which is what the routes actually read.
2. `_locate` initially searched from `cursor` alone, which fails when overlap places a chunk's start behind the cursor. It now rewinds by `len(needle)` before searching, with a whole-document fallback.

**Known deferrals, stated rather than hidden:**

- Session state is per-process. Two Uvicorn workers would not share it. Fine for a single-presenter demo; `--workers 1` is implied by the Compose `CMD`.
- `registry` never evicts finished jobs. Bounded by one demo's lifetime; a long-running deployment would need a reaper.
- `test_status_reports_a_finished_chunk_job` relies on `TestClient` draining background tasks before returning. If a future FastAPI release changes that, the assertion should switch to polling rather than being loosened.

