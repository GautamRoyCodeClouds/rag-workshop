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

    # --- Retrieval (deck Levels 5-6: the query-time half of RAG) -----------
    # top_k / min_score are the two knobs the chat panel exposes live; 5 and
    # 0.25 are the deck's own worked example.
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.25
    # Restated from retrieve()'s own default in retriever.py, not just left
    # implicit, so an env override changes both call sites consistently
    # rather than only the one that happens to pass it explicitly.
    retrieval_mmr_lambda: float = 0.5
    retrieval_pool_multiplier: int = 4

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

        def decimal(key: str, default: float) -> float:
            # Same contract as number(): empty/unset falls back to the
            # default, anything else is parsed strictly -- a malformed value
            # (e.g. "abc") raises via float() rather than silently becoming
            # 0.0, which would make min_score's threshold pass everything.
            value = e.get(key)
            return default if value is None or value == "" else float(value)

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
            retrieval_top_k=number("RETRIEVAL_TOP_K", cls.retrieval_top_k),
            retrieval_min_score=decimal("RETRIEVAL_MIN_SCORE", cls.retrieval_min_score),
            retrieval_mmr_lambda=decimal("RETRIEVAL_MMR_LAMBDA", cls.retrieval_mmr_lambda),
            retrieval_pool_multiplier=number(
                "RETRIEVAL_POOL_MULTIPLIER", cls.retrieval_pool_multiplier
            ),
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
