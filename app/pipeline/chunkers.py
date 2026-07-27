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
    attribute chunks to pages. Repetitive documents (PROSE in the tests, but
    also plenty of real handbooks) make this genuinely ambiguous: the same
    sentence can occur a dozen times, at offsets that are indistinguishable
    from the substring alone. The splitter still visits the document in order
    though, so searching forward from where the previous chunk was actually
    found -- never rewinding past it -- always lands on the correct, next
    occurrence. Rewinding by len(needle) was tried and rejected: with cursor
    tracked loosely it collapses to a search from character 0 and silently
    picks the *first* occurrence instead of the right one, corrupting every
    offset after the first repeat. Cursor must be the previous chunk's real
    start, not merely "somewhere past it", for this to hold.
    """
    if not needle:
        return cursor
    found = text.find(needle, cursor)
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
        cursor = start
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
            cursor = start
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
