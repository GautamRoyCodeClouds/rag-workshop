"""Chunker tests.

Several of these assert that a strategy is *bad* in the specific way the deck
says it is. That is deliberate: the demo's teaching value depends on fixed-size
really shredding words, so it is worth a test.
"""

import re

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

# Deliberately degenerate: short period, heavy self-similarity. A splitter
# whose offset tracking merely rewinds-and-searches (rather than genuinely
# advancing past what it already found) collapses this into a handful of
# distinct starts instead of one per chunk -- see the CRITICAL 1 finding.
REPEAT = "ab. " * 80


def _assert_literal_offsets(source: str, chunks) -> None:
    """Every chunk's start must point at its own literal text in the source."""
    for c in chunks:
        assert source[c.start : c.start + len(c.text)] == c.text, (
            f"chunk {c.index} claims start={c.start} but that slice does not "
            f"match its text {c.text[:40]!r}"
        )


def _assert_strictly_increasing(chunks) -> None:
    starts = [c.start for c in chunks]
    assert starts == sorted(starts), f"starts went backwards: {starts}"
    assert len(set(starts)) == len(starts), (
        f"starts are not strictly increasing -- a repeated offset means the "
        f"search froze on one occurrence instead of advancing: {starts}"
    )


def _assert_offsets_correct_allowing_whitespace(source: str, chunks) -> None:
    """Confirm each chunk's start is genuinely correct, not merely plausible.

    SemanticChunker can legitimately rejoin sentences with a single space
    where the source had a run of several whitespace characters, so a plain
    `source[start:start+len(text)] == text` slice is the wrong check here: it
    would fail even at the *correct* start, because the source's true span is
    longer than len(text) whenever a run got collapsed. Instead, build a
    regex from the chunk text that pins down every non-whitespace character
    exactly and lets whitespace runs match loosely, then require it to match
    anchored at exactly c.start. A chunk whose offset is wrong -- guessed,
    frozen, or pointing at some other occurrence entirely -- will not satisfy
    this either, so it is exactly as strict as the exact-match check for
    every strategy that doesn't touch whitespace, while still accepting the
    one legitimate difference semantic chunking introduces.
    """
    ws = re.compile(r"\s+")
    for c in chunks:
        tokens = [re.escape(part) for part in ws.split(c.text)]
        pattern = re.compile(r"\s+".join(tokens))
        match = pattern.match(source, c.start)
        assert match is not None, (
            f"chunk {c.index} claims start={c.start} but its text (whitespace "
            f"aside) does not appear there: {c.text[:40]!r}"
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

    def test_start_offsets_point_at_the_real_text(self):
        result = chunk(PROSE, strategy="fixed", size=60, overlap=40)
        _assert_literal_offsets(PROSE, result.chunks)

    def test_start_offsets_are_strictly_increasing(self):
        result = chunk(PROSE, strategy="fixed", size=60, overlap=40)
        _assert_strictly_increasing(result.chunks)

    def test_repetitive_text_gets_one_distinct_start_per_chunk(self):
        result = chunk(REPEAT, strategy="fixed", size=8, overlap=6)
        starts = [c.start for c in result.chunks]
        assert len(set(starts)) == len(result.chunks)


class TestRecursive:
    def test_does_not_split_words(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        for a, b in zip(result.chunks, result.chunks[1:]):
            assert not (a.text[-1].isalpha() and b.text[0].isalpha())

    def test_indexes_are_sequential_from_zero(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        assert [c.index for c in result.chunks] == list(range(len(result.chunks)))

    def test_start_offsets_are_strictly_increasing(self):
        # Non-decreasing alone passes vacuously on a frozen offset (every
        # repeat equals the one before it, which technically is "sorted").
        # Strictly increasing is what actually catches the freeze.
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        _assert_strictly_increasing(result.chunks)

    def test_start_offsets_point_at_the_real_text(self):
        result = chunk(PROSE, strategy="recursive", size=200, overlap=20)
        _assert_literal_offsets(PROSE, result.chunks)

    def test_repetitive_text_gets_one_distinct_start_per_chunk(self):
        # "ab. " * 80 at size=8, overlap=6: the reproduction from the
        # CRITICAL 1 finding. With the frozen-cursor bug this collapsed 80
        # chunks down to 2 distinct starts.
        result = chunk(REPEAT, strategy="recursive", size=8, overlap=6)
        starts = [c.start for c in result.chunks]
        assert len(set(starts)) == len(result.chunks)

    def test_prose_repetitive_offsets_are_correct_and_distinct(self):
        # PROSE at size=60, overlap=40: the other CRITICAL 1 reproduction.
        # Chunk 60 used to report start=2675 when its true position was
        # 2734 -- the substring check alone can pass vacuously when a
        # chunk's text is very short, so distinct-starts is the assertion
        # that actually catches it.
        result = chunk(PROSE, strategy="recursive", size=60, overlap=40)
        starts = [c.start for c in result.chunks]
        assert len(set(starts)) == len(result.chunks)
        _assert_literal_offsets(PROSE, result.chunks)


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

    def test_start_offsets_point_at_the_real_text(self):
        result = chunk(STRUCTURED, strategy="structure", size=700, overlap=100)
        _assert_literal_offsets(STRUCTURED, result.chunks)

    def test_start_offsets_are_strictly_increasing(self):
        result = chunk(STRUCTURED, strategy="structure", size=700, overlap=100)
        _assert_strictly_increasing(result.chunks)

    def test_preserves_preamble_before_first_heading(self):
        # IMPORTANT 3: a title page / abstract preceding heading 1 used to
        # vanish silently -- _chunk_structure sliced from the first heading
        # onward and never looked back.
        preamble = "Title page. Front matter appears before section 1 starts."
        text = preamble + "\n" + STRUCTURED
        result = chunk(text, strategy="structure", size=700, overlap=100)
        assert any(preamble in c.text for c in result.chunks)
        assert result.sections_detected == 6
        assert result.fell_back is False


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

    def test_start_offsets_are_correct_never_silently_wrong(self):
        # CRITICAL 2 reproduction: SemanticChunker rejoins sentences with
        # normalised whitespace, so 5 of these 13 chunks are not a literal
        # substring of STRUCTURED at all. The old code's unqualified
        # text.find() fallback still returned a plausible-looking offset
        # (start=13, wrong content) for every one of them. Every offset
        # produced now must genuinely resolve to that chunk's real text --
        # there is no "flagged but wrong" middle ground to accept here.
        result = chunk(
            STRUCTURED, strategy="semantic", size=700, overlap=100,
            embeddings=FakeEmbeddings(), percentile=50,
        )
        assert len(result.chunks) >= 1
        _assert_offsets_correct_allowing_whitespace(STRUCTURED, result.chunks)
        _assert_strictly_increasing(result.chunks)


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

    def test_start_offsets_point_at_the_real_text(self):
        result = chunk(PROSE, strategy="parent", size=60, overlap=40)
        _assert_literal_offsets(PROSE, result.chunks)

    def test_start_offsets_are_strictly_increasing(self):
        result = chunk(PROSE, strategy="parent", size=60, overlap=40)
        _assert_strictly_increasing(result.chunks)

    def test_repetitive_text_gets_one_distinct_start_per_chunk(self):
        result = chunk(REPEAT, strategy="parent", size=8, overlap=6)
        starts = [c.start for c in result.chunks]
        assert len(set(starts)) == len(result.chunks)


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
