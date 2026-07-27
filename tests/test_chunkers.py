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
