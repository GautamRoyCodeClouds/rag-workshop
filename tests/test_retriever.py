"""Retriever tests, run against an in-process ephemeral Chroma client.

Same style as test_store.py: no server needed, but the real chromadb API is
exercised rather than a mock of it -- the distance->similarity conversion in
particular is a claim about what *Chroma* returns, not about this module's
own arithmetic, and only a real collection can confirm it.

Most fixtures here use hand-picked *unit* vectors in 2D so every similarity
and MMR score can be checked by actual arithmetic (the numbers are written
out in each test's docstring/comments), per this repo's testing standard:
assert exact values, never just shapes.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest

from app.pipeline.retriever import (
    Candidate,
    RetrievalTrace,
    Stage,
    cosine_similarity_from_distance,
    mmr_select,
    retrieve,
)

# --------------------------------------------------------------------------
# Shared fixtures / fakes
# --------------------------------------------------------------------------


@pytest.fixture
def collection():
    # See test_store.py: EphemeralClient() instances share one process-wide
    # "ephemeral" system, so a fixed collection name would silently reattach
    # to a previous test's data. A unique name per test keeps this hermetic.
    client = chromadb.EphemeralClient()
    from app.pipeline.store import get_collection
    return get_collection(client, f"test-retriever-{uuid.uuid4().hex}")


class FakeEmbeddings:
    """Stands in for HuggingFaceEmbeddings without loading a real model.

    retrieve() only ever touches .model_name and .embed_query(text) on the
    embeddings object it is given -- never .embed_documents, since document
    vectors already live in the collection by the time retrieval runs. This
    fake returns a fixed vector regardless of the query text, which is
    exactly what lets these tests hand-compute exact expected scores: the
    query vector is under the test's full control.
    """

    def __init__(self, model_name: str, query_vector: list[float]):
        self.model_name = model_name
        self._query_vector = query_vector

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector


def add(collection, entries: list[tuple[str, str, list[float], dict]]) -> None:
    """entries: (id, text, vector, metadata) tuples, written straight to Chroma.

    Bypasses write_chunks() deliberately: these tests need exact, known
    vectors to hand-compute similarity and MMR scores against, not vectors
    produced by an actual chunking/embedding run.
    """
    collection.add(
        ids=[e[0] for e in entries],
        embeddings=[e[2] for e in entries],
        documents=[e[1] for e in entries],
        metadatas=[e[3] for e in entries],
    )


def assert_no_numpy(trace: RetrievalTrace) -> None:
    for c in trace.candidates:
        assert type(c.distance) is float
        assert type(c.similarity) is float
        if c.mmr_score is not None:
            assert type(c.mmr_score) is float
    for v in trace.query_vector_preview:
        assert type(v) is float
    assert type(trace.total_ms) is float
    for s in trace.stages:
        assert type(s.ms) is float


# --------------------------------------------------------------------------
# cosine_similarity_from_distance
# --------------------------------------------------------------------------


class TestCosineSimilarityFromDistance:
    def test_zero_distance_is_full_similarity(self):
        assert cosine_similarity_from_distance(0.0) == 1.0

    def test_max_cosine_distance_is_minus_one(self):
        # Cosine distance ranges 0..2 (1 - (-1) == 2 for opposite vectors).
        assert cosine_similarity_from_distance(2.0) == -1.0

    def test_clamps_above_one(self):
        # A tiny negative distance (floating point noise) must not produce a
        # similarity that reads as "more than identical".
        assert cosine_similarity_from_distance(-0.01) == 1.0

    def test_clamps_below_minus_one(self):
        assert cosine_similarity_from_distance(2.5) == -1.0

    @pytest.mark.slow
    def test_against_real_chroma_identical_text_is_near_one(self, collection):
        """The claim this whole conversion rests on, checked against real
        Chroma rather than assumed.

        Embeds one document, writes it, queries with the *identical* text.
        If the installed Chroma ever returned squared or unnormalised
        distances, this would fail here rather than silently mis-scoring
        every real query.
        """
        from app.pipeline.embedder import build_embeddings

        embeddings = build_embeddings()
        text = "Employees accrue one day of leave per month worked."
        vector = embeddings.embed_documents([text])[0]
        add(collection, [("doc-1", text, vector, {"embed_model": embeddings.model_name})])

        query_vector = embeddings.embed_query(text)
        result = collection.query(
            query_embeddings=[query_vector], n_results=1, include=["distances"]
        )
        distance = result["distances"][0][0]

        assert distance == pytest.approx(0.0, abs=1e-3)
        assert cosine_similarity_from_distance(distance) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# mmr_select -- pure, hand-computed fixture
# --------------------------------------------------------------------------

# Query and three unit vectors chosen so every pairwise cosine similarity is a
# clean dot product (all four are unit length, so cosine sim == dot product,
# no square roots needed to check by hand):
#
#   q = (1.0, 0.0)
#   A = (0.8, 0.6)   sim(q, A) = 0.8
#   B = (0.6, 0.8)   sim(q, B) = 0.6
#   C = (0.0, 1.0)   sim(q, C) = 0.0
#
#   sim(A, B) = 0.8*0.6 + 0.6*0.8 = 0.96   (A and B are near-redundant)
#   sim(A, C) = 0.8*0.0 + 0.6*1.0 = 0.6
#   sim(B, C) = 0.6*0.0 + 0.8*1.0 = 0.8
#
# Candidates are stored in the order [A, B, C] -> indices [0, 1, 2].
Q = [1.0, 0.0]
VEC_A = [0.8, 0.6]
VEC_B = [0.6, 0.8]
VEC_C = [0.0, 1.0]
POOL = [VEC_A, VEC_B, VEC_C]


class TestMmrSelect:
    def test_lambda_one_reduces_to_pure_similarity_order(self):
        # With lambda_=1.0 the redundancy term is multiplied by (1 - 1) = 0,
        # so every pick is just "highest remaining sim(query, d)" -- exactly
        # plain similarity ranking. Similarity order here is A(0.8) > B(0.6)
        # > C(0.0), i.e. indices [0, 1, 2].
        assert mmr_select(Q, POOL, k=3, lambda_=1.0) == [0, 1, 2]

    def test_lambda_zero_diverges_from_similarity_ranking(self):
        # Plain top-2 by similarity is [A, B] (indices [0, 1]).
        # At lambda_=0.0 (diversity only): pick 1 is still A (the first pick
        # is always argmax sim(query, d) -- see docstring). Pick 2 scores:
        #   score(B) = -sim(A, B) = -0.96
        #   score(C) = -sim(A, C) = -0.6
        # C wins (-0.6 > -0.96), so MMR picks [A, C] = [0, 2] -- a genuinely
        # different set than similarity ranking, proving the fixture actually
        # exercises the diversity term rather than agreeing by coincidence.
        similarity_top_2 = sorted(range(3), key=lambda i: _sim(Q, POOL[i]), reverse=True)[:2]
        assert similarity_top_2 == [0, 1]

        assert mmr_select(Q, POOL, k=2, lambda_=0.0) == [0, 2]
        assert mmr_select(Q, POOL, k=2, lambda_=0.0) != similarity_top_2

    def test_k_larger_than_pool_returns_every_index_once(self):
        result = mmr_select(Q, POOL, k=10, lambda_=0.5)
        assert sorted(result) == [0, 1, 2]

    def test_k_zero_returns_nothing(self):
        assert mmr_select(Q, POOL, k=0, lambda_=0.5) == []

    def test_returns_indices_not_vectors(self):
        result = mmr_select(Q, POOL, k=2, lambda_=0.5)
        assert all(isinstance(i, int) for i in result)


def _sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# retrieve() -- ranking, filtering, every-candidate-present
# --------------------------------------------------------------------------

# A second fixture, along the same axes as the MMR one but simpler (no need
# for redundancy here): four unit vectors at decreasing similarity to the
# query, so top_k / min_score behaviour has an unambiguous expected outcome.
#
#   q  = (1, 0)
#   r1 = (1.0, 0.0)        sim = 1.0
#   r2 = (0.8, 0.6)        sim = 0.8
#   r3 = (0.6, 0.8)        sim = 0.6
#   r4 = (0.0, 1.0)        sim = 0.0
RANK_Q = [1.0, 0.0]
RANK_VECTORS = {
    "r1": [1.0, 0.0],
    "r2": [0.8, 0.6],
    "r3": [0.6, 0.8],
    "r4": [0.0, 1.0],
}
MODEL = "modelQ"


def populate_ranking_fixture(collection, embed_model=MODEL):
    add(collection, [
        (rid, f"text for {rid}", vec, {"embed_model": embed_model, "chunk_index": i})
        for i, (rid, vec) in enumerate(RANK_VECTORS.items())
    ])


class TestRetrieveSimilarityRanking:
    def test_every_pool_candidate_present_in_descending_similarity_order(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
            top_k=2, min_score=-1.0, algorithm="similarity", pool_multiplier=2,
        )
        assert trace.pool_size == 4
        assert [c.id for c in trace.candidates] == ["r1", "r2", "r3", "r4"]
        assert [c.similarity for c in trace.candidates] == pytest.approx(
            [1.0, 0.8, 0.6, 0.0], abs=1e-6
        )

    def test_top_k_selected_rest_marked_not_top_k(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
            top_k=2, min_score=-1.0, algorithm="similarity", pool_multiplier=2,
        )
        by_id = {c.id: c for c in trace.candidates}
        assert by_id["r1"].selected and by_id["r1"].rejected_reason == ""
        assert by_id["r2"].selected and by_id["r2"].rejected_reason == ""
        assert not by_id["r3"].selected and by_id["r3"].rejected_reason == "not_top_k"
        assert not by_id["r4"].selected and by_id["r4"].rejected_reason == "not_top_k"
        assert [c.id for c in trace.selected] == ["r1", "r2"]
        assert trace.answerable is True

    def test_below_threshold_rejects_even_a_top_k_pick(self, collection):
        populate_ranking_fixture(collection)
        # r2's similarity (0.8) is ranked into the top 2 but falls below 0.9.
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
            top_k=2, min_score=0.9, algorithm="similarity", pool_multiplier=2,
        )
        by_id = {c.id: c for c in trace.candidates}
        assert by_id["r1"].selected
        assert not by_id["r2"].selected
        assert by_id["r2"].rejected_reason == "below_threshold"
        assert by_id["r3"].rejected_reason == "not_top_k"  # unaffected by threshold
        assert [c.id for c in trace.selected] == ["r1"]
        assert trace.answerable is True

    def test_nothing_clearing_the_threshold_is_not_answerable(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
            top_k=2, min_score=1.5, algorithm="similarity", pool_multiplier=2,
        )
        assert trace.selected == []
        assert trace.answerable is False
        # Still a full, valid trace -- not an empty/error result.
        assert trace.pool_size == 4
        by_id = {c.id: c for c in trace.candidates}
        assert by_id["r1"].rejected_reason == "below_threshold"
        assert by_id["r2"].rejected_reason == "below_threshold"
        assert by_id["r3"].rejected_reason == "not_top_k"

    def test_pool_multiplier_changes_how_many_candidates_are_fetched(self, collection):
        add(collection, [
            (f"e{i}", f"text {i}", vec, {"embed_model": MODEL})
            for i, vec in enumerate([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                      [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        ])
        embeddings = FakeEmbeddings(MODEL, [1.0, 0.0, 0.0, 0.0, 0.0])

        narrow = retrieve(collection, query="q", embeddings=embeddings,
                           top_k=1, min_score=-1.0, pool_multiplier=1)
        wide = retrieve(collection, query="q", embeddings=embeddings,
                         top_k=1, min_score=-1.0, pool_multiplier=3)

        search_detail = {s.name: s.detail for s in narrow.stages}["search"]
        assert search_detail["requested"] == 1  # top_k(1) * pool_multiplier(1)
        assert narrow.pool_size == 1

        wide_detail = {s.name: s.detail for s in wide.stages}["search"]
        assert wide_detail["requested"] == 3  # top_k(1) * pool_multiplier(3)
        assert wide.pool_size == 3

    def test_query_vector_preview_is_first_eight_components(self, collection):
        query_vector = [float(i) / 10 for i in range(10)]
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, query_vector),
            top_k=1, min_score=-1.0,
        )
        assert trace.query_vector_dims == 10
        assert trace.query_vector_preview == pytest.approx(query_vector[:8])
        assert len(trace.query_vector_preview) == 8

    def test_stage_names_and_order(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=2, min_score=-1.0)
        assert [s.name for s in trace.stages] == [
            "embed_query", "search", "rank", "filter", "assemble",
        ]

    def test_similarity_algorithm_never_sets_an_mmr_score(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=2, min_score=-1.0, algorithm="similarity")
        assert trace.mmr_lambda is None
        assert all(c.mmr_score is None for c in trace.candidates)

    def test_no_numpy_types_survive_into_the_trace(self, collection):
        populate_ranking_fixture(collection)
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=2, min_score=-1.0)
        assert_no_numpy(trace)


class TestRetrieveEmptyCollection:
    def test_empty_collection_is_a_valid_unanswerable_trace(self, collection):
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=3, min_score=0.0)
        assert trace.pool_size == 0
        assert trace.candidates == []
        assert trace.selected == []
        assert trace.answerable is False
        assert trace.model_mismatch == []


class TestModelMismatch:
    def test_lists_distinct_models_that_differ_from_the_query_model(self, collection):
        add(collection, [
            ("r1", "t1", [1.0, 0.0], {"embed_model": MODEL}),        # matches query model
            ("r2", "t2", [0.9, 0.1], {"embed_model": "modelOld"}),
            ("r3", "t3", [0.8, 0.2], {"embed_model": "modelOlder"}),
            ("r4", "t4", [0.7, 0.3], {"embed_model": "modelOld"}),    # repeats modelOld
        ])
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=4, min_score=-1.0, pool_multiplier=1)
        assert trace.model_mismatch == ["modelOld", "modelOlder"]

    def test_all_matching_models_is_an_empty_mismatch_list(self, collection):
        populate_ranking_fixture(collection)  # every record uses MODEL
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, RANK_Q),
                          top_k=4, min_score=-1.0)
        assert trace.model_mismatch == []


# --------------------------------------------------------------------------
# retrieve() with algorithm="mmr" -- exact, hand-computed scores
# --------------------------------------------------------------------------


class TestRetrieveMmr:
    def _populate(self, collection):
        add(collection, [
            ("a", "doc a", VEC_A, {"embed_model": MODEL}),
            ("b", "doc b", VEC_B, {"embed_model": MODEL}),
            ("c", "doc c", VEC_C, {"embed_model": MODEL}),
        ])

    def test_selection_order_and_hand_computed_scores(self, collection):
        """Same fixture as TestMmrSelect, now checked end to end through
        retrieve(), including the scores mmr_select() itself doesn't expose.

        lambda_=0.0: pick 1 is A (score = lambda_ * sim(q,A) = 0.0). Pick 2:
        score(B) = -sim(A,B) = -0.96, score(C) = -sim(A,C) = -0.6 -- C wins.
        B is left over; its shown mmr_score is computed against the *finished*
        selection {A, C}: -max(sim(B,A), sim(B,C)) = -max(0.96, 0.8) = -0.96.
        """
        self._populate(collection)
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, Q),
            # pool_multiplier=2, not 1: the pool is top_k * multiplier, so a
            # multiplier of 1 would fetch only the 2 *most similar* docs (A and
            # B) and C could never be chosen -- MMR cannot select a candidate
            # that retrieval never handed it. The whole point of this fixture is
            # that C is available and diversity prefers it over B.
            top_k=2, min_score=-1.0, algorithm="mmr", mmr_lambda=0.0, pool_multiplier=2,
        )

        assert [c.id for c in trace.selected] == ["a", "c"]
        assert trace.selected[0].mmr_score == pytest.approx(0.0, abs=1e-6)
        assert trace.selected[1].mmr_score == pytest.approx(-0.6, abs=1e-6)

        by_id = {c.id: c for c in trace.candidates}
        assert by_id["b"].selected is False
        assert by_id["b"].rejected_reason == "mmr_redundant"
        assert by_id["b"].mmr_score == pytest.approx(-0.96, abs=1e-6)
        assert trace.mmr_lambda == pytest.approx(0.0)

    def test_first_pick_score_at_an_intermediate_lambda(self, collection):
        """The first pick's own mmr_score, at a lambda where it is NOT zero.

        The lambda_=0.0 test above asserts selected[0].mmr_score == 0.0, which
        is the correct value there -- and therefore cannot distinguish the real
        `lambda_ * sim(q, first)` from a hardcoded 0.0. Mutating that
        expression to `0.0` passed all 25 tests. The first pick's score is
        shown in the transparency panel, so a wrong number there is a quiet
        lie told to a room.

        At lambda_=0.5, with the pool covering A, B and C:
          pick 1 = A (argmax sim to q), score = 0.5 * 0.8            =  0.4
          pick 2: score(B) = 0.5*0.6 - 0.5*sim(A,B) = 0.3 - 0.48     = -0.18
                  score(C) = 0.5*0.0 - 0.5*sim(A,C) = 0.0 - 0.30     = -0.30
                  B wins
          leftover C, scored against the finished {A, B}:
                  0.5*0.0 - 0.5*max(sim(C,A), sim(C,B)) = -0.5*0.8   = -0.4
        """
        self._populate(collection)
        trace = retrieve(
            collection, query="q", embeddings=FakeEmbeddings(MODEL, Q),
            top_k=2, min_score=-1.0, algorithm="mmr", mmr_lambda=0.5, pool_multiplier=2,
        )

        assert [c.id for c in trace.selected] == ["a", "b"]
        assert trace.selected[0].mmr_score == pytest.approx(0.4, abs=1e-6)
        assert trace.selected[1].mmr_score == pytest.approx(-0.18, abs=1e-6)
        assert {c.id: c for c in trace.candidates}["c"].mmr_score == pytest.approx(
            -0.4, abs=1e-6
        )
        # Worth stating plainly: at lambda_=0.5 MMR agrees with plain similarity
        # ranking on this fixture ([a, b]), while at lambda_=0.0 it diverges to
        # [a, c]. Together the two tests show lambda actually tuning the
        # relevance/diversity trade-off rather than being read and discarded.

    def test_answerable_true_when_mmr_selects_anything(self, collection):
        self._populate(collection)
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, Q),
                          top_k=2, min_score=-1.0, algorithm="mmr", mmr_lambda=0.0)
        assert trace.answerable is True

    def test_no_numpy_types_survive_mmr_either(self, collection):
        self._populate(collection)
        trace = retrieve(collection, query="q", embeddings=FakeEmbeddings(MODEL, Q),
                          top_k=2, min_score=-1.0, algorithm="mmr", mmr_lambda=0.5)
        assert_no_numpy(trace)
