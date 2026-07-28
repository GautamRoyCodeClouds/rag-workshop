"""Query-time retrieval -- the half of RAG this app otherwise never runs.

Steps 1-5 of the app stop at "the vectors are in the database, here they
are" (see CLAUDE.md). This module exists purely to feed the transparency
panel: given a query, it shows the room *every* candidate ChromaDB
considered, not just the ones that won, and exactly which stage eliminated
each loser. The trace is the product here, not a debug afterthought -- so
`retrieve()` always returns a full `RetrievalTrace`, even for an empty
collection, rather than raising.

There is deliberately no LLM call anywhere in this file. Turning `selected`
into an answer (or "I don't know" when `answerable` is False) is a future
seam, not this module's job.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """One pool member and the full story of why it did or didn't win.

    Every candidate ChromaDB returned gets one of these, in descending
    similarity order -- including the ones that lost. That is the whole
    point of the panel: "here is what nearly made it, and why it didn't."
    """

    id: str
    text: str
    metadata: dict
    distance: float           # raw value Chroma returned
    similarity: float         # converted, see cosine_similarity_from_distance
    selected: bool
    rejected_reason: str      # "" | "below_threshold" | "not_top_k" | "mmr_redundant"
    mmr_score: float | None   # None when algorithm != "mmr"


@dataclass(frozen=True)
class Stage:
    """One timed step of the pipeline, for the panel's waterfall view."""

    name: str          # "embed_query" | "search" | "rank" | "filter" | "assemble"
    detail: dict       # whatever that stage needs to explain itself
    ms: float


@dataclass
class RetrievalTrace:
    """Everything the transparency panel needs to explain one query.

    Not frozen: retrieve() builds this incrementally as each stage runs,
    the same way session.py builds up per-session state.
    """

    query: str
    algorithm: str              # "similarity" | "mmr"
    top_k: int
    min_score: float
    mmr_lambda: float | None
    pool_size: int              # how many candidates were fetched before ranking
    query_vector_preview: list[float]   # first 8 components, plain floats
    query_vector_dims: int
    embed_model: str            # the model used for THIS query
    model_mismatch: list[str]   # distinct embed_model values among candidates that differ from the query's
    candidates: list[Candidate]
    selected: list[Candidate]
    stages: list[Stage]
    total_ms: float
    answerable: bool            # False => the caller must answer "I don't know"


def cosine_similarity_from_distance(distance: float) -> float:
    """Convert Chroma's cosine distance to a similarity in [-1, 1].

    Verified against a real ChromaDB EphemeralClient in test_retriever.py,
    not assumed: querying a record with the exact text it was embedded from
    returns distance ~2e-7 (floating-point noise, not exactly 0 -- see the
    test's comment), so similarity = 1 - distance lands at ~1.0. If a future
    chroma version ever returned squared or unnormalised distances that test
    would catch it; this function would then need to change, not just the
    test.

    Clamped because floating point can push a near-identical pair fractionally
    past +-1 (e.g. 1 - (-1e-8) = 1.00000001), and callers comparing this value
    against min_score should never see something outside a similarity's valid
    range.
    """
    similarity = 1.0 - float(distance)
    return max(-1.0, min(1.0, similarity))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity between two vectors.

    embedder.py normalises vectors to unit length at write time, so in
    practice this reduces to a dot product -- but mmr_select is documented as
    pure and independent of that invariant, so it divides by the norms rather
    than assuming them away. A zero vector (norm 0) can never usefully be
    "similar" to anything; returning 0.0 for that case avoids a ZeroDivisionError
    without pretending the comparison means something.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_greedy(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    k: int,
    lambda_: float,
) -> tuple[list[int], list[float]]:
    """Run greedy MMR, returning (selected indices, each pick's own score).

    Internal: mmr_select() (the public, pure, testable surface named in the
    spec) wraps this and throws the scores away. retrieve() needs the scores
    too, to explain *why* each pick won, so the loop is factored out here
    rather than duplicated.
    """
    n = len(candidate_vectors)
    k = max(0, min(k, n))
    sims_to_query = [_cosine_similarity(query_vector, v) for v in candidate_vectors]

    selected: list[int] = []
    scores: list[float] = []
    remaining = list(range(n))

    while len(selected) < k:
        if not selected:
            # First pick is a special case, not just the formula with an
            # empty selected set: the redundancy term is 0 for every
            # candidate when nothing has been picked yet, so ranking by
            # `lambda_ * sim - (1 - lambda_) * 0` ties everything at 0 the
            # moment lambda_ == 0 -- exactly the "maximise diversity alone"
            # case the spec calls out. The definition is unambiguous without
            # the formula: take whichever candidate is closest to the query.
            best_i = max(remaining, key=lambda i: sims_to_query[i])
            best_score = lambda_ * sims_to_query[best_i]
        else:
            def score_of(i: int) -> float:
                redundancy = max(
                    _cosine_similarity(candidate_vectors[i], candidate_vectors[s])
                    for s in selected
                )
                return lambda_ * sims_to_query[i] - (1.0 - lambda_) * redundancy

            best_i = max(remaining, key=score_of)
            best_score = score_of(best_i)

        selected.append(best_i)
        scores.append(best_score)
        remaining.remove(best_i)

    return selected, scores


def mmr_select(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """Greedy maximal marginal relevance. Returns indices into candidate_vectors.

    Pure and Chroma-free by design, so the algebra can be hand-checked against
    a small fixture without spinning up a collection -- see test_retriever.py's
    TestMmrSelect for the worked examples this is graded against.
    """
    selected, _scores = _mmr_greedy(query_vector, candidate_vectors, k, lambda_)
    return selected


def retrieve(
    collection,
    *,
    query: str,
    embeddings,
    top_k: int,
    min_score: float,
    algorithm: str = "similarity",
    mmr_lambda: float = 0.5,
    pool_multiplier: int = 4,
) -> RetrievalTrace:
    """Retrieve top_k chunks for query, with a full trace of every candidate.

    Every candidate ChromaDB's ANN search returned is present in
    trace.candidates, in descending similarity order, each carrying enough
    to explain its fate:

      - outside the pool entirely: never appears (ChromaDB's own ANN index
        already discarded it before this function ever saw it)
      - in the pool but not ranked into the top top_k: rejected_reason is
        "not_top_k" (similarity) or "mmr_redundant" (mmr)
      - ranked in, but below min_score: rejected_reason "below_threshold"
      - both ranked in and above min_score: selected=True

    An empty collection (or a threshold nothing clears) is not an error --
    it is a valid, `answerable=False` trace, matching the "I don't know"
    contract the caller must honour.
    """
    t_start = time.perf_counter()
    stages: list[Stage] = []

    # --- Stage 1: embed_query ------------------------------------------------
    # Must use the same model that wrote the vectors, or "similarity" is
    # comparing two different coordinate systems (see model_mismatch below,
    # which is the panel's guardrail for exactly that mistake).
    t0 = time.perf_counter()
    embed_model = getattr(embeddings, "model_name", "")
    query_vector = [float(x) for x in embeddings.embed_query(query)]
    stages.append(Stage(
        name="embed_query",
        detail={"model": embed_model, "dims": len(query_vector)},
        ms=(time.perf_counter() - t0) * 1000,
    ))

    # --- Stage 2: search ------------------------------------------------------
    # Fetch a pool bigger than top_k so ranking (especially MMR, which needs
    # runners-up to compute redundancy against) has something to work with.
    # max(..., 1): chroma's query() raises on n_results=0 (verified against
    # the installed 0.6.3, not assumed), which top_k=0 would otherwise hit.
    t0 = time.perf_counter()
    pool_requested = max(top_k * pool_multiplier, 1)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=pool_requested,
        # Documents/metadatas/distances render the panel's candidate table;
        # embeddings are fetched too because MMR needs the actual vectors to
        # score redundancy between candidates, not just their distance to
        # the query.
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    # Chroma 0.6.3 returns query() embeddings as a numpy.ndarray of
    # numpy.float64 (verified directly, same as store.py's read_records
    # docstring notes for get()). float() is cheap insurance against a future
    # chroma version or dtype handing back non-float-subclass numpy, same
    # reasoning as store.py's read_records.
    vectors = [[float(x) for x in row] for row in result["embeddings"][0]]
    pool_size = len(ids)
    stages.append(Stage(
        name="search",
        detail={"requested": pool_requested, "pool_size": pool_size},
        ms=(time.perf_counter() - t0) * 1000,
    ))

    candidates_raw = [
        {
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": float(distances[i]),
            "similarity": cosine_similarity_from_distance(distances[i]),
            "vector": vectors[i],
        }
        for i in range(pool_size)
    ]

    # --- Stage 3: rank ----------------------------------------------------
    # Picks which pool indices make the top_k cut. Filtering by min_score is
    # a separate stage below -- a candidate can be ranked in here and still
    # lose to the threshold, and the panel should show that as two distinct
    # decisions, not one.
    t0 = time.perf_counter()
    mmr_scores: dict[int, float] = {}
    if algorithm == "mmr":
        candidate_vectors = [c["vector"] for c in candidates_raw]
        selected_indices, selection_scores = _mmr_greedy(
            query_vector, candidate_vectors, top_k, mmr_lambda
        )
        mmr_scores.update(zip(selected_indices, selection_scores))
        # Runners-up MMR didn't pick still get an mmr_score, computed against
        # the *finished* selection -- the panel's whole reason for existing
        # is showing why a candidate lost, and "redundant with what was
        # actually chosen" is more honest than leaving this blank. This is
        # NOT the score that candidate had at its own turn in the greedy
        # loop (that used a smaller, still-growing selected set); recovering
        # that number honestly would mean re-running the loop step by step
        # and remembering every intermediate rejection, and it can only be
        # smaller than or equal to what is shown here, since redundancy
        # against a bigger finished set is never lower than redundancy
        # against a subset of it.
        query_sims = {
            i: _cosine_similarity(query_vector, candidate_vectors[i])
            for i in range(len(candidate_vectors))
        }
        for i in range(len(candidate_vectors)):
            if i in mmr_scores:
                continue
            redundancy = (
                max(_cosine_similarity(candidate_vectors[i], candidate_vectors[s])
                    for s in selected_indices)
                if selected_indices else 0.0
            )
            mmr_scores[i] = mmr_lambda * query_sims[i] - (1.0 - mmr_lambda) * redundancy
    else:
        # Plain similarity ranking: highest similarity first. Python's sort
        # is stable, so ties keep Chroma's own (already distance-ordered)
        # tie-break rather than an arbitrary one.
        order = sorted(
            range(pool_size), key=lambda i: candidates_raw[i]["similarity"], reverse=True
        )
        selected_indices = order[:top_k]
    stages.append(Stage(
        name="rank",
        detail={"algorithm": algorithm, "top_k": top_k, "ranked_in": len(selected_indices)},
        ms=(time.perf_counter() - t0) * 1000,
    ))

    # --- Stage 4: filter ------------------------------------------------------
    t0 = time.perf_counter()
    selected_indices_set = set(selected_indices)
    cleared_indices = [
        i for i in selected_indices if candidates_raw[i]["similarity"] >= min_score
    ]
    below_threshold_set = selected_indices_set - set(cleared_indices)
    stages.append(Stage(
        name="filter",
        detail={
            "min_score": min_score,
            "cleared": len(cleared_indices),
            "rejected": len(below_threshold_set),
        },
        ms=(time.perf_counter() - t0) * 1000,
    ))

    # --- Stage 5: assemble ------------------------------------------------
    t0 = time.perf_counter()

    def build(i: int) -> Candidate:
        raw = candidates_raw[i]
        if i in below_threshold_set:
            selected, reason = False, "below_threshold"
        elif i in selected_indices_set:
            selected, reason = True, ""
        else:
            selected, reason = False, ("mmr_redundant" if algorithm == "mmr" else "not_top_k")
        return Candidate(
            id=raw["id"],
            text=raw["text"],
            metadata=raw["metadata"],
            distance=raw["distance"],
            similarity=raw["similarity"],
            selected=selected,
            rejected_reason=reason,
            mmr_score=mmr_scores.get(i) if algorithm == "mmr" else None,
        )

    by_similarity = sorted(
        range(pool_size), key=lambda i: candidates_raw[i]["similarity"], reverse=True
    )
    built = {i: build(i) for i in range(pool_size)}
    candidates_list = [built[i] for i in by_similarity]
    # Selection order, not similarity order: for MMR this is the pick
    # sequence (relevance first, then each successive diversity trade-off),
    # which is exactly what the panel wants to narrate. For plain similarity
    # ranking the two orders already coincide.
    selected_list = [built[i] for i in cleared_indices]

    model_mismatch = sorted({
        raw["metadata"].get("embed_model", "") for raw in candidates_raw
    } - {embed_model})

    answerable = len(selected_list) > 0

    stages.append(Stage(
        name="assemble",
        detail={"candidates": pool_size, "selected": len(selected_list)},
        ms=(time.perf_counter() - t0) * 1000,
    ))

    return RetrievalTrace(
        query=query,
        algorithm=algorithm,
        top_k=top_k,
        min_score=min_score,
        mmr_lambda=float(mmr_lambda) if algorithm == "mmr" else None,
        pool_size=pool_size,
        query_vector_preview=query_vector[:8],
        query_vector_dims=len(query_vector),
        embed_model=embed_model,
        model_mismatch=model_mismatch,
        candidates=candidates_list,
        selected=selected_list,
        stages=stages,
        total_ms=(time.perf_counter() - t_start) * 1000,
        answerable=answerable,
    )
