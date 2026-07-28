"""Generator tests.

Nothing here talks to a live Ollama -- every transport is a fake `opener`
injected into `probe`/`stream_answer`, so the suite is hermetic and fast, and
still exercises exactly the failure paths a real Ollama produces (connection
refused, timeout, HTTP error, malformed body, model absent).

The streaming reasoning stripper gets its own boundary-sweep test, because a
per-chunk regex would pass every *convenient* split and still leak reasoning
on an inconvenient one -- see the module docstring in generator.py.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from app.pipeline.generator import (
    GeneratorStatus,
    _StreamingReasoningStripper,
    build_prompt,
    probe,
    stream_answer,
    strip_reasoning,
)


class _FakeResponse:
    """Stands in for `http.client.HTTPResponse`: a context manager that
    supports `.read()` (probe) and iteration over byte lines (stream_answer).
    """

    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self._body = body
        self._lines = lines or []

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return iter(self._lines)


def _tags_response(model_names: list[str]) -> _FakeResponse:
    body = json.dumps({"models": [{"name": name} for name in model_names]}).encode()
    return _FakeResponse(body=body)


def _raising_opener(exc: BaseException):
    def opener(request, timeout):
        raise exc
    return opener


def _fixed_opener(response: _FakeResponse):
    def opener(request, timeout):
        return response
    return opener


# --- probe --------------------------------------------------------------

def test_probe_reports_available_when_model_is_present():
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b",
        opener=_fixed_opener(_tags_response(["deepseek-r1:1.5b", "llama2"])),
    )
    assert status == GeneratorStatus(
        True, "deepseek-r1:1.5b", "http://host.docker.internal:11434", ""
    )


def test_probe_reports_connection_refused():
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b",
        opener=_raising_opener(ConnectionRefusedError("[Errno 111] Connection refused")),
    )
    assert status.available is False
    # A presenter reading this mid-talk needs the actual fix, not just "it failed".
    assert "OLLAMA_HOST" in status.detail
    assert "host.docker.internal:11434" in status.detail


def test_probe_reports_timeout():
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b", timeout=2.0,
        opener=_raising_opener(TimeoutError("timed out")),
    )
    assert status.available is False
    assert "2.0" in status.detail
    assert "did not respond" in status.detail


def test_probe_reports_http_error():
    http_error = urllib.error.HTTPError(
        "http://host.docker.internal:11434/api/tags", 500, "Internal Server Error",
        hdrs=None, fp=None,
    )
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b",
        opener=_raising_opener(http_error),
    )
    assert status.available is False
    assert "500" in status.detail


def test_probe_reports_malformed_json():
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b",
        opener=_fixed_opener(_FakeResponse(body=b"not json at all")),
    )
    assert status.available is False
    assert "JSON" in status.detail


def test_probe_reports_reachable_server_missing_the_model():
    status = probe(
        "http://host.docker.internal:11434", "deepseek-r1:1.5b",
        opener=_fixed_opener(_tags_response(["llama2", "mistral"])),
    )
    assert status.available is False
    # The five failure states must be genuinely distinguishable -- this one
    # in particular must not collide with the "unreachable" wording above,
    # since the fix ("ollama pull ...") is completely different.
    assert "deepseek-r1:1.5b" in status.detail
    assert "pull" in status.detail
    assert "OLLAMA_HOST" not in status.detail


# --- strip_reasoning (whole string) --------------------------------------

def test_strip_reasoning_with_no_think_block_is_unchanged():
    assert strip_reasoning("plain answer, nothing hidden") == "plain answer, nothing hidden"


def test_strip_reasoning_removes_one_complete_block():
    assert strip_reasoning("<think>secret plan</think>the answer") == "the answer"


def test_strip_reasoning_removes_multiple_blocks():
    text = "<think>a</think>first<think>b</think>second"
    assert strip_reasoning(text) == "firstsecond"


def test_strip_reasoning_drops_an_unterminated_block_entirely():
    # The model got cut off mid-thought: no closing tag ever arrives. The
    # reasoning must not leak just because it was never closed.
    assert strip_reasoning("visible<think>unfinished reasoning with no close") == "visible"


def test_strip_reasoning_handles_content_immediately_after_close_tag():
    assert strip_reasoning("<think>r</think>answer right after") == "answer right after"


# --- streaming stripper: boundary sweep ----------------------------------

SAMPLE_WITH_ONE_BLOCK = "before <think>hidden reasoning here</think> after"
SAMPLE_WITH_TWO_BLOCKS = "<think>a</think>mid<think>b</think>end"


@pytest.mark.parametrize("split_at", range(len(SAMPLE_WITH_ONE_BLOCK) + 1))
def test_streaming_stripper_matches_reference_at_every_split_point_one_block(split_at):
    """Feed the text split into two fragments at every possible boundary.

    A stripper that resets its buffer per-fragment (the mutation this test
    exists to catch) passes whichever splits happen to fall outside the tag,
    and fails on the ones that land inside "<think>" or "</think>" -- so
    sweeping every index is what actually exercises the boundary logic,
    rather than the one or two convenient splits a hand-picked test would use.
    """
    first, second = SAMPLE_WITH_ONE_BLOCK[:split_at], SAMPLE_WITH_ONE_BLOCK[split_at:]
    stripper = _StreamingReasoningStripper()
    result = stripper.feed(first) + stripper.feed(second) + stripper.finish()
    assert result == strip_reasoning(SAMPLE_WITH_ONE_BLOCK) == "before  after"


@pytest.mark.parametrize("split_at", range(len(SAMPLE_WITH_TWO_BLOCKS) + 1))
def test_streaming_stripper_matches_reference_at_every_split_point_two_blocks(split_at):
    first, second = SAMPLE_WITH_TWO_BLOCKS[:split_at], SAMPLE_WITH_TWO_BLOCKS[split_at:]
    stripper = _StreamingReasoningStripper()
    result = stripper.feed(first) + stripper.feed(second) + stripper.finish()
    assert result == strip_reasoning(SAMPLE_WITH_TWO_BLOCKS) == "midend"


def test_streaming_stripper_splits_the_tag_itself_one_character_at_a_time():
    """The extreme case of a split boundary: every fragment is a single
    character, so "<think>" arrives as seven separate feed() calls.
    """
    text = "x<think>y</think>z"
    stripper = _StreamingReasoningStripper()
    out = "".join(stripper.feed(ch) for ch in text) + stripper.finish()
    assert out == "xz"


def test_streaming_stripper_emits_nothing_for_an_unterminated_block():
    stripper = _StreamingReasoningStripper()
    out = stripper.feed("visible<think>never closes, ") + stripper.feed("cut off here")
    out += stripper.finish()
    assert out == "visible"


def test_streaming_stripper_finish_drops_a_pending_close_tag_fragment_while_still_in_think():
    """An unterminated block whose tail happens to look like the start of
    "</think>" (e.g. "</th") must still emit nothing: the buffered fragment
    is dropped by finish() because the stripper is still inside the think
    block, not released just because it resembles a tag prefix. This is the
    one case that distinguishes "finish() checks _in_think" from "finish()
    always releases whatever is pending" -- the latter mutation passes every
    other unterminated-block test here because those leave nothing pending.
    """
    stripper = _StreamingReasoningStripper()
    visible = stripper.feed("visible<think>hidden reasoning </th")
    assert visible == "visible"
    assert stripper.finish() == ""


def test_streaming_stripper_flushes_a_false_alarm_prefix_at_end_of_stream():
    """"<th" that never completes into "<think>" is ordinary content, not a
    tag -- it must be released by finish(), not silently swallowed.
    """
    stripper = _StreamingReasoningStripper()
    out = stripper.feed("price is <th") + stripper.feed("under $5")
    out += stripper.finish()
    assert out == "price is <thunder $5"


# --- build_prompt ----------------------------------------------------------

def test_build_prompt_instructs_context_only_and_honesty_about_not_knowing():
    prompt = build_prompt("What is the leave policy?", [])
    lower = prompt.lower()
    assert "only" in lower and "context" in lower
    assert "do not know" in lower or "don't know" in lower


def test_build_prompt_includes_chunk_text_and_page_citation():
    chunks = [
        {"text": "Employees get 20 days of annual leave.",
         "metadata": {"page": 4, "source": "handbook.pdf"}},
    ]
    prompt = build_prompt("How much annual leave?", chunks)
    assert "Employees get 20 days of annual leave." in prompt
    assert "handbook.pdf" in prompt
    assert "page 4" in prompt
    assert "How much annual leave?" in prompt


def test_build_prompt_with_no_chunks_says_so_rather_than_inventing_context():
    prompt = build_prompt("anything", [])
    assert "no matching context" in prompt.lower()


# --- stream_answer -----------------------------------------------------------

def _ndjson_lines(*events: dict) -> list[bytes]:
    return [json.dumps(e).encode() + b"\n" for e in events]


def test_stream_answer_yields_plain_text_with_no_think_block():
    lines = _ndjson_lines(
        {"response": "The ", "done": False},
        {"response": "answer.", "done": False},
        {"response": "", "done": True},
    )
    deltas = list(stream_answer(
        "http://x", "deepseek-r1:1.5b", "prompt",
        opener=_fixed_opener(_FakeResponse(lines=lines)),
    ))
    assert "".join(deltas) == "The answer."


def test_stream_answer_strips_a_think_block_split_across_many_deltas():
    reasoning_and_answer = "<think>let me consider this</think>Final answer here."
    # Split into small, arbitrary fragments to mimic real per-token streaming.
    fragments = [reasoning_and_answer[i:i + 3] for i in range(0, len(reasoning_and_answer), 3)]
    events = [{"response": frag, "done": False} for frag in fragments]
    events.append({"response": "", "done": True})
    deltas = list(stream_answer(
        "http://x", "deepseek-r1:1.5b", "prompt",
        opener=_fixed_opener(_FakeResponse(lines=_ndjson_lines(*events))),
    ))
    assert "".join(deltas) == "Final answer here."
    assert "consider" not in "".join(deltas)


def test_stream_answer_returns_nothing_when_connection_refused():
    deltas = list(stream_answer(
        "http://x", "deepseek-r1:1.5b", "prompt",
        opener=_raising_opener(ConnectionRefusedError("refused")),
    ))
    assert deltas == []


def test_stream_answer_stops_cleanly_on_mid_stream_failure():
    class _DyingIter:
        def __init__(self):
            self._sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self._sent:
                self._sent = True
                return json.dumps({"response": "partial", "done": False}).encode() + b"\n"
            raise ConnectionResetError("connection reset by peer")

    class _DyingResponse(_FakeResponse):
        def __iter__(self):
            return _DyingIter()

    deltas = list(stream_answer(
        "http://x", "deepseek-r1:1.5b", "prompt",
        opener=_fixed_opener(_DyingResponse()),
    ))
    # Whatever arrived before the failure is kept; the generator just stops,
    # it does not raise into whatever is consuming it (an SSE handler).
    assert deltas == ["partial"]


def test_stream_answer_skips_unparseable_lines_without_raising():
    lines = [b"not json\n"] + _ndjson_lines({"response": "ok", "done": True})
    deltas = list(stream_answer(
        "http://x", "deepseek-r1:1.5b", "prompt",
        opener=_fixed_opener(_FakeResponse(lines=lines)),
    ))
    assert deltas == ["ok"]
