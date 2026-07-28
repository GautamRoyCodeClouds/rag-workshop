"""Optional answer generation over Ollama -- the bonus half of the demo.

Retrieval is the star of this workshop; generation is a bonus shown *if* it
works. The hard requirement, stated in CLAUDE.md and worth repeating here: the
app must behave correctly with Ollama absent, unreachable, or missing the
model. Nothing in this module raises for any of those states -- they are
ordinary, reportable outcomes (`GeneratorStatus.available=False`), never
exceptions that would surface as a 500 to someone in the room.

Two environment facts drive the shape of the code below:

- Ollama runs on the *host*, bound to 127.0.0.1 by default. From inside the
  container `http://host.docker.internal:11434` is reachable only once the
  presenter sets `OLLAMA_HOST=0.0.0.0` and restarts Ollama (and compose adds
  `extra_hosts: host.docker.internal:host-gateway`). Until then, every call
  here fails at the transport layer -- that is the *expected* default state,
  not a bug, so `probe`'s unreachable message names the fix.
- `deepseek-r1:1.5b` is a reasoning model: a cold call took 15.8s for six
  tokens, and its output is wrapped in `<think>...</think>` blocks that must
  never reach the room. Both facts show up below: `stream_answer`'s default
  timeout is generous, and reasoning is stripped incrementally from the
  stream itself, not from the joined result after the fact.

Only `urllib` is used for HTTP, deliberately -- `requests`/`httpx` are already
in the environment, but this project is offline-first and treats every new
runtime dependency as something to justify. `httpx` is a dev/test dependency
only and must stay that way.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass

# The opener is a seam for tests, not a production knob: real callers never
# pass one and get genuine `urllib.request.urlopen`. Fake transports in tests
# implement the same two-argument shape (request, timeout) and return
# something usable as a context manager that yields bytes -- exactly what
# `http.client.HTTPResponse` already is -- so no monkeypatching of urllib's
# own module state is needed to make this hermetic.
Opener = Callable[[urllib.request.Request, float], "http.client.HTTPResponse"]


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


@dataclass(frozen=True)
class GeneratorStatus:
    available: bool
    model: str
    base_url: str
    detail: str  # human-readable reason when unavailable; "" when available


def probe(
    base_url: str,
    model: str,
    timeout: float = 2.0,
    *,
    opener: Opener = _default_opener,
) -> GeneratorStatus:
    """Ask Ollama for its tag list and report whether `model` is present.

    Never raises. Every way this can go wrong -- connection refused, DNS
    failure, timeout, an HTTP error status, a malformed response body, or a
    server that answers but does not have `model` pulled -- becomes
    `available=False` with a `detail` a presenter can act on, rather than an
    exception the caller has to remember to catch.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")

    try:
        with opener(request, timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # The server answered but with a non-2xx status -- Ollama itself is
        # unwell, which is a different fix than "it isn't running at all".
        return GeneratorStatus(
            False, model, base_url,
            f"Ollama at {base_url} returned HTTP {exc.code} for {url} -- "
            "the server is reachable but reported an error; check its logs.",
        )
    except TimeoutError as exc:
        # A bare TimeoutError (as opposed to a URLError wrapping one, handled
        # by the OSError branch below) is what a fake transport in tests
        # raises directly to simulate "no response within the deadline".
        return GeneratorStatus(
            False, model, base_url,
            f"Ollama did not respond within {timeout}s at {url} -- "
            f"it may be starting up or overloaded ({exc}).",
        )
    except OSError as exc:
        # Covers connection-refused, DNS failure, and urllib's own
        # URLError (a OSError subclass, sometimes itself wrapping a timeout)
        # in one branch: from the outside these all look like "nothing is
        # listening there", and the fix is the same regardless of which OS
        # error produced it.
        return GeneratorStatus(
            False, model, base_url,
            f"Ollama unreachable at {base_url} ({exc}) -- if it is running "
            "on the host, set OLLAMA_HOST=0.0.0.0 and restart it, and make "
            "sure compose has extra_hosts: host.docker.internal:host-gateway.",
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return GeneratorStatus(
            False, model, base_url,
            f"Ollama at {base_url} returned unparseable JSON from {url} "
            f"({exc}) -- is something other than Ollama listening on that port?",
        )

    # Ollama's /api/tags entries are already "name:tag" (e.g.
    # "deepseek-r1:1.5b"), matching the format callers pass as `model` -- no
    # separate tag-splitting needed.
    names = {entry.get("name") for entry in payload.get("models", [])}
    if model not in names:
        return GeneratorStatus(
            False, model, base_url,
            f"Ollama is reachable at {base_url} but has no model named "
            f"'{model}' -- run `ollama pull {model}` on the host.",
        )

    return GeneratorStatus(True, model, base_url, "")


# --- Reasoning stripping -----------------------------------------------------

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _longest_boundary_overlap(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is also a prefix of `tag`.

    Capped at len(tag) - 1: a full match would already have been found by
    `str.find` before this is ever called. This is what lets the streaming
    stripper recognise a tag that arrives split across fragments -- e.g.
    "<thi" then "nk>" -- without ever looking more than `len(tag) - 1`
    characters back.
    """
    limit = min(len(buf), len(tag) - 1)
    for size in range(limit, 0, -1):
        if buf.endswith(tag[:size]):
            return size
    return 0


class _StreamingReasoningStripper:
    """Incrementally removes <think>...</think> blocks from a token stream.

    A per-chunk regex would leak reasoning whenever a tag is split across
    fragment boundaries -- which happens constantly, since tokens arrive a
    few characters at a time. This keeps a small buffer (never more than
    `len(tag) - 1` characters) so a tag is recognised regardless of where the
    fragment boundaries fall, and emits real content immediately once it is
    provably not part of a tag.

    While inside a think block, content is discarded as it arrives rather
    than accumulated -- so a block that never closes (the model got cut off
    mid-thought) emits nothing, by construction, rather than requiring a
    special case for the unterminated block.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        buf = self._pending + chunk
        self._pending = ""
        out: list[str] = []

        while True:
            if not self._in_think:
                idx = buf.find(_OPEN_TAG)
                if idx == -1:
                    keep = _longest_boundary_overlap(buf, _OPEN_TAG)
                    out.append(buf[: len(buf) - keep] if keep else buf)
                    self._pending = buf[len(buf) - keep :] if keep else ""
                    break
                out.append(buf[:idx])
                buf = buf[idx + len(_OPEN_TAG) :]
                self._in_think = True
            else:
                idx = buf.find(_CLOSE_TAG)
                if idx == -1:
                    # Inside a think block with no close tag in view yet:
                    # discard everything except the tail that could still be
                    # the start of "</think>" arriving in the next fragment.
                    keep = _longest_boundary_overlap(buf, _CLOSE_TAG)
                    self._pending = buf[len(buf) - keep :] if keep else ""
                    break
                buf = buf[idx + len(_CLOSE_TAG) :]
                self._in_think = False

        return "".join(out)

    def finish(self) -> str:
        """Flush at end of stream.

        Buffered text that never completed into an opening tag was never a
        tag at all -- ordinary content that happened to start with "<" -- so
        it is released now. Text buffered *inside* an unterminated think
        block is dropped instead: it is unfinished reasoning, and the whole
        point of this class is that reasoning never reaches the caller,
        finished or not.
        """
        if self._in_think:
            self._pending = ""
            return ""
        leftover, self._pending = self._pending, ""
        return leftover


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks from a complete string.

    Built on the same state machine as the streaming path (fed the whole
    string as one chunk, then flushed) rather than a separate regex, so the
    two can never disagree about an edge case such as an unterminated block.
    """
    stripper = _StreamingReasoningStripper()
    return stripper.feed(text) + stripper.finish()


# --- Prompt assembly ---------------------------------------------------------

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble retrieved chunks into a grounded prompt.

    The instruction to answer only from context, and to admit ignorance when
    the context falls short, is the actual lesson of this half of the talk:
    a RAG answer is only as trustworthy as its grounding, and a model that
    will not say "I don't know" defeats the point of retrieving anything.
    """
    if chunks:
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source") or "unknown source"
            page = metadata.get("page", "")
            label = f"[{i}] {source}, page {page}" if page != "" else f"[{i}] {source}"
            blocks.append(f"{label}\n{chunk['text']}")
        context = "\n\n".join(blocks)
    else:
        # An empty retrieval result is still a valid input -- the prompt
        # must not silently invent context, it must hand the model nothing
        # and let the "say you don't know" instruction do its job.
        context = "(no matching context was retrieved)"

    return (
        "You are answering a question using ONLY the context below, retrieved "
        "from the workshop's own document store. Do not use any outside "
        "knowledge. If the context does not contain enough information to "
        "answer, say plainly that you do not know rather than guessing -- "
        "that is the point being demonstrated, not a fallback to apologise for.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer, citing the [n] labels above where they support your answer:"
    )


# --- Streaming generation -----------------------------------------------------

def stream_answer(
    base_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float = 120.0,
    opener: Opener = _default_opener,
) -> Iterator[str]:
    """Yield incremental text deltas from Ollama's streaming /api/generate.

    A cold deepseek-r1:1.5b call took 15.8s for six tokens, hence the
    generous default timeout -- and hence streaming at all, so the UI has
    something to show instead of a frozen spinner for that whole span.

    Reasoning is stripped token-by-token as it arrives, not from the joined
    result afterward: joining first would mean buffering the entire (slow)
    response before anything could be shown, defeating the reason to stream
    in the first place.

    Any transport failure -- before the stream starts, or mid-stream -- ends
    the generator quietly rather than raising. This runs inside an SSE
    handler; an exception there is a broken event stream, not a caught
    error, and probe() is what tells the UI whether to attempt this at all.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    body = json.dumps({"model": model, "prompt": prompt, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        response = opener(request, timeout)
    except (OSError, urllib.error.HTTPError):
        # Nothing was ever connected -- e.g. Ollama not running at all.
        return

    stripper = _StreamingReasoningStripper()
    try:
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = event.get("response", "")
                if delta:
                    visible = stripper.feed(delta)
                    if visible:
                        yield visible
                if event.get("done"):
                    break
    except (OSError, http.client.HTTPException):
        # The connection died mid-stream (host went away, container network
        # blipped, whatever). Whatever text already yielded stays with the
        # caller; we just stop producing more, cleanly.
        return

    trailing = stripper.finish()
    if trailing:
        yield trailing
