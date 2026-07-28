# Contributing

Thanks for looking. This is a teaching demo for a workshop on RAG indexing, so
the bar for a change is a little different from a normal library: **the code is
projected on a screen and read by attendees.** Clarity and honesty about what
the machine is actually doing matter more than cleverness.

## Getting set up

```bash
docker compose up -d
docker compose run --rm app pytest -q -m "not slow"   # fast: no model load
docker compose run --rm app pytest -q -m slow         # loads the real model
```

**The image bakes source in at build time** — `app/`, `tests/` and `pytest.ini`
are `COPY`'d, not bind-mounted. Run `docker compose build app` after every
source change or you are testing stale code. This wastes more contributor time
on this project than anything else.

## Three rules that are not negotiable

**No PDF is ever committed.** `.gitignore` excludes `*.pdf` absolutely, with no
allow-list, and `.dockerignore` does the same so no document can enter a build
context. Tests generate synthetic PDFs at runtime with `reportlab`. If you need
a sample document, generate one.

**Never fake progress.** Embedding progress is genuinely per-batch. Chunk
*rendering* streams, but splitting is atomic, and the code says so where it
streams. No artificial delays, no simulated percentages, no invented stages.
The workshop's whole thesis is that you should know what your pipeline is
really doing.

**Comments must be true.** Every non-obvious decision carries a comment giving
its reason, and config defaults name the slide they came from. A comment
describing behaviour the code does not have is treated as a defect here, on the
same footing as a logic bug. If you change behaviour, change the comment in the
same commit.

## Tests: verify by mutation

A green suite is not evidence. This project has a documented history of bugs
that shipped *with passing tests*, because the assertions held just as well when
the code was broken:

- a test asserting `page >= 1` passed against an implementation that always
  returned `1`
- `vector_norm` could be replaced with `return 1.0` and the entire suite passed,
  because every fixture vector was already unit length
- a pagination test never checked *which* records came back, so hardcoding
  `offset=0` passed
- `starts == sorted(starts)` held while an offset cursor was frozen, returning
  the same value 78 times

So after writing a test: **break the implementation deliberately, rebuild, and
confirm the test fails.** If it still passes, the test is worthless — delete it
or fix it. Assert specific values, not shapes or bounds. Say in the test's
comment which mutation it is there to catch; several existing tests do, and they
are the ones worth imitating.

Character-offset attribution is the highest-risk area in the codebase. It has
produced more bugs than everything else combined, and offsets become the page
citations shown to a room. `page_for_offset` lives in exactly one place
(`app/pipeline/loader.py`) so the loader and the session cannot drift apart.

## Things that look like mistakes and are not

Documented in `CLAUDE.md`. Briefly: the raw `chromadb` client instead of
`langchain-chroma` (the wrapper computes embeddings internally, which would make
per-batch progress and the vector preview impossible); the slide deck's samples
using LlamaIndex while the app uses LangChain; `unlocked_step()` never returning
3; and `ollama_base_url` being read by nothing.

## Pull requests

- One concern per PR, with the reasoning in the description rather than only the
  diff.
- Full suite green, including `-m slow`.
- New behaviour needs a test that fails without it — and say in the PR which
  mutation you checked it against.
