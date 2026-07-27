# Task 8 — progressive-unlock frontend

## Design plan and critique

- **Subject / audience / job:** a live two-hour RAG workshop console for a presenter and attendees; make PDF → chunks → vectors legible one stage at a time.
- **Tokens:** deck navy `#0a0f1e`; cyan, amber, pink, violet and green accents; Bricolage Grotesque display, Chivo body and JetBrains Mono data labels, all local fallbacks only.
- **Layout:** a vertical numbered rail encodes the actual pipeline order. The rail and the live code-like previews are the signature; surrounding cards remain quiet.
- **Self-critique:** avoided a generic dashboard/metric-grid treatment. The first browser pass exposed a hidden-control CSS specificity problem (the percentile control remained visible); a regression test and universal `[hidden]` rule corrected it.

## RED / GREEN

- RED: new page tests against Task 7’s placeholder: 3 failed (five sections, local CSS/JS assets, server state); the URL-only assertion passed because the placeholder had no assets.
- RED regression: the hidden-control stylesheet assertion failed before the `[hidden]` rule was added.
- GREEN: `docker compose run --rm app pytest tests/test_api.py -q` → **42 passed**.
- GREEN: `docker compose run --rm app pytest -m 'not slow' -q` → **154 passed, 2 deselected**.

## Browser evidence

- Rebuilt stack with `docker compose up -d --build`.
- ChromeDriver final check: title `RAG Ingestion Pipeline`; 5 sections; steps 2–5 initially `inert` with `aria-disabled=true`; initial percentile computed display `none`.
- Semantic interaction (after making step 2 reachable for the UI-only check): size and overlap disabled; percentile computed display `flex`.
- Console log was empty. Requested resources were same-origin only; DOM resource-origin check found no external URLs.
- Screenshots: `/tmp/task8-desktop.png` (1440×1200, 103534 bytes) and `/tmp/task8-mobile.png` (390×844, 57119 bytes). Both were visually inspected; mobile keeps the rail/cards usable without a horizontal layout.
- Synthetic-PDF upload/chunk flow was covered by the API suite; a full browser upload/embedding run was not repeated because the model-backed operation is already covered in backend tests.

## Files and review

- `app/templates/index.html`: five accessible, server-hydrated stages and local asset links.
- `app/static/app.css`: deck tokens, progressive rail, focus styles, reduced-motion and mobile behavior.
- `app/static/app.js`: textContent-based document rendering; cursor-aware SSE/poll fallback; terminal server session is the unlock authority.
- `tests/test_api.py`: real-page asset/state/offline/hidden-control contracts.
- Removed `app/static/.gitkeep`.

## Concerns

No known frontend blockers. Existing test-suite warnings are third-party Chroma/Pydantic telemetry/deprecation warnings.
