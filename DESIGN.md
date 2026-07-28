# Design system

The visual system for the ingestion console. Read this before adding a surface,
so a new page inherits the vocabulary instead of inventing a parallel one.

**Register: product.** Design serves the task. The interface should disappear
into what the user is doing — which here means understanding a RAG pipeline, not
admiring a console. Density is permitted; decoration is not.

**Scene.** A presenter drives this from a laptop in a dimmed room, projected to
an audience reading from a distance, for two hours. That sentence decides two
things: the theme is dark, and 10px text is not acceptable however good its
contrast.

## Two layers, and why

```
Layer 1  primitives   raw palette values, no opinion about use
Layer 2  semantic     what a value MEANS -- surfaces, ink roles, states, scales
Rules                 consume layer 2 only
```

No rule outside `:root` hardcodes a colour. The payoff is that the palette can
be retuned by editing one block instead of auditing 240 declarations, and a
reader can tell *why* a colour is present, not just which one it is.

## Primitives

The deck's palette, verbatim, so the console and the slides match.

| Token | Value |
|---|---|
| `--bg` | `#0a0f1e` |
| `--ink` | `#e9eef9` |
| `--muted` | `#94a3c4` |
| `--cyan` | `#38e0cf` |
| `--amber` | `#ffb547` |
| `--pink` | `#ff5c8a` |
| `--violet` | `#8b7cff` |
| `--green` | `#5ddb8b` |
| `--line` | `rgba(255,255,255,.11)` |

## Semantic tokens

**Surfaces**, by elevation. `sunken` is *darker* than the page: raw data should
read as recessed into a panel, not floating above it.

| Token | Use for |
|---|---|
| `--surface-page` | the page body |
| `--surface-raised` | panels sitting on the page (a step) |
| `--surface-control` | controls sitting on a panel (a button) |
| `--surface-sunken` | machine output (chunk and record previews) |

**Ink.**

| Token | Use for |
|---|---|
| `--ink-body` | interface prose |
| `--ink-muted` | labels, secondary prose |
| `--ink-data` | machine output — dimmer, so it doesn't compete with the UI's own voice |
| `--ink-on-accent` | text on a filled accent surface |

**State.** Every one is a state, never decoration. `--state-active` is the only
one that appears on an idle control.

| Token | Means |
|---|---|
| `--state-active` | selected, current, interactive |
| `--state-info` | neutral informational callout |
| `--state-warn` | attention; something was changed or removed |
| `--state-error` | failure, and destructive-control hover |
| `--state-ok` | success |
| `--state-vector` | vector data specifically |
| `--focus-ring` | focus outline — amber, not the accent, so it stays visible against cyan-accented controls on a projector |

## Type scale

Fixed rem on a ~1.15 ratio. **Not `clamp()`**: this is tool UI read at one
distance, and a heading that shrinks inside a narrow panel looks worse, not
responsive.

| Token | px | Role |
|---|---|---|
| `--text-2xs` | 12 | dense metadata, badges |
| `--text-xs` | 13 | labels, monospace readouts |
| `--text-sm` | 14 | controls, callouts |
| `--text-base` | 15 | prose |
| `--text-lg` | 17 | emphasised figures |
| `--text-xl` | 21 | section headings |
| `--text-3xl` | 44 | page title |

The floor is 12px. It was 10.2px, spread across seven sizes between `.64rem`
and `.78rem` — differences of half a pixel, which is noise rather than
hierarchy. Contrast was never the problem (see below); size was.

## Spacing

`--sp-1` … `--sp-10`: `0.25 0.375 0.5 0.625 0.75 1 1.25 1.5 2 2.5` rem.

Replaced 41 ad hoc values. Rail and node offsets stay literal and are commented
where they appear — they are geometric alignment, not rhythm, and forcing them
onto the scale would misalign the timeline.

## Radius and motion

`--r-sm` 8px · `--r-md` 12px · `--r-lg` 14px · `--r-pill` 999px

`--dur-fast` 120ms · `--dur-base` 180ms · `--dur-slow` 260ms ·
`--ease-out` `cubic-bezier(.22,1,.36,1)`

120–260ms because users are in a task, not watching choreography. The easing
decelerates hard, so a transition reads as settling rather than sliding. No
bounce, and every transition is suppressed under
`prefers-reduced-motion: reduce`.

## Component vocabulary

| Component | Purpose |
|---|---|
| `.step` | one pipeline stage; state via `data-locked` / `data-active` |
| `.callout` | one shape, recoloured through `--c` by `.warn` / `.err` / `.ok` |
| `.preview` | scrollable machine output, capped at 44vh so it never buries later steps |
| `.strategy` | a selectable option card; selection is `aria-pressed`, so the accessible and visible states cannot disagree |
| `.stats` | figure above label, figure larger and accented |
| `.bar` | real progress only |
| `.drop` | a file target |

**State lives in attributes, not classes.** `data-locked`, `data-active`,
`aria-pressed`, `aria-disabled`, `[hidden]`, `:disabled`. One source of truth,
inspectable in devtools, and the accessible state cannot drift from the visible
one.

## Verified, not assumed

Every text/background pair was measured against WCAG. All pass at the 4.5:1
small-text threshold; the weakest is `--state-vector` on `--surface-sunken` at
**6.01:1**. `--ink-body` on `--surface-page` is 16.4:1. So the palette is sound
and did not need changing — the work was structural.

## Known deviations

Recorded rather than hidden, because a system you can't trust the documentation
of is worse than none.

- **Amber carries three unrelated roles**: warning (`.callout.warn`), awaiting
  input (`.drop`), and identifier (`.record-id`). Two are single-use, so they
  reference the primitive directly with a comment rather than minting a token
  used once. If a third surface needs "awaiting input", promote it then.
- **The progress bar's width is set imperatively** from `app.js`
  (`style.width`), so that one value lives outside CSS. Left alone deliberately:
  the behavioural test harness pins it, and the churn would exceed the gain.
- **No z-index scale.** Nothing in this app stacks — there are zero `z-index`
  declarations. A scale was drafted and removed rather than shipped unused. Add
  one when a surface actually needs to stack.
