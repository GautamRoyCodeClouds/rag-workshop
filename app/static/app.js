/* Progressive-unlock client for the ingestion console.
 *
 * Two ideas run through this file:
 *
 * 1. The server is the authority on reachability. Every response carries an
 *    `unlocked_step`, and this script renders that number. It never decides for
 *    itself which step should open next.
 *
 * 2. Every long-running action is generation-guarded. Starting a new upload,
 *    chunk or embed run invalidates the previous one, so a slow response that
 *    lands after the presenter has moved on is discarded instead of writing
 *    stale chunks over fresh ones mid-demo.
 *
 * There is no build step and no framework: attendees can read the file that the
 * browser actually runs.
 */
'use strict';

const $ = (id) => document.getElementById(id);

const state = { unlocked: 1, strategy: null, chunkCount: 0, offset: 0, limit: 25 };

/* Bumped by beginOperation(). Anything asynchronous captures the value current
 * when it started and compares before touching the DOM. */
let operationGeneration = 0;

/* Cancel callbacks for in-flight job followers, so a superseded run stops
 * consuming events rather than racing the new one. */
const activeFollowers = new Set();

/* ---------------------------------------------------------------------------
 * Small DOM helpers
 *
 * Everything user- or server-supplied goes in through textContent, never
 * innerHTML. Chunk text is arbitrary PDF content and record metadata comes
 * back from the database; building these as strings would make the preview
 * panels an injection vector.
 * ------------------------------------------------------------------------ */
function setText(id, value) {
  $(id).textContent = value;
}

function clear(element) {
  element.replaceChildren();
}

function banner(message = '') {
  const el = $('banner');
  el.textContent = message;
  el.hidden = !message;
}

function addText(parent, tag, text, className = '') {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  parent.appendChild(el);
  return el;
}

/* Fetch wrapper that surfaces the server's own error text. FastAPI returns
 * {"detail": "..."} on failure, and those messages are written to be read
 * aloud -- "Needs a text layer", "File is too large" -- so they are worth far
 * more than a bare status code. Falls back to the status line if the body is
 * not JSON. */
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch { /* not JSON; keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ---------------------------------------------------------------------------
 * Step unlocking
 * ------------------------------------------------------------------------ */

/* Render the server's unlocked_step across all five sections.
 *
 * Steps 2 and 3 unlock together -- the server never reports 3, because
 * choosing a strategy and running it happen on the same screen -- hence the
 * `n === 3 && state.unlocked >= 2` clause.
 *
 * Locked sections get `inert` as well as the dimming in CSS. Without it a
 * locked step's buttons stay clickable and keyboard-reachable, so the visual
 * lock would be decorative rather than real. */
function applyUnlock(step) {
  state.unlocked = Number(step) || 1;
  for (let n = 1; n <= 5; n += 1) {
    const section = $(`step-${n}`);
    const reachable = n <= state.unlocked || (n === 3 && state.unlocked >= 2);
    section.dataset.locked = String(!reachable);
    section.dataset.active = String(n === state.unlocked);
    section.toggleAttribute('inert', !reachable);
    section.setAttribute('aria-disabled', String(!reachable));
  }
}

/* Terminal job events carry the whole session, so the unlock level comes from
 * the server rather than being inferred from "the chunk job finished". */
function applyTerminal(session) {
  if (session && session.unlocked_step) applyUnlock(session.unlocked_step);
}

/* Start a new user-initiated operation: invalidate the previous generation,
 * stop its followers, and re-enable the action buttons. Returns the new
 * generation for the caller to capture. */
function beginOperation() {
  operationGeneration += 1;
  activeFollowers.forEach((cancel) => cancel());
  activeFollowers.clear();
  $('run-chunk').disabled = false;
  $('run-embed').disabled = false;
  return operationGeneration;
}

/* ---------------------------------------------------------------------------
 * Following a background job
 *
 * Chunking and embedding return a job id immediately; progress arrives over
 * Server-Sent Events. Two things make that robust enough for a live talk:
 *
 * - A polling fallback. If EventSource fails outright, or the connection drops
 *   (a proxy buffering the stream, a laptop sleeping), the same job is followed
 *   by polling /api/status instead. Progress is never simply lost.
 *
 * - A shared cursor. Every event carries a server-assigned monotonic id, and
 *   both transports advance one cursor, asking only for events after it. That
 *   is what makes the SSE-to-polling handoff exact-once: without it, switching
 *   transports would replay chunks the room has already seen.
 * ------------------------------------------------------------------------ */
function follow(jobId, onEvent, onDone, generation = operationGeneration) {
  let cursor = 0;
  let settled = false;
  let source = null;

  const current = () => generation === operationGeneration;
  const cancel = () => {
    settled = true;
    if (source) source.close();
  };
  activeFollowers.add(cancel);

  /* Deliver one event, unless it belongs to a superseded operation or has
   * already been seen on the other transport. */
  const consume = (event, id) => {
    if (!current() || settled) return;
    const eventId = Number(id || event.id || 0);
    if (eventId && eventId <= cursor) return;
    if (eventId) cursor = eventId;
    onEvent(event);
  };

  /* Terminal, and idempotent: whichever transport arrives first wins, and the
   * loser's later call is a no-op. */
  const finish = (error = '', session = null) => {
    if (settled || !current()) return;
    settled = true;
    activeFollowers.delete(cancel);
    if (source) source.close();
    onDone(error, session);
  };

  const poll = async () => {
    while (!settled && current()) {
      let status;
      try {
        status = await api(`/api/status/${jobId}?after=${cursor}`);
      } catch (error) {
        finish(error.message);
        return;
      }
      status.events.forEach((event) => consume(event, event.id));
      cursor = Math.max(cursor, Number(status.cursor || 0));
      if (status.status !== 'running') {
        const failed = status.status === 'error' || status.status === 'cancelled';
        finish(failed ? status.error : '', status.session);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  };

  try {
    source = new EventSource(`/api/events/${jobId}?after=${cursor}`);
    source.onmessage = (message) => {
      let event;
      try {
        event = JSON.parse(message.data);
      } catch {
        return;
      }
      consume(event, message.lastEventId || event.id);
      /* A terminal event closes the stream, then one poll collects the
       * authoritative final status and session snapshot. */
      if (event.type === 'done' || event.type === 'error' || event.type === 'cancelled') {
        source.close();
        poll();
      }
    };
    source.onerror = () => {
      if (!settled) {
        source.close();
        poll();
      }
    };
  } catch {
    /* No EventSource at all (very old browser, or a blocked stream). */
    poll();
  }
}

/* ---------------------------------------------------------------------------
 * Step 1: load a document
 * ------------------------------------------------------------------------ */

/* The cleaning counts are the teaching point of step 1: the room sees that a
 * PDF is not clean text, and that something had to be removed before
 * embedding. The note only appears when there was actually something to
 * remove. */
function renderUpload(upload) {
  const stats = $('upload-stats');
  clear(stats);
  stats.hidden = false;
  [
    ['Pages', upload.page_count],
    ['Characters', Number(upload.char_count).toLocaleString()],
    ['Blank pages', upload.pages_without_text],
  ].forEach(([label, value]) => {
    const item = document.createElement('div');
    addText(item, 'b', value);
    addText(item, 'span', label);
    stats.appendChild(item);
  });

  const removed = upload.boilerplate_lines_removed;
  const invisible = upload.invisible_chars_removed;
  const note = $('clean-note');
  note.hidden = !(removed || invisible);
  if (!note.hidden) {
    note.textContent =
      `Cleaned before embedding: removed ${removed} boilerplate lines and stripped `
      + `${invisible} invisible characters. Raw junk embeds perfectly well and `
      + 'pollutes results.';
  }
}

function resetEmbeddingView() {
  state.offset = 0;
  clear($('records'));
  addText($('records'), 'p', 'Stored vectors will appear here.', 'empty');
  $('embed-bar').style.width = '0%';
  $('embed-progress').setAttribute('aria-valuenow', '0');
  setText('embed-status', '');
  setText('record-count', '');
}

/* Loading a new document, or re-chunking, invalidates everything downstream.
 * Clearing the views keeps the screen honest: leaving the previous run's
 * chunks visible under a new document would be the most misleading thing this
 * page could do. */
function resetDownstreamView() {
  state.chunkCount = 0;
  clear($('chunk-preview'));
  addText($('chunk-preview'), 'p', 'Chunks will appear here as the document is split.', 'empty');
  clear($('chunk-notes'));
  $('chunk-notes').hidden = true;
  setText('chunk-count', '');
  resetEmbeddingView();
}

/* Shared by the file picker, drag-and-drop, and the presenter's local-document
 * shortcut: they differ only in which request they make. */
async function loadDocument(request) {
  banner();
  const generation = beginOperation();
  try {
    const body = await request();
    if (generation !== operationGeneration) return;
    resetDownstreamView();
    renderUpload(body.upload);
    applyUnlock(body.unlocked_step);
  } catch (error) {
    if (generation === operationGeneration) banner(error.message);
  }
}

function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  return loadDocument(() => api('/api/upload', { method: 'POST', body: form }));
}

$('pick').onclick = () => $('file').click();
$('file').onchange = (event) => {
  if (event.target.files[0]) uploadFile(event.target.files[0]);
};

const drop = $('drop');
drop.addEventListener('dragover', (event) => {
  event.preventDefault();          // without this the browser opens the file
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('over');
  if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]);
});

/* Only rendered when the server has a local document configured. */
if ($('use-local')) {
  $('use-local').onclick = () => loadDocument(() => api('/api/use-local', { method: 'POST' }));
}

/* ---------------------------------------------------------------------------
 * Step 2: choose a chunking strategy
 * ------------------------------------------------------------------------ */

/* Which controls each strategy actually reads, straight from the server's
 * strategy registry via data attributes. Sliders that would be ignored are
 * disabled and dimmed rather than left live: a control that silently does
 * nothing teaches the opposite of the lesson. */
function selectStrategy(button) {
  document.querySelectorAll('.strategy').forEach((item) => {
    item.setAttribute('aria-pressed', String(item === button));
  });
  state.strategy = button.dataset.key;

  const usesSize = button.dataset.usesSize === 'true';
  const usesOverlap = button.dataset.usesOverlap === 'true';
  $('size').disabled = !usesSize;
  $('overlap').disabled = !usesOverlap;
  $('size-label').dataset.disabled = String(!usesSize);
  $('overlap-label').dataset.disabled = String(!usesOverlap);
  $('percentile-label').hidden = button.dataset.extra !== 'percentile';

  const notes = {
    fixed: 'Watch the boundaries: this cuts on character count alone and will slice words in half.',
    recursive: 'Tries paragraph, then line, then sentence, then word. The deck’s recommended default.',
    structure: 'Splits on the document’s headings. If it finds none, it falls back to recursive.',
    semantic: 'Size and overlap do not apply: cut points come from embedding distance between neighbouring sentences.',
    parent: 'Small children get embedded while larger parents stay beside them for retrieval context.',
  };
  setText('strategy-note', notes[state.strategy] || '');
}

document.querySelectorAll('.strategy').forEach((button) => {
  button.onclick = () => selectStrategy(button);
});

[['size', 'size-out'], ['overlap', 'overlap-out'], ['percentile', 'percentile-out']]
  .forEach(([input, output]) => {
    $(input).oninput = () => setText(output, $(input).value);
  });

/* ---------------------------------------------------------------------------
 * Step 3: the chunks
 * ------------------------------------------------------------------------ */

/* One chunk row: index, character count, source page, and parent id when the
 * strategy has parents. The page number comes from the chunk's start offset on
 * the server -- it is the citation a real RAG system would show. */
function renderChunk(event) {
  const preview = $('chunk-preview');
  if (state.chunkCount === 0) clear(preview);   // drop the placeholder
  const item = document.createElement('article');
  item.className = 'chunk';

  const meta = document.createElement('div');
  meta.className = 'chunk-meta';
  [
    `#${event.index}`,
    `${event.char_count} chars`,
    `page ${event.page}`,
    event.parent_id ? `parent ${event.parent_id}` : '',
  ].filter(Boolean).forEach((value) => addText(meta, 'span', value));

  item.appendChild(meta);
  addText(item, 'div', event.text, 'chunk-text');
  preview.appendChild(item);
}

$('run-chunk').onclick = async () => {
  banner();
  const generation = beginOperation();
  const button = $('run-chunk');
  button.disabled = true;
  clear($('chunk-preview'));
  $('chunk-notes').hidden = true;
  clear($('chunk-notes'));
  state.chunkCount = 0;

  /* Notes are collected and shown once at the end -- e.g. structure-aware
   * falling back to recursive because the document had no headings. Shown
   * mid-stream they would flicker past. */
  const notes = [];

  try {
    const { job_id: jobId } = await api('/api/chunk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        strategy: state.strategy,
        size: Number($('size').value),
        overlap: Number($('overlap').value),
        percentile: Number($('percentile').value),
      }),
    });
    if (generation !== operationGeneration) return;

    follow(
      jobId,
      (event) => {
        if (event.type === 'chunk') {
          state.chunkCount += 1;
          renderChunk(event);
          setText('chunk-count', `${state.chunkCount} chunks`);
        } else if (event.type === 'note') {
          notes.push(event.message);
        } else if (event.type === 'stage') {
          setText('chunk-count', event.message);
        }
      },
      (error, session) => {
        button.disabled = false;
        if (error) {
          banner(error);
          return;
        }
        /* New chunks invalidate any vectors already written. */
        resetEmbeddingView();
        if (notes.length) {
          $('chunk-notes').hidden = false;
          notes.forEach((note) => addText($('chunk-notes'), 'div', note));
        }
        setText('chunk-count', `${state.chunkCount} chunks`);
        applyTerminal(session);
      },
      generation,
    );
  } catch (error) {
    if (generation === operationGeneration) {
      button.disabled = false;
      banner(error.message);
    }
  }
};

/* ---------------------------------------------------------------------------
 * Step 4: embed and store
 * ------------------------------------------------------------------------ */
$('run-embed').onclick = async () => {
  banner();
  const generation = beginOperation();
  const button = $('run-embed');
  button.disabled = true;
  const started = Date.now();
  $('embed-bar').style.width = '0%';
  $('embed-progress').setAttribute('aria-valuenow', '0');

  try {
    const { job_id: jobId } = await api('/api/embed', { method: 'POST' });
    if (generation !== operationGeneration) return;

    follow(
      jobId,
      (event) => {
        if (event.type === 'embedded') {
          /* done/total come from completed batches on the server. The bar
           * tracks real work -- it is never advanced on a timer. */
          const pct = Math.round((event.done / event.total) * 100);
          $('embed-bar').style.width = `${pct}%`;
          $('embed-progress').setAttribute('aria-valuenow', String(pct));
          const elapsed = ((Date.now() - started) / 1000).toFixed(1);
          setText('embed-status', `${event.done} / ${event.total} chunks embedded · ${elapsed}s`);
        } else if (event.type === 'stage') {
          setText('embed-status', event.message);
        } else if (event.type === 'summary') {
          setText('embed-status', `${event.vectors_written} vectors written to ChromaDB`);
        }
      },
      async (error, session) => {
        button.disabled = false;
        if (error) {
          banner(error);
          return;
        }
        $('embed-bar').style.width = '100%';
        $('embed-progress').setAttribute('aria-valuenow', '100');
        applyTerminal(session);
        state.offset = 0;
        await loadRecords();       // step 5 fills itself in
      },
      generation,
    );
  } catch (error) {
    if (generation === operationGeneration) {
      button.disabled = false;
      banner(error.message);
    }
  }
};

/* ---------------------------------------------------------------------------
 * Step 5: what ChromaDB is holding
 * ------------------------------------------------------------------------ */

/* One stored record: id, the text, then the vector itself. The truncated
 * vector, its dimensionality and its norm are the payoff of the whole
 * exercise -- the room sees that a record really is numbers, and that the norm
 * reads 1.0 because the vectors are normalised. */
function renderRecord(record) {
  const item = document.createElement('article');
  item.className = 'record';
  addText(item, 'div', record.id, 'record-id');
  addText(item, 'div', record.text, 'chunk-text');
  addText(
    item,
    'div',
    `[${record.vector_preview.join(', ')}, …] ${record.dims} dims · norm ${record.vector_norm}`,
    'record-vec',
  );
  const details = document.createElement('details');
  addText(details, 'summary', 'metadata');
  addText(details, 'pre', JSON.stringify(record.metadata, null, 2));
  item.appendChild(details);
  return item;
}

async function loadRecords(generation = operationGeneration) {
  try {
    const page = await api(`/api/collection?offset=${state.offset}&limit=${state.limit}`);
    if (generation !== operationGeneration) return;
    setText(
      'record-count',
      page.total
        ? `${page.total} records · showing ${page.offset + 1}-${page.offset + page.records.length}`
        : '0 records',
    );
    const records = $('records');
    clear(records);
    if (!page.records.length) {
      addText(records, 'p', 'No records in this collection.', 'empty');
    } else {
      page.records.forEach((record) => records.appendChild(renderRecord(record)));
    }
    $('prev-page').disabled = page.offset === 0;
    $('next-page').disabled = page.offset + page.records.length >= page.total;
  } catch (error) {
    if (generation === operationGeneration) banner(error.message);
  }
}

$('prev-page').onclick = () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadRecords();
};
$('next-page').onclick = () => {
  state.offset += state.limit;
  loadRecords();
};

/* Drops the collection as well as the session, so the next run starts from a
 * genuinely empty database rather than showing the previous document's
 * records. */
$('reset').onclick = async () => {
  if (!confirm('Clear this session and drop the ChromaDB collection?')) return;
  banner();
  const generation = beginOperation();
  try {
    const body = await api('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drop_collection: true }),
    });
    if (generation !== operationGeneration) return;
    resetDownstreamView();
    $('upload-stats').hidden = true;
    $('clean-note').hidden = true;
    applyUnlock(body.unlocked_step);
  } catch (error) {
    if (generation === operationGeneration) banner(error.message);
  }
};

/* ---------------------------------------------------------------------------
 * Boot
 *
 * The server renders the session into window.__STATE__, so a mid-demo refresh
 * comes back to the step the presenter was on with its stats intact, rather
 * than to an empty step 1.
 * ------------------------------------------------------------------------ */
(function init() {
  const served = window.__STATE__ || {};
  selectStrategy(document.querySelector('.strategy[aria-pressed="true"]'));
  if (served.upload) renderUpload(served.upload);
  applyUnlock(served.unlocked_step || 1);
  if ((served.unlocked_step || 1) >= 5) loadRecords();
}());
