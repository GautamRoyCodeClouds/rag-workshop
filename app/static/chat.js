/* Client for the retrieval-chat page.
 *
 * This is a second, independent script -- not a second copy of app.js
 * loaded onto this page. app.js wires up ids (`run-chunk`, `run-embed`, ...)
 * that only exist on the ingestion page and would throw as soon as it ran
 * here. What *is* reused is the strategy behind app.js's `follow()`: SSE
 * first, with an exact-once cursor so a fallback to polling never replays or
 * drops an event. That is duplicated below rather than imported, because
 * there is no build step and therefore no module system to import through --
 * see CLAUDE.md's "no build step" rule.
 *
 * As in app.js: every value that came from the server or the database goes
 * through textContent, never innerHTML. Chunk text is arbitrary PDF content
 * and chat history is replayed from disk across restarts, so treating either
 * as trusted markup would make this page an injection vector.
 */
'use strict';

const $ = (id) => document.getElementById(id);

const STAGE_ORDER = ['embed_query', 'search', 'rank', 'filter', 'assemble'];

const DEFAULT_INSPECTOR_MESSAGE = 'Ask a question to see the pipeline run, stage by stage.';
// SessionState.to_json() strips each history entry's trace before it ever
// reaches window.__STATE__ (see its docstring) -- a trace carries every pool
// candidate, and repeating that on every page load would be the same waste
// the ingestion page already avoids for chunk bodies. So a message restored
// after a refresh genuinely has no trace to show; this says that plainly
// instead of leaving the panel looking like it forgot to load.
const UNAVAILABLE_INSPECTOR_MESSAGE =
  "This message was asked before the last page refresh. Its retrieval trace "
  + 'was never kept in the page -- only the answer was. Ask it again to see '
  + 'the trace live.';

/* Bumped by beginOperation(), same pattern as app.js: anything asynchronous
 * captures the value current when it started and checks before touching the
 * DOM, so a slow response that lands after a newer question (or a reset)
 * cannot write over what the room is now looking at. */
let operationGeneration = 0;
const activeFollowers = new Set();

/* The one message body currently receiving streamed tokens, if any. Needed
 * because beginOperation() can cancel a follower mid-stream (the user asked
 * another question, or cleared history) without that follower's own onDone
 * ever running -- see beginOperation() below for why this would otherwise
 * leave a blinking cursor on an abandoned answer forever. */
let activeStreamBody = null;

const state = {
  traces: new Map(),   // message_id -> RetrievalTrace, for messages asked this page-load
};

/* ---------------------------------------------------------------------------
 * Small DOM helpers -- identical contract to app.js's
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
 * Job following -- ported from app.js's follow(). See that file's own
 * comment for the full rationale; nothing about the transport logic changes
 * here, only that a chat generation job has no "session" payload to hand
 * back on its terminal event (ingestion jobs return the whole SessionState;
 * this one is just tokens), so callers of follow() below simply ignore that
 * second argument.
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

  const consume = (event, id) => {
    if (!current() || settled) return;
    const eventId = Number(id || event.id || 0);
    if (eventId && eventId <= cursor) return;
    if (eventId) cursor = eventId;
    onEvent(event);
  };

  const finish = (error = '') => {
    if (settled || !current()) return;
    settled = true;
    activeFollowers.delete(cancel);
    if (source) source.close();
    onDone(error);
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
        finish(failed ? status.error : '');
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
    poll();
  }
}

/* Start a new user-initiated operation: invalidate the previous one and stop
 * its followers. Also releases the streaming cursor left on an abandoned
 * answer -- cancel() above stops the follower silently (it never calls
 * onDone), so without this a question superseded mid-stream would leave a
 * "still writing" cursor on screen forever. */
function beginOperation() {
  operationGeneration += 1;
  activeFollowers.forEach((cancel) => cancel());
  activeFollowers.clear();
  if (activeStreamBody) {
    delete activeStreamBody.dataset.streaming;
    activeStreamBody = null;
  }
  return operationGeneration;
}

/* ---------------------------------------------------------------------------
 * The inspector: stages, the model-mismatch warning, and the candidate table
 * ------------------------------------------------------------------------ */

function stageRow(name) {
  return document.querySelector(`.stage[data-stage="${name}"]`);
}

/* Resets the five static stage rows to "not yet run". They exist in the
 * markup from first load (see chat.html) rather than being created here, so
 * the pipeline's shape is visible before anyone has asked a question --
 * and so a page-load test can see the five stage names without a query
 * ever running. */
function resetStages() {
  STAGE_ORDER.forEach((name) => {
    const li = stageRow(name);
    li.dataset.ran = 'false';
    li.querySelector('.stage-ms').textContent = 'not yet run';
    const detail = li.querySelector('.stage-detail');
    clear(detail);
    detail.hidden = true;
    li.querySelector('.stage-toggle').setAttribute('aria-expanded', 'false');
  });
}

/* Every stage in the trace already finished by the time this response
 * lands -- retrieve() runs synchronously and returns a complete trace, not a
 * stream of partial ones. So this renders all five stages at once, in their
 * fixed pipeline order, rather than staggering them in with a delay: a timed
 * reveal here would be a fake stage exactly like the one CLAUDE.md's
 * "never fake progress" rule rules out for the ingestion page -- the work
 * would not actually still be happening while the room watched it "arrive". */
function renderStages(trace) {
  const byName = new Map(trace.stages.map((stage) => [stage.name, stage]));
  STAGE_ORDER.forEach((name) => {
    const li = stageRow(name);
    const stage = byName.get(name);
    const msEl = li.querySelector('.stage-ms');
    const detail = li.querySelector('.stage-detail');
    clear(detail);
    if (!stage) {
      li.dataset.ran = 'false';
      msEl.textContent = 'not yet run';
      return;
    }
    li.dataset.ran = 'true';
    msEl.textContent = `${stage.ms.toFixed(1)} ms`;
    Object.entries(stage.detail).forEach(([key, value]) => {
      addText(detail, 'div', `${key}: ${value}`);
    });
  });
}

/* The panel's one unmissable warning. A non-empty model_mismatch means the
 * collection holds vectors from a different embedding model than the one
 * that just embedded the query -- every similarity number below is then
 * comparing two unrelated coordinate systems, which is the single most
 * common scoring bug in RAG (see the trace-note in chat.html). It goes in
 * the error colour, and it is the first thing rendered in the panel. */
function renderMismatch(trace) {
  const el = $('mismatch-warning');
  clear(el);
  if (!trace.model_mismatch || trace.model_mismatch.length === 0) {
    el.hidden = true;
    return;
  }
  addText(el, 'strong', 'Embedding model mismatch. ');
  addText(
    el,
    'span',
    `This query used ${trace.embed_model}, but the collection also holds vectors `
    + `from ${trace.model_mismatch.join(', ')}. Similarity against those rows is `
    + 'not meaningful -- re-embed with a single model or ignore those candidates.',
  );
  el.hidden = false;
}

const STATUS_LABEL = {
  '': 'selected',
  below_threshold: 'below threshold',
  not_top_k: 'not top-k',
  mmr_redundant: 'mmr redundant',
};

function metaOr(value, fallback = '—') {
  return value === '' || value === undefined || value === null ? fallback : String(value);
}

/* Every candidate ChromaDB returned, winners and losers alike -- the whole
 * point of the panel, per CLAUDE.md's retriever docstring. Rows read like a
 * DevTools Network panel: one line each, a click expands the chunk text and
 * its metadata rather than showing it inline and crowding the table. */
function renderCandidates(trace) {
  const tbody = $('candidate-rows');
  clear(tbody);
  setText('candidate-count', `${trace.candidates.length} shown of a ${trace.pool_size}-candidate pool`);

  trace.candidates.forEach((candidate, index) => {
    const row = document.createElement('tr');
    row.className = 'candidate-row';
    row.dataset.selected = String(candidate.selected);
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.setAttribute('aria-expanded', 'false');

    // Distance and similarity side by side, deliberately -- see the panel's
    // standing note that similarity = 1 - distance, the conversion Chroma
    // itself does not do for you.
    addText(row, 'td', String(index + 1));
    addText(row, 'td', candidate.similarity.toFixed(4));
    addText(row, 'td', candidate.distance.toFixed(4));
    addText(row, 'td', candidate.mmr_score === null || candidate.mmr_score === undefined
      ? '—' : candidate.mmr_score.toFixed(4));
    addText(row, 'td', metaOr(candidate.metadata.page));
    const statusCell = document.createElement('td');
    const badge = addText(statusCell, 'span', STATUS_LABEL[candidate.rejected_reason] ?? candidate.rejected_reason, 'status-badge');
    badge.dataset.reason = candidate.rejected_reason;
    row.appendChild(statusCell);
    tbody.appendChild(row);

    const detailRow = document.createElement('tr');
    detailRow.className = 'candidate-detail-row';
    detailRow.hidden = true;
    const detailCell = document.createElement('td');
    detailCell.colSpan = 6;
    addText(
      detailCell,
      'div',
      `${metaOr(candidate.metadata.source, 'unknown source')} · chunk ${metaOr(candidate.metadata.chunk_index)} · id ${candidate.id}`,
      'muted mono',
    );
    addText(detailCell, 'div', candidate.text, 'candidate-detail');
    detailRow.appendChild(detailCell);
    tbody.appendChild(detailRow);

    const toggle = () => {
      detailRow.hidden = !detailRow.hidden;
      row.setAttribute('aria-expanded', String(!detailRow.hidden));
    };
    row.addEventListener('click', toggle);
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggle();
      }
    });
  });

  $('candidates').hidden = trace.candidates.length === 0;
}

function setInspectorEmpty(message) {
  const empty = $('inspector-empty');
  clear(empty);
  addText(empty, 'p', message);
  empty.hidden = false;
}

function showInspectorForTrace(trace) {
  $('inspector-empty').hidden = true;
  renderMismatch(trace);
  renderStages(trace);
  renderCandidates(trace);
}

function showInspectorPlaceholder(message) {
  renderMismatch({ model_mismatch: [] });
  resetStages();
  $('candidates').hidden = true;
  setInspectorEmpty(message);
}

/* Which message's trace the panel currently shows, so a newly streamed
 * token can tell whether it should keep updating the panel (the user is
 * still looking at *this* answer) or leave it alone (they switched to an
 * older message while this one kept streaming in the background). */
let inspecting = null;

function selectInspect(messageId, button) {
  document.querySelectorAll('.msg-inspect').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  button.setAttribute('aria-pressed', 'true');
  inspecting = messageId;
  const trace = state.traces.get(messageId);
  if (trace) {
    showInspectorForTrace(trace);
  } else {
    showInspectorPlaceholder(UNAVAILABLE_INSPECTOR_MESSAGE);
  }
}

/* ---------------------------------------------------------------------------
 * Rendering one message
 * ------------------------------------------------------------------------ */
function citationLabel(citation) {
  const source = citation.source || 'unknown source';
  return citation.page === '' || citation.page === undefined
    ? source
    : `${source} · p.${citation.page}`;
}

/* One question/answer pair. Handles all four answer.kind values the API can
 * return -- unknown, extractive, generated, and (for a still-streaming
 * generated answer) the empty text a caller fills in afterwards -- from one
 * function, so the four states cannot drift out of sync with each other. */
function addMessage(entry) {
  const article = document.createElement('article');
  article.className = 'msg';

  const question = document.createElement('div');
  question.className = 'msg-q';
  addText(question, 'span', 'You', 'msg-role');
  addText(question, 'p', entry.question);
  article.appendChild(question);

  const answer = document.createElement('div');
  answer.className = 'msg-a';
  answer.dataset.kind = entry.answer.kind;

  const head = document.createElement('div');
  head.className = 'msg-a-head';
  addText(head, 'span', 'Answer', 'msg-role');
  addText(head, 'span', entry.answer.kind, 'msg-kind-badge');
  answer.appendChild(head);

  const body = document.createElement('div');
  body.className = 'msg-a-body';
  body.textContent = entry.answer.text;
  answer.appendChild(body);

  if (entry.answer.citations && entry.answer.citations.length) {
    const citations = document.createElement('div');
    citations.className = 'msg-a-citations';
    entry.answer.citations.forEach((citation) => {
      addText(citations, 'span', citationLabel(citation), 'citation-chip');
    });
    answer.appendChild(citations);
  }

  const meta = document.createElement('div');
  meta.className = 'msg-a-meta muted mono';
  if (entry.answer.kind === 'generated') {
    meta.textContent = `Generated by ${entry.generation.model}.`;
  } else if (entry.answer.kind === 'extractive') {
    // The single most important sentence an extractive answer can carry:
    // no model wrote this, and here (generation.detail) is why generation
    // was not attempted or not available.
    meta.textContent = `No model wrote this -- assembled directly from the retrieved chunks above. ${entry.generation.detail}`;
  }
  if (meta.textContent) answer.appendChild(meta);

  if (entry.answer.kind === 'unknown') {
    // "I don't know" is a correct answer here, not an error -- see
    // CLAUDE.md's states table -- but the *reason* differs and gets a
    // different, actionable next step rather than one generic message.
    const hint = document.createElement('p');
    hint.className = 'msg-a-meta';
    if (/no documents are indexed/.test(entry.answer.text)) {
      hint.appendChild(document.createTextNode('The collection is empty. '));
      const link = document.createElement('a');
      link.href = '/';
      link.textContent = 'Index a document';
      hint.appendChild(link);
      hint.appendChild(document.createTextNode(' to add one.'));
    } else {
      hint.textContent = 'Candidates were retrieved but none cleared the similarity '
        + 'threshold -- open the inspector below to see exactly which ones, and why.';
    }
    answer.appendChild(hint);
  }

  const errorSlot = document.createElement('div');
  errorSlot.className = 'msg-a-error';
  answer.appendChild(errorSlot);

  const inspectBtn = document.createElement('button');
  inspectBtn.type = 'button';
  inspectBtn.className = 'msg-inspect';
  inspectBtn.textContent = 'View retrieval trace';
  inspectBtn.setAttribute('aria-pressed', 'false');
  inspectBtn.onclick = () => selectInspect(entry.message_id, inspectBtn);
  answer.appendChild(inspectBtn);

  article.appendChild(answer);
  $('history').appendChild(article);
  $('empty-state').hidden = true;
  return { body, errorSlot, inspectBtn };
}

/* ---------------------------------------------------------------------------
 * Streaming a generated answer
 *
 * Only this part of the page streams -- retrieval already ran synchronously
 * by the time addMessage() is called, so the inspector is real, finished
 * data from the first render. Tokens are the one thing genuinely still being
 * produced, over the same job-registry/SSE machinery the ingestion page uses
 * for chunk and embed progress.
 * ------------------------------------------------------------------------ */
function followGeneration(bodyEl, errorSlot, jobId, generation) {
  bodyEl.dataset.streaming = 'true';
  activeStreamBody = bodyEl;
  let text = '';
  follow(
    jobId,
    (event) => {
      if (event.type === 'token') {
        text += event.text;
        bodyEl.textContent = text;
      }
    },
    (error) => {
      delete bodyEl.dataset.streaming;
      if (activeStreamBody === bodyEl) activeStreamBody = null;
      if (error) {
        // The retrieval trace already rendered is unaffected -- only the
        // generation step failed, and the panel above it stays exactly as
        // informative as it was before the stream broke.
        addText(
          errorSlot, 'p',
          `Generation stopped: ${error} -- the retrieval trace above is still valid.`,
          'callout warn',
        );
      }
      setInFlight(false);
    },
    generation,
  );
}

/* ---------------------------------------------------------------------------
 * Asking a question
 * ------------------------------------------------------------------------ */
function setInFlight(active) {
  $('send').disabled = active;
  $('message').disabled = active;
  $('ask-form').setAttribute('aria-busy', String(active));
  setText('ask-status', active ? 'Asking…' : '');
}

async function askQuestion(rawMessage) {
  const message = rawMessage.trim();
  if (!message) {
    banner('Type a question first.');
    return;
  }
  banner();
  const generation = beginOperation();
  setInFlight(true);

  let response;
  try {
    response = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        top_k: Number($('top-k').value),
        min_score: Number($('min-score').value),
        algorithm: $('algorithm').value,
        mmr_lambda: Number($('mmr-lambda').value),
      }),
    });
  } catch (error) {
    if (generation === operationGeneration) {
      banner(error.message);
      setInFlight(false);
    }
    return;
  }
  if (generation !== operationGeneration) return;

  $('message').value = '';
  state.traces.set(response.message_id, response.trace);
  const { body, errorSlot, inspectBtn } = addMessage(response);
  selectInspect(response.message_id, inspectBtn);

  if (response.answer.kind === 'generated' && response.generation.job_id) {
    followGeneration(body, errorSlot, response.generation.job_id, generation);
  } else {
    setInFlight(false);
  }
}

$('ask-form').addEventListener('submit', (event) => {
  event.preventDefault();
  askQuestion($('message').value);
});

document.querySelectorAll('.example-q').forEach((button) => {
  // A first-time visitor gets three worked questions instead of an empty
  // box -- clicking asks immediately rather than just filling the textarea,
  // since the point is to see the pipeline run, not to type.
  button.onclick = () => askQuestion(button.textContent);
});

document.querySelectorAll('.stage-toggle').forEach((button) => {
  button.onclick = () => {
    const detail = button.parentElement.querySelector('.stage-detail');
    detail.hidden = !detail.hidden;
    button.setAttribute('aria-expanded', String(!detail.hidden));
  };
});

$('algorithm').onchange = () => {
  // Mirrors selectStrategy()'s treatment of size/overlap in app.js: a
  // control that would silently be ignored gets disabled and dimmed rather
  // than left live, so the UI never implies mmr_lambda matters when the
  // chosen algorithm ignores it.
  const usesMmr = $('algorithm').value === 'mmr';
  $('mmr-lambda').disabled = !usesMmr;
  $('mmr-lambda-label').dataset.disabled = String(!usesMmr);
};

[['top-k', 'top-k-out'], ['min-score', 'min-score-out'], ['mmr-lambda', 'mmr-lambda-out']]
  .forEach(([input, output]) => {
    $(input).oninput = () => setText(output, $(input).value);
  });

$('reset-chat').onclick = async () => {
  if (!confirm('Clear this chat history? The document and its stored vectors are untouched.')) return;
  banner();
  const generation = beginOperation();
  try {
    await api('/api/chat/reset', { method: 'POST' });
  } catch (error) {
    if (generation === operationGeneration) banner(error.message);
    return;
  }
  if (generation !== operationGeneration) return;
  clear($('history'));
  $('empty-state').hidden = false;
  state.traces.clear();
  inspecting = null;
  showInspectorPlaceholder(DEFAULT_INSPECTOR_MESSAGE);
};

/* ---------------------------------------------------------------------------
 * Boot
 *
 * window.__STATE__.chat replays past questions and answers, but never their
 * traces (see UNAVAILABLE_INSPECTOR_MESSAGE above) -- so history renders
 * immediately and looks complete, while each message's own inspect button
 * is what reveals whether this page-load actually has that trace in memory.
 * ------------------------------------------------------------------------ */
(function init() {
  const served = window.__STATE__ || {};
  const history = served.chat || [];
  history.forEach((entry) => addMessage(entry));
  $('empty-state').hidden = history.length > 0;
}());
