/* Progressive-unlock client: the server remains the authority for reachability. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { unlocked: 1, strategy: null, chunkCount: 0, offset: 0, limit: 25 };

function setText(id, value) { $(id).textContent = value; }
function clear(element) { element.replaceChildren(); }
function banner(message = '') { const el = $('banner'); el.textContent = message; el.hidden = !message; }
function addText(parent, tag, text, className = '') { const el = document.createElement(tag); el.textContent = text; if (className) el.className = className; parent.appendChild(el); return el; }

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) { let detail = `${response.status} ${response.statusText}`; try { detail = (await response.json()).detail || detail; } catch { /* non-JSON */ } throw new Error(detail); }
  return response.json();
}

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

function applyTerminal(session) { if (session && session.unlocked_step) applyUnlock(session.unlocked_step); }

/* Every event has a server monotonic id. The shared cursor makes SSE loss and
   polling handoff exact-once, rather than replaying visible chunks. */
function follow(jobId, onEvent, onDone) {
  let cursor = 0; let settled = false; let source = null;
  const consume = (event, id) => {
    const eventId = Number(id || event.id || 0);
    if (eventId && eventId <= cursor) return;
    if (eventId) cursor = eventId;
    onEvent(event);
  };
  const finish = (error = '', session = null) => { if (settled) return; settled = true; if (source) source.close(); onDone(error, session); };
  const poll = async () => {
    while (!settled) {
      let status;
      try { status = await api(`/api/status/${jobId}?after=${cursor}`); } catch (error) { finish(error.message); return; }
      status.events.forEach((event) => consume(event, event.id));
      cursor = Math.max(cursor, Number(status.cursor || 0));
      if (status.status !== 'running') { finish(status.status === 'error' || status.status === 'cancelled' ? status.error : '', status.session); return; }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  };
  try {
    source = new EventSource(`/api/events/${jobId}?after=${cursor}`);
    source.onmessage = (message) => {
      let event;
      try { event = JSON.parse(message.data); } catch { return; }
      consume(event, message.lastEventId || event.id);
      if (event.type === 'done' || event.type === 'error' || event.type === 'cancelled') { source.close(); poll(); }
    };
    source.onerror = () => { if (!settled) { source.close(); poll(); } };
  } catch { poll(); }
}

function renderUpload(upload) {
  const stats = $('upload-stats'); clear(stats); stats.hidden = false;
  [['Pages', upload.page_count], ['Characters', Number(upload.char_count).toLocaleString()], ['Blank pages', upload.pages_without_text]].forEach(([label, value]) => { const item = document.createElement('div'); addText(item, 'b', value); addText(item, 'span', label); stats.appendChild(item); });
  const removed = upload.boilerplate_lines_removed; const invisible = upload.invisible_chars_removed; const note = $('clean-note'); note.hidden = !(removed || invisible);
  if (!note.hidden) note.textContent = `Cleaned before embedding: removed ${removed} boilerplate lines and stripped ${invisible} invisible characters. Raw junk embeds perfectly well and pollutes results.`;
}
async function loadDocument(request) { banner(); try { const body = await request(); renderUpload(body.upload); applyUnlock(body.unlocked_step); } catch (error) { banner(error.message); } }
function uploadFile(file) { const form = new FormData(); form.append('file', file); return loadDocument(() => api('/api/upload', { method: 'POST', body: form })); }
$('pick').onclick = () => $('file').click(); $('file').onchange = (event) => { if (event.target.files[0]) uploadFile(event.target.files[0]); };
const drop = $('drop'); drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('over'); }); drop.addEventListener('dragleave', () => drop.classList.remove('over')); drop.addEventListener('drop', (event) => { event.preventDefault(); drop.classList.remove('over'); if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]); });
if ($('use-local')) $('use-local').onclick = () => loadDocument(() => api('/api/use-local', { method: 'POST' }));

function selectStrategy(button) {
  document.querySelectorAll('.strategy').forEach((item) => item.setAttribute('aria-pressed', String(item === button))); state.strategy = button.dataset.key;
  const usesSize = button.dataset.usesSize === 'true'; const usesOverlap = button.dataset.usesOverlap === 'true';
  $('size').disabled = !usesSize; $('overlap').disabled = !usesOverlap; $('size-label').dataset.disabled = String(!usesSize); $('overlap-label').dataset.disabled = String(!usesOverlap); $('percentile-label').hidden = button.dataset.extra !== 'percentile';
  const notes = { fixed: 'Watch the boundaries: this cuts on character count alone and will slice words in half.', recursive: 'Tries paragraph, then line, then sentence, then word. The deck’s recommended default.', structure: 'Splits on the document’s headings. If it finds none, it falls back to recursive.', semantic: 'Size and overlap do not apply: cut points come from embedding distance between neighbouring sentences.', parent: 'Small children get embedded while larger parents stay beside them for retrieval context.' };
  setText('strategy-note', notes[state.strategy] || '');
}
document.querySelectorAll('.strategy').forEach((button) => { button.onclick = () => selectStrategy(button); });
[['size', 'size-out'], ['overlap', 'overlap-out'], ['percentile', 'percentile-out']].forEach(([input, output]) => { $(input).oninput = () => setText(output, $(input).value); });

function renderChunk(event) { const preview = $('chunk-preview'); if (state.chunkCount === 0) clear(preview); const item = document.createElement('article'); item.className = 'chunk'; const meta = document.createElement('div'); meta.className = 'chunk-meta'; [`#${event.index}`, `${event.char_count} chars`, `page ${event.page}`, event.parent_id ? `parent ${event.parent_id}` : ''].filter(Boolean).forEach((value) => addText(meta, 'span', value)); item.appendChild(meta); addText(item, 'div', event.text, 'chunk-text'); preview.appendChild(item); }
$('run-chunk').onclick = async () => {
  banner(); const button = $('run-chunk'); button.disabled = true; clear($('chunk-preview')); $('chunk-notes').hidden = true; clear($('chunk-notes')); state.chunkCount = 0; const notes = [];
  try { const { job_id: jobId } = await api('/api/chunk', { method:'POST', headers:{ 'Content-Type':'application/json' }, body:JSON.stringify({ strategy:state.strategy, size:Number($('size').value), overlap:Number($('overlap').value), percentile:Number($('percentile').value) }) });
    follow(jobId, (event) => { if (event.type === 'chunk') { state.chunkCount += 1; renderChunk(event); setText('chunk-count', `${state.chunkCount} chunks`); } else if (event.type === 'note') notes.push(event.message); else if (event.type === 'stage') setText('chunk-count', event.message); }, (error, session) => { button.disabled = false; if (error) { banner(error); return; } if (notes.length) { $('chunk-notes').hidden = false; notes.forEach((note) => addText($('chunk-notes'), 'div', note)); } setText('chunk-count', `${state.chunkCount} chunks`); applyTerminal(session); });
  } catch (error) { button.disabled = false; banner(error.message); }
};

$('run-embed').onclick = async () => {
  banner(); const button = $('run-embed'); button.disabled = true; const started = Date.now();
  try { const { job_id: jobId } = await api('/api/embed', { method:'POST' }); follow(jobId, (event) => { if (event.type === 'embedded') { const pct = Math.round((event.done / event.total) * 100); $('embed-bar').style.width = `${pct}%`; setText('embed-status', `${event.done} / ${event.total} chunks embedded · ${((Date.now() - started) / 1000).toFixed(1)}s`); } else if (event.type === 'stage') setText('embed-status', event.message); else if (event.type === 'summary') setText('embed-status', `${event.vectors_written} vectors written to ChromaDB`); }, async (error, session) => { button.disabled = false; if (error) { banner(error); return; } $('embed-bar').style.width = '100%'; applyTerminal(session); state.offset = 0; await loadRecords(); });
  } catch (error) { button.disabled = false; banner(error.message); }
};

function renderRecord(record) { const item = document.createElement('article'); item.className = 'record'; addText(item, 'div', record.id, 'record-id'); addText(item, 'div', record.text, 'chunk-text'); addText(item, 'div', `[${record.vector_preview.join(', ')}, …] ${record.dims} dims · norm ${record.vector_norm}`, 'record-vec'); const details = document.createElement('details'); addText(details, 'summary', 'metadata'); addText(details, 'pre', JSON.stringify(record.metadata, null, 2)); item.appendChild(details); return item; }
async function loadRecords() { try { const page = await api(`/api/collection?offset=${state.offset}&limit=${state.limit}`); setText('record-count', page.total ? `${page.total} records · showing ${page.offset + 1}-${page.offset + page.records.length}` : '0 records'); const records = $('records'); clear(records); if (!page.records.length) addText(records, 'p', 'No records in this collection.', 'empty'); else page.records.forEach((record) => records.appendChild(renderRecord(record))); $('prev-page').disabled = page.offset === 0; $('next-page').disabled = page.offset + page.records.length >= page.total; } catch (error) { banner(error.message); } }
$('prev-page').onclick = () => { state.offset = Math.max(0, state.offset - state.limit); loadRecords(); }; $('next-page').onclick = () => { state.offset += state.limit; loadRecords(); };
$('reset').onclick = async () => { if (!confirm('Clear this session and drop the ChromaDB collection?')) return; banner(); try { const body = await api('/api/reset', { method:'POST', headers:{ 'Content-Type':'application/json' }, body:JSON.stringify({ drop_collection:true }) }); clear($('chunk-preview')); addText($('chunk-preview'), 'p', 'Chunks will appear here as the document is split.', 'empty'); clear($('records')); addText($('records'), 'p', 'Stored vectors will appear here.', 'empty'); $('upload-stats').hidden = true; $('clean-note').hidden = true; $('embed-bar').style.width = '0'; setText('embed-status', ''); setText('chunk-count', ''); setText('record-count', ''); applyUnlock(body.unlocked_step); } catch (error) { banner(error.message); } };

(function init() { const served = window.__STATE__ || {}; selectStrategy(document.querySelector('.strategy[aria-pressed="true"]')); if (served.upload) renderUpload(served.upload); applyUnlock(served.unlocked_step || 1); if ((served.unlocked_step || 1) >= 5) loadRecords(); }());
