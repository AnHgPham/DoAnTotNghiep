// ═══════════════════════════════════════════════════════════
//  Few-Shot KWS — Premium Web UI v2
// ═══════════════════════════════════════════════════════════

const API = '';

// ── State ───────────────────────────────
let audioBlobs = { enroll: null, detect: null, long: null };
let mediaRecorder = null;
let audioCtx = null;
let analyser = null;
let recAnimFrame = null;

// ── Navigation ──────────────────────────
document.querySelectorAll('.nav-item[data-tab]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    // Activate both sidebar and mobile instances
    document.querySelectorAll(`.nav-item[data-tab="${btn.dataset.tab}"]`).forEach(b => b.classList.add('active'));
    document.querySelectorAll('.tab-panel').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Toast ────────────────────────────────
function toast(type, msg) {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 300); }, 3000);
}

// ── Drag & Drop ──────────────────────────
function setupDrop(zoneId, fileId, target) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(fileId);
  if (!zone || !input) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      audioBlobs[target] = e.dataTransfer.files[0];
      zone.querySelector('p').textContent = '✓ ' + e.dataTransfer.files[0].name;
      toast('success', 'File loaded');
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) {
      audioBlobs[target] = input.files[0];
      zone.querySelector('p').textContent = '✓ ' + input.files[0].name;
      toast('success', 'File loaded');
    }
  });
}

// ── Microphone ───────────────────────────
async function toggleMicRecord(target) {
  if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    const chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(recAnimFrame);
      audioBlobs[target] = new Blob(chunks, { type: 'audio/wav' });
      const btnId = target === 'enroll' ? 'micBtn' : 'detectMicBtn';
      document.getElementById(btnId).classList.remove('recording');
      const statusId = target === 'enroll' ? 'micStatus' : 'detectMicStatus';
      document.getElementById(statusId).textContent = '✓ Recorded';
      toast('success', 'Audio recorded');
      if (target === 'enroll') autoEnrollMic();
    };
    mediaRecorder.start();
    const btnId = target === 'enroll' ? 'micBtn' : 'detectMicBtn';
    document.getElementById(btnId).classList.add('recording');
    const statusId = target === 'enroll' ? 'micStatus' : 'detectMicStatus';
    document.getElementById(statusId).textContent = '● Recording...';
    drawWaveform(target);
    setTimeout(() => { if (mediaRecorder?.state === 'recording') mediaRecorder.stop(); }, 1500);
  } catch { toast('error', 'Microphone access denied'); }
}

function drawWaveform(target) {
  const canvasId = target === 'enroll' ? 'enrollWaveform' : 'detectWaveform';
  const canvas = document.getElementById(canvasId);
  if (!canvas || !analyser) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const data = new Uint8Array(analyser.frequencyBinCount);
  function draw() {
    recAnimFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    ctx.fillStyle = '#f5f5fa';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#818cf8';
    ctx.beginPath();
    const sliceW = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * h / 2;
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
    }
    ctx.stroke();
  }
  draw();
}

// ── Auto-enroll mic ──────────────────────
async function autoEnrollMic() {
  const keyword = document.getElementById('micKeyword').value.trim();
  if (!keyword) { toast('error', 'Enter keyword name first'); return; }
  if (!audioBlobs.enroll) return;
  const fd = new FormData();
  fd.append('keyword', keyword);
  fd.append('audio', audioBlobs.enroll, 'rec.wav');
  try {
    const r = await fetch(API + '/api/enroll/mic', { method: 'POST', body: fd });
    const d = await r.json();
    if (r.ok) {
      addLog(`Added "${d.word}" (${d.count} samples, thr=${d.threshold})`);
      toast('success', `Enrolled sample for "${d.word}"`);
      refreshEnrolled();
    } else toast('error', d.error || 'Failed');
  } catch { toast('error', 'Network error'); }
}

// ── Enroll GSC ───────────────────────────
async function enrollGSC() {
  const words = document.getElementById('gscWords').value.trim();
  if (!words) { toast('error', 'Enter keywords'); return; }
  toast('info', 'Enrolling from GSC...');
  const fd = new FormData();
  fd.append('words', words);
  fd.append('k', '5');
  try {
    const r = await fetch(API + '/api/enroll/gsc', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.results) {
      d.results.forEach(res => {
        addLog(res.status === 'ok'
          ? `✓ "${res.word}" (${res.samples} samples, thr=${res.threshold})`
          : `✗ "${res.word}": ${res.status}`);
      });
      toast('success', `Enrolled ${d.enrolled} keywords`);
      refreshEnrolled();
    }
  } catch { toast('error', 'Network error'); }
}

// ── Clear ────────────────────────────────
async function clearAll() {
  await fetch(API + '/api/enroll/clear', { method: 'POST' });
  toast('info', 'All keywords cleared');
  addLog('Cleared all keywords');
  refreshEnrolled();
}

// ── Refresh enrolled ─────────────────────
async function refreshEnrolled() {
  try {
    const r = await fetch(API + '/api/enroll/status');
    const d = await r.json();
    const c = document.getElementById('enrolledChips');
    if (d.total === 0) {
      c.innerHTML = '<p class="text-sm text-muted">No keywords enrolled yet</p>';
      return;
    }
    c.innerHTML = Object.entries(d.enrolled).map(([w, info]) =>
      `<div class="chip">${w}<span class="chip-count">${info.count}</span></div>`
    ).join('');
  } catch {}
}

// ── Log ──────────────────────────────────
function addLog(msg) {
  const el = document.getElementById('enrollLog');
  const t = new Date().toLocaleTimeString();
  el.innerHTML = `<div class="log-entry"><span class="log-time">${t}</span>${msg}</div>` + el.innerHTML;
}

// ── Profiles ─────────────────────────────
async function saveProfile() {
  const name = document.getElementById('profileName').value.trim() || 'default';
  const fd = new FormData(); fd.append('name', name);
  const r = await fetch(API + '/api/enroll/save', { method: 'POST', body: fd });
  const d = await r.json();
  r.ok ? toast('success', `Profile "${name}" saved`) : toast('error', d.error);
  loadProfileList();
}

async function loadProfile() {
  const name = document.getElementById('profileName').value.trim() || 'default';
  const fd = new FormData(); fd.append('name', name);
  const r = await fetch(API + '/api/enroll/load', { method: 'POST', body: fd });
  const d = await r.json();
  if (r.ok) { toast('success', `Loaded "${name}" (${d.keywords} kw)`); refreshEnrolled(); }
  else toast('error', d.error);
}

async function loadProfileList() {
  try {
    const r = await fetch(API + '/api/profiles');
    const d = await r.json();
    const el = document.getElementById('profileList');
    el.textContent = d.profiles.length ? 'Saved: ' + d.profiles.join(', ') : 'No profiles yet';
  } catch {}
}

// ── Detect Single ────────────────────────
async function detectSingle() {
  let file = audioBlobs.detect;
  if (!file) { toast('error', 'Upload or record audio first'); return; }
  const out = document.getElementById('detectResult');
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  const fd = new FormData();
  fd.append('audio', file, 'audio.wav');
  fd.append('threshold', document.getElementById('detectThr').value);
  fd.append('use_per_class', document.getElementById('perClassChk').checked);
  try {
    const r = await fetch(API + '/api/detect/single', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error); out.innerHTML = ''; return; }
    const cls = d.detected ? 'detected' : 'rejected';
    out.innerHTML = `<div class="result-card ${cls}">
      <div class="result-keyword">${d.detected ? d.keyword.toUpperCase() : 'UNKNOWN'}</div>
      <div class="result-meta">
        <span><span class="badge ${d.detected ? 'badge-success' : 'badge-danger'}">${d.detected ? 'Detected' : 'Rejected'}</span></span>
        <span class="text-sm">dist: <strong>${d.distance}</strong></span>
        <span class="text-sm">thr: <strong>${d.threshold}</strong></span>
      </div>
    </div>`;
    renderDistBars(d.all_distances, d.threshold);
    if (d.mfcc) renderMFCC(d.mfcc);
  } catch { toast('error', 'Network error'); out.innerHTML = ''; }
}

function renderDistBars(dists, thr) {
  const card = document.getElementById('detectChartCard');
  const c = document.getElementById('distBars');
  card.classList.remove('hidden');
  const entries = Object.entries(dists);
  const mx = Math.max(...entries.map(e => e[1]), thr) * 1.2;
  c.innerHTML = entries.map(([w, d]) => {
    const pct = Math.min((d / mx) * 100, 100);
    const cls = d <= thr ? 'match' : 'no-match';
    return `<div class="dist-row">
      <div class="dist-label">${w}</div>
      <div class="dist-track"><div class="dist-fill ${cls}" style="width:0%" data-w="${pct}%"></div></div>
      <div class="dist-val">${d.toFixed(3)}</div>
    </div>`;
  }).join('');
  // Animate
  requestAnimationFrame(() => {
    c.querySelectorAll('.dist-fill').forEach(f => { f.style.width = f.dataset.w; });
  });
}

function renderMFCC(mfcc) {
  document.getElementById('mfccCard').classList.remove('hidden');
  const canvas = document.getElementById('mfccCanvas');
  const ctx = canvas.getContext('2d');
  const rows = mfcc.length, cols = mfcc[0].length;
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const cw = w / rows, ch = h / cols;
  let mn = Infinity, mx = -Infinity;
  for (const row of mfcc) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const rng = mx - mn || 1;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = (mfcc[i][j] - mn) / rng;
    ctx.fillStyle = `hsl(${(1 - v) * 120}, 80%, ${20 + v * 40}%)`;
    ctx.fillRect(i * cw, (cols - 1 - j) * ch, cw + 1, ch + 1);
  }
}

// ── Detect Long ──────────────────────────
let groundTruthFile = null;

async function detectLong() {
  const fileInput = document.getElementById('longFile');
  let file = fileInput.files.length ? fileInput.files[0] : audioBlobs.long;
  if (!file) { toast('error', 'Upload audio first'); return; }
  const out = document.getElementById('longResult');
  const accDiv = document.getElementById('longAccuracy');
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  accDiv.classList.add('hidden');

  const fd = new FormData();
  fd.append('audio', file, 'audio.wav');
  fd.append('threshold', document.getElementById('longThr').value);
  fd.append('use_per_class', document.getElementById('longPerClass').checked);
  fd.append('seg_method', document.getElementById('longSeg').value);
  fd.append('min_duration_ms', document.getElementById('longMinDur').value);
  try {
    const r = await fetch(API + '/api/detect/long', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error); return; }

    // Parse ground truth if provided
    let gtLabels = null;
    if (groundTruthFile) {
      const gtText = await groundTruthFile.text();
      // Support both: one-per-line AND comma-separated on single line
      let lines = gtText.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'));
      if (lines.length === 1 && lines[0].includes(',')) {
        // Single line with commas: "yes,no,stop,go,up"
        gtLabels = lines[0].split(',').map(w => w.trim().toLowerCase()).filter(Boolean);
      } else {
        gtLabels = lines.map(l => l.toLowerCase());
      }
    }

    // Compute accuracy if GT exists
    let accCorrect = 0, accTotal = 0;
    if (gtLabels && gtLabels.length > 0) {
      accTotal = Math.min(d.results.length, gtLabels.length);
      for (let i = 0; i < accTotal; i++) {
        const predicted = d.results[i].detected ? d.results[i].keyword : 'unknown';
        if (predicted === gtLabels[i]) accCorrect++;
      }
      const accPct = accTotal > 0 ? (accCorrect / accTotal * 100).toFixed(1) : '0.0';
      document.getElementById('longAccTotal').textContent = accTotal;
      document.getElementById('longAccCorrect').textContent = accCorrect;
      document.getElementById('longAccPct').textContent = accPct + '%';
      accDiv.classList.remove('hidden');
    }

    let html = `<div class="card">
      <div class="card-header"><div class="card-title">Results</div>
        <span class="badge badge-neutral">${d.duration}s · ${d.segments} segments</span></div>
      <p class="mb-16 text-sm">Sequence: ${d.sequence.length
        ? d.sequence.map(w => `<span class="badge badge-success">${w}</span>`).join(' ')
        : '<span class="text-muted">No keywords detected</span>'}</p>`;
    if (d.results.length && d.duration > 0) {
      html += '<div class="timeline-track">';
      for (const s of d.results) {
        const l = (s.t0 / d.duration) * 100;
        const w = Math.max(((s.t1 - s.t0) / d.duration) * 100, 1.5);
        html += `<div class="timeline-seg ${s.detected ? 'det' : 'unk'}" style="left:${l}%;width:${w}%"
                      title="${s.t0}s-${s.t1}s: ${s.keyword} (d=${s.distance})">${s.detected ? s.keyword : '?'}</div>`;
      }
      html += '</div>';
    }

    // Table with GT comparison column
    const hasGT = gtLabels && gtLabels.length > 0;
    html += `<table class="data-table mt-16"><thead><tr>
      <th>#</th><th>Time</th><th>Predicted</th>${hasGT ? '<th>Expected</th><th>Match</th>' : ''}<th>Top 3 Candidates</th><th>L2 Dist</th><th>Threshold</th><th>Status</th>
    </tr></thead><tbody>`;
    d.results.forEach((s, i) => {
      const predicted = s.detected ? s.keyword : 'unknown';
      const expected = hasGT && i < gtLabels.length ? gtLabels[i] : null;
      const match = expected !== null ? (predicted === expected) : null;
      const matchBadge = match === true ? '<span class="badge badge-success">✓</span>'
        : match === false ? '<span class="badge badge-danger">✗</span>' : '';
      const top3 = (s.top_3 || []).map((c, idx) =>
        `<span class="text-xs ${idx === 0 ? 'fw-600' : 'text-muted'}">${idx+1}. ${c.word} <span class="font-mono">(${c.dist})</span></span>`
      ).join('<br>');
      html += `<tr>
        <td>${i + 1}</td><td>${s.t0}s – ${s.t1}s</td>
        <td><strong>${predicted}</strong></td>
        ${hasGT ? `<td>${expected || '—'}</td><td>${matchBadge}</td>` : ''}
        <td>${top3}</td>
        <td class="font-mono">${s.distance}</td>
        <td class="font-mono">${s.threshold}</td>
        <td><span class="badge ${s.detected ? 'badge-success' : 'badge-danger'}">${s.detected ? 'Detected' : 'Rejected'}</span></td></tr>`;
    });
    html += '</tbody></table></div>';
    out.innerHTML = html;
  } catch { toast('error', 'Network error'); }
}

// ── Streaming ────────────────────────────
let streamWs = null;
let streamAudioCtx = null;
let streamSource = null;
let streamProcessor = null;
let streamAnalyser = null;
let streamAnimFrame = null;
let streamStartTime = null;
let streamTimerInterval = null;
let streamDetectionCount = 0;
let isStreaming = false;

async function toggleStreaming() {
  if (isStreaming) { stopStreaming(); return; }

  const btn = document.getElementById('streamBtn');
  const status = document.getElementById('streamStatus');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });

    streamAudioCtx = new AudioContext({ sampleRate: 16000 });
    streamSource = streamAudioCtx.createMediaStreamSource(stream);
    streamAnalyser = streamAudioCtx.createAnalyser();
    streamAnalyser.fftSize = 2048;
    streamSource.connect(streamAnalyser);

    // ScriptProcessor to capture PCM chunks
    streamProcessor = streamAudioCtx.createScriptProcessor(4096, 1, 1);
    streamSource.connect(streamProcessor);
    streamProcessor.connect(streamAudioCtx.destination);

    // WebSocket
    const wsUrl = `ws://${location.host}/ws/stream`;
    streamWs = new WebSocket(wsUrl);
    streamWs.binaryType = 'arraybuffer';

    streamWs.onopen = () => {
      isStreaming = true;
      btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> Stop Streaming';
      btn.classList.add('recording');
      status.textContent = 'Listening...';
      status.className = 'badge badge-success';
      streamDetectionCount = 0;
      streamStartTime = Date.now();
      document.getElementById('statDetections').textContent = '0';
      document.getElementById('statLastWord').textContent = '—';
      document.getElementById('streamFeed').innerHTML = '';
      startStreamTimer();
      drawStreamWaveform();
      toast('success', 'Streaming started');
    };

    streamWs.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.detected) {
        streamDetectionCount++;
        document.getElementById('statDetections').textContent = streamDetectionCount;
        document.getElementById('statLastWord').textContent = d.keyword.toUpperCase();
      }

      // Add to feed
      const feed = document.getElementById('streamFeed');
      const t = new Date().toLocaleTimeString();
      const top3 = (d.top_3 || []).map((c, i) => `${c.word}(${c.dist})`).join(', ');
      const cls = d.detected ? 'badge-success' : 'badge-neutral';
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML = `<span class="log-time">${t}</span>
        <span class="badge ${cls}" style="margin-right:6px">${d.detected ? d.keyword.toUpperCase() : 'listening'}</span>
        <span class="text-xs text-muted">L2=${d.distance} thr=${d.threshold}</span>
        <span class="text-xs text-muted" style="margin-left:6px">[${top3}]</span>`;
      feed.insertBefore(entry, feed.firstChild);

      // Flash effect on detection
      if (d.detected) {
        entry.style.background = 'rgba(99,102,241,.12)';
        entry.style.borderLeft = '3px solid var(--accent-500)';
        entry.style.padding = '8px 12px';
        entry.style.borderRadius = 'var(--radius-sm)';
        entry.style.marginBottom = '6px';
      }
    };

    streamWs.onerror = () => toast('error', 'WebSocket error');
    streamWs.onclose = () => { if (isStreaming) stopStreaming(); };

    // Send audio chunks
    streamProcessor.onaudioprocess = (e) => {
      if (!streamWs || streamWs.readyState !== WebSocket.OPEN) return;
      const data = e.inputBuffer.getChannelData(0);
      streamWs.send(data.buffer.slice(0));
    };

  } catch (err) {
    toast('error', 'Microphone access denied');
    console.error(err);
  }
}

function stopStreaming() {
  isStreaming = false;
  const btn = document.getElementById('streamBtn');
  const status = document.getElementById('streamStatus');

  if (streamWs) { streamWs.close(); streamWs = null; }
  if (streamProcessor) { streamProcessor.disconnect(); streamProcessor = null; }
  if (streamSource) { streamSource.disconnect(); streamSource = null; }
  if (streamAnalyser) { streamAnalyser = null; }
  if (streamAudioCtx) {
    streamAudioCtx.close();
    streamAudioCtx = null;
  }
  cancelAnimationFrame(streamAnimFrame);
  clearInterval(streamTimerInterval);

  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg> Start Streaming';
  btn.classList.remove('recording');
  status.textContent = 'Stopped';
  status.className = 'badge badge-danger';
  toast('info', `Streaming stopped — ${streamDetectionCount} detections`);
}

function startStreamTimer() {
  streamTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - streamStartTime) / 1000);
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    document.getElementById('statElapsed').textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
  }, 1000);
}

function drawStreamWaveform() {
  const canvas = document.getElementById('streamWaveform');
  if (!canvas || !streamAnalyser) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const data = new Uint8Array(streamAnalyser.frequencyBinCount);

  function draw() {
    streamAnimFrame = requestAnimationFrame(draw);
    if (!streamAnalyser) return;
    streamAnalyser.getByteTimeDomainData(data);
    ctx.fillStyle = '#f5f5fa';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#818cf8';
    ctx.beginPath();
    const sliceW = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * h / 2;
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
    }
    ctx.stroke();
  }
  draw();
}

// ── Model Info ───────────────────────────
async function loadModelInfo() {
  try {
    const r = await fetch(API + '/api/model/info');
    const d = await r.json();
    document.getElementById('modelCards').innerHTML = [
      ['Architecture', d.architecture],
      ['Parameters', d.parameters?.toLocaleString() || '—'],
      ['Embedding', d.embedding_dim],
      ['Device', d.device],
      ['Input', d.input_shape],
      ['Checkpoint', d.checkpoint],
    ].map(([l, v]) => `<div class="metric-card"><div class="metric-value">${v}</div><div class="metric-label">${l}</div></div>`).join('');
    const ev = document.getElementById('evalResults');
    if (d.evaluations && Object.keys(d.evaluations).length) {
      ev.innerHTML = Object.entries(d.evaluations).map(([name, data]) => {
        if (typeof data === 'object' && !Array.isArray(data) && data.auc !== undefined) {
          return `<div class="mt-12"><p class="text-sm fw-600 mb-8">${name}</p>
            <div class="grid-auto">${[
              ['AUC', data.auc], ['EER', data.eer], ['KW-ACC', data.keyword_acc], ['F1', data.f1]
            ].map(([l, v]) => `<div class="metric-card"><div class="metric-value">${(v || 0).toFixed(3)}</div><div class="metric-label">${l}</div></div>`).join('')}</div></div>`;
        }
        return '';
      }).join('') || '<p class="text-muted text-sm">No evaluation data</p>';
    } else ev.innerHTML = '<p class="text-muted text-sm">No evaluation results found</p>';
  } catch { document.getElementById('evalResults').innerHTML = '<p class="text-muted text-sm">Could not load</p>'; }
}

// ── Presets ───────────────────────────────
async function loadPresets() {
  try {
    const r = await fetch(API + '/api/presets');
    const d = await r.json();
    document.getElementById('presetPills').innerHTML = Object.entries(d.presets).map(([n, w]) =>
      `<span class="preset-pill" onclick="selectPreset(this,'${w}')">${n}</span>`
    ).join('');
  } catch {}
}
function selectPreset(el, words) {
  document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('gscWords').value = words;
}

// ── Batch Evaluation ─────────────────────
let batchTxtFile = null;

async function runBatchEval() {
  if (!batchTxtFile) { toast('error', 'Upload a TXT ground truth file first'); return; }
  const out = document.getElementById('batchResult');
  const summary = document.getElementById('batchSummary');
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  summary.classList.add('hidden');

  const fd = new FormData();
  fd.append('labels_file', batchTxtFile, batchTxtFile.name);
  fd.append('threshold', document.getElementById('batchThr').value);
  fd.append('use_per_class', document.getElementById('batchPerClass').checked);

  try {
    const r = await fetch(API + '/api/detect/batch', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error || 'Failed'); out.innerHTML = ''; return; }

    // Summary
    document.getElementById('batchTotal').textContent = d.total;
    document.getElementById('batchCorrect').textContent = d.correct;
    document.getElementById('batchAccuracy').textContent = d.accuracy + '%';
    summary.classList.remove('hidden');

    // Table
    let html = `<div class="card"><div class="card-header"><div class="card-title">Results</div>
      <span class="badge ${d.accuracy >= 80 ? 'badge-success' : d.accuracy >= 50 ? 'badge-warn' : 'badge-danger'}">${d.accuracy}% accuracy</span></div>
      <table class="data-table"><thead><tr>
        <th>#</th><th>File</th><th>Expected</th><th>Predicted</th><th>Distance</th><th>Status</th>
      </tr></thead><tbody>`;
    d.results.forEach((r, i) => {
      const icon = r.correct ? '✓' : '✗';
      const cls = r.correct ? 'badge-success' : r.status === 'file_not_found' ? 'badge-neutral' : 'badge-danger';
      const label = r.correct ? 'Correct' : r.status === 'file_not_found' ? 'Not Found' : 'Wrong';
      html += `<tr>
        <td>${i + 1}</td>
        <td class="font-mono text-xs">${r.file}</td>
        <td><strong>${r.expected}</strong></td>
        <td>${r.predicted}</td>
        <td class="font-mono">${r.distance || '—'}</td>
        <td><span class="badge ${cls}">${icon} ${label}</span></td>
      </tr>`;
    });
    html += '</tbody></table></div>';
    out.innerHTML = html;
    toast('success', `Evaluated ${d.total} files — ${d.accuracy}% accuracy`);
  } catch { toast('error', 'Network error'); out.innerHTML = ''; }
}

// ── Init ─────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupDrop('detectDrop', 'detectFile', 'detect');
  setupDrop('longDrop', 'longFile', 'long');
  document.getElementById('micFile').addEventListener('change', function () {
    if (this.files.length) {
      audioBlobs.enroll = this.files[0];
      document.getElementById('micStatus').textContent = '✓ ' + this.files[0].name;
      autoEnrollMic();
    }
  });

  // Batch file upload
  const batchInput = document.getElementById('batchFile');
  const batchDrop = document.getElementById('batchDrop');
  if (batchInput && batchDrop) {
    batchInput.addEventListener('change', () => {
      if (batchInput.files.length) {
        batchTxtFile = batchInput.files[0];
        batchDrop.querySelector('p').textContent = '✓ ' + batchTxtFile.name;
        toast('success', 'Ground truth file loaded');
      }
    });
    batchDrop.addEventListener('dragover', e => { e.preventDefault(); batchDrop.classList.add('dragover'); });
    batchDrop.addEventListener('dragleave', () => batchDrop.classList.remove('dragover'));
    batchDrop.addEventListener('drop', e => {
      e.preventDefault(); batchDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        batchTxtFile = e.dataTransfer.files[0];
        batchDrop.querySelector('p').textContent = '✓ ' + batchTxtFile.name;
        toast('success', 'Ground truth file loaded');
      }
    });
  }

  // Ground truth file upload (Long File tab)
  const gtInput = document.getElementById('gtFile');
  const gtDrop = document.getElementById('gtDrop');
  if (gtInput && gtDrop) {
    gtInput.addEventListener('change', () => {
      if (gtInput.files.length) {
        groundTruthFile = gtInput.files[0];
        gtDrop.querySelector('p').textContent = '✓ ' + groundTruthFile.name;
        toast('success', 'Ground truth loaded');
      }
    });
    gtDrop.addEventListener('dragover', e => { e.preventDefault(); gtDrop.classList.add('dragover'); });
    gtDrop.addEventListener('dragleave', () => gtDrop.classList.remove('dragover'));
    gtDrop.addEventListener('drop', e => {
      e.preventDefault(); gtDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        groundTruthFile = e.dataTransfer.files[0];
        gtDrop.querySelector('p').textContent = '✓ ' + groundTruthFile.name;
        toast('success', 'Ground truth loaded');
      }
    });
  }

  loadPresets();
  loadProfileList();
  refreshEnrolled();
  loadModelInfo();
});
