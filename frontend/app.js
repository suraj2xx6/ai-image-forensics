const input = document.querySelector('#file-input');
const zone = document.querySelector('#drop-zone');
const loading = document.querySelector('#loading');
const results = document.querySelector('#results');
const errorBanner = document.querySelector('#error-banner');
let reportData = null;
let previewUrl = null;

function setText(selector, value) {
  document.querySelector(selector).textContent = value == null || value === '' ? 'Not present' : String(value);
}

function row(label, value) {
  const wrapper = document.createElement('div');
  wrapper.className = 'table-row';
  const left = document.createElement('span');
  left.textContent = label;
  const right = document.createElement('span');
  right.textContent = value == null ? 'Not present' : (typeof value === 'object' ? JSON.stringify(value) : String(value));
  wrapper.append(left, right);
  return wrapper;
}

function renderEvidence(items) {
  const list = document.querySelector('#evidence-list');
  list.replaceChildren();
  for (const item of items) {
    const card = document.createElement('div');
    card.className = 'evidence-item';
    const tag = document.createElement('span');
    tag.className = `tag ${String(item.severity || '').toLowerCase()}`;
    tag.textContent = `${item.category || 'SIGNAL'} · ${item.severity || 'LOW'}`;
    const title = document.createElement('h4');
    title.textContent = item.finding || 'Observation';
    const detail = document.createElement('p');
    detail.textContent = item.explanation || item.evidence || '';
    card.append(tag, title, detail);
    list.append(card);
  }
  document.querySelector('#evidence-count').textContent = String(items.length);
  if (!items.length) list.textContent = 'No notable findings.';
}

function renderReport(report) {
  reportData = report;
  const classification = report.classification || {};
  const verdictLabels = {
    INCONCLUSIVE: 'INCONCLUSIVE',
    LIKELY_AI_GENERATED: 'LIKELY AI-GENERATED',
    AI_GENERATED: 'AI-GENERATED',
    LIKELY_AUTHENTIC: 'LIKELY GENUINE',
    AUTHENTIC: 'GENUINE',
    MANIPULATED: 'MANIPULATED'
  };
  const verdict = document.querySelector('#verdict');
  verdict.textContent = verdictLabels[classification.label] || 'INCONCLUSIVE';
  verdict.className = classification.label?.includes('AUTHENTIC') ? 'authentic-result' :
    (classification.label?.includes('AI_GENERATED') ? 'generated-result' : '');
  const messages = {
    INCONCLUSIVE: 'Available evidence is not sufficient for a reliable origin classification.',
    LIKELY_AI_GENERATED: 'The installed model favors AI-generated imagery. Review the supporting signals and limitations.',
    AI_GENERATED: 'The installed model strongly favors AI-generated imagery. This remains a probabilistic finding.',
    LIKELY_AUTHENTIC: 'The classifier leans toward genuine, camera or hand-crafted imagery. This is an estimate, not proof of origin.',
    AUTHENTIC: 'The classifier strongly favors genuine, camera or hand-crafted imagery. This is an estimate, not proof of origin.',
    MANIPULATED: 'Evidence indicates image editing; review the details and source context.'
  };
  setText('#verdict-note', messages[classification.label] || messages.INCONCLUSIVE);
  const calibrated = Boolean(report.model?.calibrated);
  const score = calibrated ? classification.ai_probability : classification.raw_model_score;
  document.querySelector('#ai-score-label').textContent = calibrated ? 'AI PROBABILITY' : (report.model?.available ? 'AI MODEL SCORE' : 'AI PROBABILITY');
  document.querySelector('#ai-score-note').textContent = calibrated
    ? 'Calibrated estimate; not proof of origin.'
    : (report.model?.available ? 'Uncalibrated model score; it is not a validated probability.' : 'Install a classifier to estimate image origin.');
  const scoreText = score == null ? 'Unavailable' :
    `${score < .01 ? (score * 100).toFixed(2) : Math.round(score * 100)}%`;
  setText('#ai-probability', scoreText);
  setText('#confidence', calibrated && classification.confidence != null ? `${Math.round(classification.confidence * 100)}%` : (report.model?.available ? 'Uncalibrated' : 'Unavailable'));
  setText('#model-state', report.model?.available ? `${report.model.name} · ${report.model.version}${report.model.calibrated ? ' · calibrated' : ' · uncalibrated'}` : 'No trained model installed');
  document.querySelector('#meter-fill').style.width = score == null ? '0%' : `${Math.max(0, Math.min(100, score * 100))}%`;
  document.querySelector('#meter-fill').style.background = score > .7 ? 'var(--amber)' : 'var(--mint)';
  setText('#filename-label', report.filename);
  setText('#sha256', report.file?.sha256);
  setText('#analysis-id', `ANALYSIS ${report.analysis_id}`);
  setText('#analysis-time', new Date(report.timestamp).toLocaleString());
  document.querySelector('#file-details').replaceChildren();
  for (const [label, value] of Object.entries({ FORMAT: report.file?.format, SIZE: `${(report.file?.size / 1024).toFixed(1)} KB`, DIMENSIONS: `${report.file?.width} × ${report.file?.height}`, MIME: report.file?.mime_type, FILENAME: report.filename, VERDICT: classification.label })) {
    const cell = document.createElement('div'); cell.className = 'file-detail';
    const key = document.createElement('span'); key.textContent = label;
    const data = document.createElement('strong'); data.textContent = value || '—';
    cell.append(key, data); document.querySelector('#file-details').append(cell);
  }
  renderEvidence(report.evidence || []);

  const metadata = document.querySelector('#metadata-content'); metadata.replaceChildren();
  const metadataGroups = report.metadata || {};
  for (const group of ['file', 'exif', 'xmp', 'iptc', 'icc', 'exiftool']) {
    const values = metadataGroups[group] || {};
    for (const [key, value] of Object.entries(values)) {
      if (key === 'byte_size' || key === 'width' || key === 'height' || key === 'filename' || key === 'mime_type' || key === 'format') continue;
      metadata.append(row(`${group.toUpperCase()} · ${key}`, value));
    }
  }
  if (metadataGroups.software?.length) metadata.append(row('SOFTWARE', metadataGroups.software.join(', ')));
  if (!metadata.childElementCount) metadata.textContent = 'No descriptive metadata was found.';

  const forensic = document.querySelector('#forensics-content'); forensic.replaceChildren();
  for (const [group, values] of Object.entries(report.forensics || {})) {
    if (group === 'feature_vector') continue;
    for (const [key, value] of Object.entries(values || {})) {
      if (key === 'note' || value == null) continue;
      forensic.append(row(`${group.toUpperCase()} · ${key.replaceAll('_', ' ')}`, typeof value === 'number' ? value.toPrecision(4) : value));
    }
  }
  if (!forensic.childElementCount) forensic.textContent = 'No visual measurements are available.';
  results.hidden = false;
}

async function analyze(file) {
  errorBanner.hidden = true;
  results.hidden = true;
  loading.hidden = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  document.querySelector('#preview').src = previewUrl;
  const form = new FormData(); form.append('file', file, file.name);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Analysis failed (${response.status}).`);
    renderReport(body);
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    errorBanner.textContent = error.message || 'Unable to analyze this image.';
    errorBanner.hidden = false;
  } finally { loading.hidden = true; }
}

input.addEventListener('change', () => { if (input.files?.[0]) analyze(input.files[0]); input.value = ''; });
zone.addEventListener('dragover', event => { event.preventDefault(); zone.classList.add('dragging'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
zone.addEventListener('drop', event => { event.preventDefault(); zone.classList.remove('dragging'); const file = event.dataTransfer.files?.[0]; if (file) analyze(file); });
document.querySelector('#download-report').addEventListener('click', () => {
  if (!reportData) return;
  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob); const link = document.createElement('a');
  link.href = url; link.download = `forensics-${reportData.analysis_id}.json`; link.click(); URL.revokeObjectURL(url);
});
document.querySelector('#copy-hash').addEventListener('click', async () => {
  if (reportData?.file?.sha256) await navigator.clipboard.writeText(reportData.file.sha256);
});
fetch('/api/version').then(response => response.json()).then(data => {
  document.querySelector('#app-version').textContent = data.model_available ? `MODEL ${data.model_version}` : 'MODEL NOT INSTALLED';
  document.querySelector('#analysis-mode').textContent = data.sightengine_enabled ? 'SIGHTENGINE API' : 'LOCAL ANALYSIS';
  const privacyNote = document.querySelector('#privacy-note');
  privacyNote.textContent = data.sightengine_enabled
    ? 'Images are sent to Sightengine for analysis; this app does not store original image bytes.'
    : 'Images are analyzed locally; this app stores reports, not original image bytes.';
  if (data.sightengine_setup_incomplete) {
    privacyNote.textContent = 'Sightengine credentials are incomplete; local detection is being used.';
  }
}).catch(() => {});
