import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { apiGet, apiPostForm, formFromObject } from './api';
import { t } from './i18n';
import type { TextKey } from './i18n';
import type {
  ArtifactStatus,
  CalibrationResult,
  CalibrationRow,
  DetectResult,
  EnrollmentStatus,
  Lang,
  LongResult,
  LongSegment,
  ModelProfile,
  ModelProfilesResponse,
  OpenSetCase,
  OpenSetResult,
  PresetResponse,
  TopCandidate
} from './types';

type Tab = 'enroll' | 'single' | 'long' | 'openset' | 'streaming' | 'model' | 'reports';
type Timing = { label: string; start_sec: number; end_sec: number };

const GSC_17_KNOWN = 'yes,stop,happy,bird,dog,tree,marvin,four,learn,wow,sheila,zero,down,left,right,off,three';
const GSC_17_UNKNOWN = 'no,go,up,on,one,two,five,six,seven,eight,nine,bed,cat,house,backward,forward,follow';

const tabs: { id: Tab; labelKey: TextKey }[] = [
  { id: 'enroll', labelKey: 'enrollment' },
  { id: 'single', labelKey: 'singleDetect' },
  { id: 'long', labelKey: 'longAudio' },
  { id: 'openset', labelKey: 'openSet' },
  { id: 'streaming', labelKey: 'streaming' },
  { id: 'model', labelKey: 'modelInfo' },
  { id: 'reports', labelKey: 'reports' }
];

function pct(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function num(value?: number | null, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

function splitWords(value: string): string[] {
  return value.split(/[\s,]+/).map((x) => x.trim().toLowerCase()).filter(Boolean);
}

function profileText(profile: ModelProfile, lang: Lang, key: 'description' | 'notes'): string {
  return (profile[`${key}_${lang}` as keyof ModelProfile] as string)
    || (profile[key] as string)
    || (profile[`${key}_en` as keyof ModelProfile] as string)
    || '';
}

function metricTone(value: number | undefined, goodAt = 0.8): string {
  if (value === undefined || Number.isNaN(value)) return '';
  return value >= goodAt ? 'good' : 'warn';
}

function overlap(a0: number, a1: number, b0: number, b1: number): number {
  return Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
}

function parseLabelText(text: string): string[] {
  return text
    .replace(/\r/g, '\n')
    .split(/[\n,]+/)
    .map((line) => line.trim().split(/\s+/)[0])
    .filter(Boolean);
}

async function readFileText(file: File | null): Promise<string> {
  if (!file) return '';
  return file.text();
}

function parseTimingJson(text: string): Timing[] {
  if (!text.trim()) return [];
  const raw = JSON.parse(text);
  const rows = Array.isArray(raw)
    ? raw
    : (raw.timings || raw.labels || raw.words || raw.segments || raw.items || raw.events || []);
  const sampleRate = Number(raw.sample_rate || raw.sampleRate || raw.sr || 16000);
  return rows.map((item: Record<string, unknown>) => ({
    label: String(item.label ?? item.word ?? item.keyword ?? '').toLowerCase(),
    start_sec: Number(item.start_sec ?? item.start ?? item.t0 ?? (
      item.start_sample !== undefined ? Number(item.start_sample) / sampleRate : 0
    )),
    end_sec: Number(item.end_sec ?? item.end ?? item.t1 ?? (
      item.end_sample !== undefined ? Number(item.end_sample) / sampleRate : 0
    ))
  })).filter((item: Timing) => item.label && item.end_sec > item.start_sec);
}

function Card({ title, children, actions }: { title?: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="panel">
      {(title || actions) && (
        <div className="panel-head">
          {title && <h2>{title}</h2>}
          {actions && <div className="panel-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

function MetricCard({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <div className={`metric-card ${tone || ''}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function Checkbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="check-row">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function PolicyToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className={`policy-toggle ${checked ? 'on' : 'off'}`}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>
        <strong>{label}</strong>
        <small>{checked ? 'ON' : 'OFF'}</small>
      </span>
    </label>
  );
}

function TopCandidates({ items }: { items?: TopCandidate[] }) {
  if (!items?.length) return <p className="muted">-</p>;
  return (
    <ol className="top-list">
      {items.slice(0, 3).map((item, index) => (
        <li key={`${item.word}-${index}`}>
          <strong>{item.word}</strong> <span>({num(item.dist, 4)})</span>
        </li>
      ))}
    </ol>
  );
}

function PolicyCards({ settings, lang }: { settings?: DetectResult['settings']; lang: Lang }) {
  if (!settings) return null;
  return (
    <div className="metric-grid tight">
      <MetricCard label={t(lang, 'threshold')} value={num(settings.threshold, 2)} />
      <MetricCard label={t(lang, 'perClass')} value={settings.use_per_class ? 'ON' : 'OFF'} tone={settings.use_per_class ? 'good' : 'warn'} />
      <MetricCard label={t(lang, 'closeGuard')} value={settings.close_word_guard ? 'ON' : 'OFF'} tone={settings.close_word_guard ? 'good' : 'warn'} />
      <MetricCard label={t(lang, 'acceptMargin')} value={num(settings.accept_margin, 4)} />
      <MetricCard label="Engine" value={settings.engine} />
      <MetricCard label="Model" value={settings.model_label || settings.model_profile || '-'} />
    </div>
  );
}

function ModelCard({ profile, active, lang, onSelect }: {
  profile: ModelProfile;
  active: boolean;
  lang: Lang;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`model-card ${active ? 'active' : ''} ${profile.exists ? '' : 'missing'}`}
      aria-current={active ? 'true' : undefined}
      disabled={!profile.exists}
      onClick={onSelect}
    >
      <div className="model-card-top">
        <span className={`badge ${active ? 'success' : profile.exists ? 'neutral' : 'danger'}`}>
          {active ? t(lang, 'active') : profile.exists ? t(lang, 'ready') : t(lang, 'missing')}
        </span>
        <span className="badge neutral">{profile.checkpoint_name || 'checkpoint'}</span>
      </div>
      <h3>{profile.short_label || profile.label}</h3>
      <p>{profileText(profile, lang, 'notes') || profileText(profile, lang, 'description')}</p>
      <div className="mini-metrics">
        {(profile.metrics || []).slice(0, 3).map((metric) => (
          <span key={metric.label}><strong>{metric.value}</strong>{metric.label}</span>
        ))}
      </div>
    </button>
  );
}

function Timeline({ title, timings, duration, color }: {
  title: string;
  timings: { label: string; start_sec: number; end_sec: number; ok?: boolean }[];
  duration: number;
  color: 'expected' | 'detected';
}) {
  const minWidth = Math.max(760, Math.round(duration * 34), timings.length * 64);
  return (
    <div className="timeline-block">
      <div className="timeline-title-row">
        <div className="timeline-title">{title}</div>
        <span className="badge neutral">{timings.length}</span>
      </div>
      <div className="timeline-scroll" tabIndex={0}>
        <div className="timeline-track" style={{ minWidth }}>
          <div className="timeline-axis" aria-hidden="true" />
          {timings.map((item, index) => {
            const left = Math.max(0, Math.min(100, (item.start_sec / Math.max(duration, 0.001)) * 100));
            const width = Math.max(1.8, ((item.end_sec - item.start_sec) / Math.max(duration, 0.001)) * 100);
            return (
              <div
                key={`${item.label}-${index}`}
                className={`timeline-seg ${color} ${item.ok === false ? 'bad' : ''}`}
                title={`${item.label}: ${num(item.start_sec, 2)}-${num(item.end_sec, 2)}s`}
                style={{ left: `${left}%`, width: `${width}%` }}
              >
                <span>{item.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState<Lang>('vi');
  const [activeTab, setActiveTab] = useState<Tab>('enroll');
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState('');
  const [profileToSwitch, setProfileToSwitch] = useState<ModelProfile | null>(null);
  const [enrollment, setEnrollment] = useState<EnrollmentStatus | null>(null);
  const [presets, setPresets] = useState<PresetResponse | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactStatus | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const [enrollWords, setEnrollWords] = useState(GSC_17_KNOWN);
  const [enrollK, setEnrollK] = useState(5);
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singleThreshold, setSingleThreshold] = useState(0.3);
  const [singlePerClass, setSinglePerClass] = useState(true);
  const [singleGuard, setSingleGuard] = useState(true);
  const [singleResult, setSingleResult] = useState<DetectResult | null>(null);

  const [longFile, setLongFile] = useState<File | null>(null);
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [timingFile, setTimingFile] = useState<File | null>(null);
  const [longThreshold, setLongThreshold] = useState(0.3);
  const [longPerClass, setLongPerClass] = useState(true);
  const [longGuard, setLongGuard] = useState(true);
  const [longMinDur, setLongMinDur] = useState(120);
  const [longSeg, setLongSeg] = useState('Energy');
  const [longLabels, setLongLabels] = useState<string[]>([]);
  const [longTimings, setLongTimings] = useState<Timing[]>([]);
  const [longResult, setLongResult] = useState<LongResult | null>(null);

  const [openKnown, setOpenKnown] = useState(GSC_17_KNOWN);
  const [openUnknown, setOpenUnknown] = useState(GSC_17_UNKNOWN);
  const [openK, setOpenK] = useState(5);
  const [openThreshold, setOpenThreshold] = useState(0.3);
  const [openPerClass, setOpenPerClass] = useState(false);
  const [openGuard, setOpenGuard] = useState(true);
  const [openMargin, setOpenMargin] = useState(0.05);
  const [openSeed, setOpenSeed] = useState(1234);
  const [openResult, setOpenResult] = useState<OpenSetResult | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [streamEvents, setStreamEvents] = useState<DetectResult[]>([]);
  const streamRefs = useRef<{ ws?: WebSocket; ctx?: AudioContext; source?: MediaStreamAudioSourceNode; processor?: ScriptProcessorNode; media?: MediaStream }>({});

  const activeModel = useMemo(
    () => profiles.find((profile) => profile.id === activeProfile) || null,
    [profiles, activeProfile]
  );

  async function refreshAll() {
    setError('');
    const [profilePayload, enrollPayload, presetPayload, artifactPayload] = await Promise.all([
      apiGet<ModelProfilesResponse>('/api/model/profiles'),
      apiGet<EnrollmentStatus>('/api/enroll/status'),
      apiGet<PresetResponse>('/api/presets'),
      apiGet<ArtifactStatus>('/api/artifacts/status')
    ]);
    setProfiles(profilePayload.profiles || []);
    setActiveProfile(profilePayload.active);
    setEnrollment(enrollPayload);
    setPresets(presetPayload);
    setArtifacts(artifactPayload);
  }

  useEffect(() => {
    refreshAll().catch((err: Error) => setError(err.message));
  }, []);

  async function runTask<T>(name: string, fn: () => Promise<T>): Promise<T | null> {
    setBusy(name);
    setError('');
    try {
      return await fn();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setBusy('');
    }
  }

  async function switchModel(policy: 'rebuild' | 'clear') {
    if (!profileToSwitch) return;
    const result = await runTask('switchModel', () => apiPostForm('/api/model/select', formFromObject({
      profile_id: profileToSwitch.id,
      enrollment_policy: policy
    })));
    if (result) {
      setProfileToSwitch(null);
      await refreshAll();
    }
  }

  async function enrollGsc() {
    await runTask('enroll', async () => {
      await apiPostForm('/api/enroll/gsc', formFromObject({ words: enrollWords, k: enrollK }));
      await refreshAll();
    });
  }

  async function clearEnrollment() {
    await runTask('clear', async () => {
      await apiPostForm('/api/enroll/clear', new FormData());
      await refreshAll();
    });
  }

  async function detectSingle() {
    if (!singleFile) {
      setError(t(lang, 'noFile'));
      return;
    }
    const form = formFromObject({
      audio: singleFile,
      threshold: singleThreshold,
      use_per_class: singlePerClass,
      use_close_word_guard: singleGuard
    });
    const data = await runTask('single', () => apiPostForm<DetectResult>('/api/detect/single', form));
    if (data) setSingleResult(data);
  }

  async function detectLong() {
    if (!longFile) {
      setError(t(lang, 'noFile'));
      return;
    }
    const [labelText, timingText] = await Promise.all([readFileText(labelFile), readFileText(timingFile)]);
    const labels = parseLabelText(labelText);
    const timings = timingText ? parseTimingJson(timingText) : [];
    setLongLabels(labels);
    setLongTimings(timings);
    const form = formFromObject({
      audio: longFile,
      threshold: longThreshold,
      use_per_class: longPerClass,
      use_close_word_guard: longGuard,
      seg_method: longSeg,
      min_duration_ms: longMinDur
    });
    const data = await runTask('long', () => apiPostForm<LongResult>('/api/detect/long', form));
    if (data) setLongResult(data);
  }

  function openSetForm(): FormData {
    return formFromObject({
      preset: 'gsc_17_17',
      known_words: openKnown,
      unknown_words: openUnknown,
      samples_per_word: openK,
      threshold: openThreshold,
      use_per_class: openPerClass,
      use_close_word_guard: openGuard,
      accept_margin: openMargin,
      seed: openSeed
    });
  }

  async function runOpenSet() {
    const data = await runTask('openset', () => apiPostForm<OpenSetResult>('/api/open-set/test', openSetForm()));
    if (data) setOpenResult(data);
  }

  async function runCalibration() {
    const form = openSetForm();
    form.append('threshold_min', '0.10');
    form.append('threshold_max', '1.20');
    form.append('threshold_step', '0.05');
    form.append('accept_margin_values', '0.00,0.02,0.05,0.08,0.10');
    form.append('use_per_class_options', 'true,false');
    const data = await runTask('calibration', () => apiPostForm<CalibrationResult>('/api/open-set/calibrate', form));
    if (data) setCalibration(data);
  }

  function applyCalibration(row: CalibrationRow) {
    setOpenThreshold(row.threshold);
    setOpenPerClass(row.use_per_class);
    setOpenGuard(row.close_word_guard);
    setOpenMargin(row.accept_margin);
  }

  async function exportReport() {
    const data = await runTask('export', () => apiPostForm<{ markdown: string }>('/api/export/session-report', formFromObject({
      title: 'Few-Shot KWS Demo Session'
    })));
    if (data?.markdown) await navigator.clipboard.writeText(data.markdown);
  }

  async function toggleStreaming() {
    if (streaming) {
      streamRefs.current.processor?.disconnect();
      streamRefs.current.source?.disconnect();
      streamRefs.current.ctx?.close();
      streamRefs.current.ws?.close();
      streamRefs.current.media?.getTracks().forEach((track) => track.stop());
      streamRefs.current = {};
      setStreaming(false);
      return;
    }
    await runTask('streaming', async () => {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const source = ctx.createMediaStreamSource(media);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/stream`);
      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data) as DetectResult;
        setStreamEvents((items) => [payload, ...items].slice(0, 30));
      };
      source.connect(processor);
      processor.connect(ctx.destination);
      processor.onaudioprocess = (event) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        ws.send(copy.buffer);
      };
      streamRefs.current = { ws, ctx, source, processor, media };
      setStreaming(true);
    });
  }

  const timingMatches = useMemo(() => {
    if (!longResult || !longTimings.length) return { matched: 0, rows: [] as { timing: Timing; segment?: LongSegment; ok: boolean; reason: string }[] };
    const rows = longTimings.map((timing) => {
      let best: LongSegment | undefined;
      let bestOverlap = 0;
      for (const segment of longResult.results) {
        const amount = overlap(timing.start_sec, timing.end_sec, segment.t0, segment.t1);
        if (amount > bestOverlap) {
          best = segment;
          bestOverlap = amount;
        }
      }
      if (!best || bestOverlap <= 0) return { timing, ok: false, reason: t(lang, 'noTimingOverlap') };
      const predicted = best.detected ? best.keyword : 'unknown';
      const ok = predicted === timing.label;
      const reason = detectionReason(best, predicted, timing, ok, lang);
      return { timing, segment: best, ok, reason };
    });
    return { matched: rows.filter((row) => row.ok).length, rows };
  }, [lang, longResult, longTimings]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div>
          <h1>{t(lang, 'appTitle')}</h1>
          <p>{t(lang, 'appSubtitle')}</p>
        </div>
        <nav className="nav-list" aria-label="Demo workflow">
          {tabs.map((tab) => (
            <button key={tab.id} className={activeTab === tab.id ? 'active' : ''} type="button" onClick={() => setActiveTab(tab.id)}>
              {t(lang, tab.labelKey)}
            </button>
          ))}
        </nav>
        <div className="lang-row" role="group" aria-label="Language">
          <button type="button" className={lang === 'vi' ? 'active' : ''} onClick={() => setLang('vi')}>VI</button>
          <button type="button" className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
        </div>
      </aside>

      <main className="main">
        <Card
          title={t(lang, 'activeModel')}
          actions={<button className="btn ghost" type="button" onClick={() => refreshAll()}>{t(lang, 'reload')}</button>}
        >
          <div className="model-grid">
            {profiles.map((profile) => (
              <ModelCard
                key={profile.id}
                profile={profile}
                active={profile.id === activeProfile}
                lang={lang}
                onSelect={() => setProfileToSwitch(profile)}
              />
            ))}
          </div>
          {activeModel && <p className="note">{profileText(activeModel, lang, 'description')}</p>}
        </Card>

        {error && <div className="alert danger" role="alert"><strong>{t(lang, 'error')}:</strong> {error}</div>}

        {activeTab === 'enroll' && (
          <Card title={t(lang, 'enrollment')}>
            <div className="form-grid">
              <Field label={t(lang, 'customWords')}>
                <textarea value={enrollWords} onChange={(event) => setEnrollWords(event.target.value)} rows={4} />
              </Field>
              <Field label={t(lang, 'samplesPerWord')}>
                <input type="number" min={1} max={20} value={enrollK} onChange={(event) => setEnrollK(Number(event.target.value))} />
              </Field>
            </div>
            <div className="button-row">
              <button className="btn primary" type="button" onClick={enrollGsc} disabled={busy === 'enroll'}>{busy === 'enroll' ? t(lang, 'running') : t(lang, 'enroll')}</button>
              <button className="btn ghost" type="button" onClick={() => setEnrollWords(GSC_17_KNOWN)}>GSC 17 known</button>
              <button className="btn danger" type="button" onClick={clearEnrollment}>{t(lang, 'clearAll')}</button>
            </div>
            <EnrollmentSummary enrollment={enrollment} lang={lang} />
            {presets && (
              <div className="preset-grid">
                {Object.entries(presets.presets || {}).map(([name, words]) => (
                  <button key={name} type="button" onClick={() => setEnrollWords(words)}>{name}</button>
                ))}
              </div>
            )}
          </Card>
        )}

        {activeTab === 'single' && (
          <Card title={t(lang, 'singleDetect')}>
            <div className="form-grid">
              <Field label={t(lang, 'uploadAudio')}><input type="file" accept="audio/*" onChange={(event) => setSingleFile(event.target.files?.[0] || null)} /></Field>
              <Field label={t(lang, 'threshold')}><input type="number" step={0.01} value={singleThreshold} onChange={(event) => setSingleThreshold(Number(event.target.value))} /></Field>
            </div>
            <div className="check-grid">
              <Checkbox label={t(lang, 'perClass')} checked={singlePerClass} onChange={setSinglePerClass} />
              <Checkbox label={t(lang, 'closeGuard')} checked={singleGuard} onChange={setSingleGuard} />
            </div>
            <button className="btn primary" type="button" onClick={detectSingle} disabled={busy === 'single'}>{busy === 'single' ? t(lang, 'running') : t(lang, 'detect')}</button>
            {singleResult && <DetectionResult result={singleResult} lang={lang} />}
          </Card>
        )}

        {activeTab === 'long' && (
          <Card title={t(lang, 'longAudio')}>
            <div className="long-form">
              <div className="long-form-row files">
                <Field label={t(lang, 'uploadAudio')}><input type="file" accept="audio/*" onChange={(event) => setLongFile(event.target.files?.[0] || null)} /></Field>
                <Field label={t(lang, 'labels')}><input type="file" accept=".txt,.csv" onChange={(event) => setLabelFile(event.target.files?.[0] || null)} /></Field>
                <Field label={t(lang, 'timings')}><input type="file" accept=".json" onChange={(event) => setTimingFile(event.target.files?.[0] || null)} /></Field>
              </div>
              <div className="long-form-row settings">
                <Field label={t(lang, 'threshold')}><input type="number" step={0.01} value={longThreshold} onChange={(event) => setLongThreshold(Number(event.target.value))} /></Field>
                <Field label={t(lang, 'segmentation')}>
                  <select value={longSeg} onChange={(event) => setLongSeg(event.target.value)}>
                    <option>Energy</option>
                    <option>Silero VAD</option>
                  </select>
                </Field>
                <Field label={t(lang, 'minDuration')}><input type="number" min={80} max={5000} value={longMinDur} onChange={(event) => setLongMinDur(Number(event.target.value))} /></Field>
              </div>
              <div className="long-action-row">
                <PolicyToggle label={t(lang, 'perClass')} checked={longPerClass} onChange={setLongPerClass} />
                <PolicyToggle label={t(lang, 'closeGuard')} checked={longGuard} onChange={setLongGuard} />
                <button className="btn primary" type="button" onClick={detectLong} disabled={busy === 'long'}>
                  {busy === 'long' ? t(lang, 'running') : t(lang, 'runLongDetect')}
                </button>
              </div>
            </div>
            {longResult && (
              <LongResultView result={longResult} labels={longLabels} timings={longTimings} matches={timingMatches} lang={lang} />
            )}
          </Card>
        )}

        {activeTab === 'openset' && (
          <Card title={t(lang, 'openSet')}>
            <div className="alert info"><strong>{t(lang, 'recommended')}:</strong> {t(lang, 'guardRecommendation')}</div>
            <div className="form-grid">
              <Field label={t(lang, 'knownWords')}><textarea value={openKnown} rows={3} onChange={(event) => setOpenKnown(event.target.value)} /></Field>
              <Field label={t(lang, 'unknownWords')}><textarea value={openUnknown} rows={3} onChange={(event) => setOpenUnknown(event.target.value)} /></Field>
              <Field label={t(lang, 'samplesPerWord')}><input type="number" min={1} max={10} value={openK} onChange={(event) => setOpenK(Number(event.target.value))} /></Field>
              <Field label={t(lang, 'threshold')}><input type="number" step={0.01} value={openThreshold} onChange={(event) => setOpenThreshold(Number(event.target.value))} /></Field>
              <Field label={t(lang, 'acceptMargin')}><input type="number" min={0} max={0.1} step={0.01} value={openMargin} onChange={(event) => setOpenMargin(Number(event.target.value))} /></Field>
              <Field label="Seed"><input type="number" value={openSeed} onChange={(event) => setOpenSeed(Number(event.target.value))} /></Field>
            </div>
            <div className="check-grid">
              <Checkbox label={t(lang, 'perClass')} checked={openPerClass} onChange={setOpenPerClass} />
              <Checkbox label={t(lang, 'closeGuard')} checked={openGuard} onChange={setOpenGuard} />
            </div>
            <div className="button-row">
              <button className="btn primary" type="button" onClick={runOpenSet} disabled={busy === 'openset'}>{busy === 'openset' ? t(lang, 'running') : t(lang, 'runOpenSet')}</button>
              <button className="btn ghost" type="button" onClick={runCalibration} disabled={busy === 'calibration'}>{busy === 'calibration' ? t(lang, 'running') : t(lang, 'runCalibration')}</button>
            </div>
            {openResult && <OpenSetView result={openResult} lang={lang} />}
            {calibration && <CalibrationView data={calibration} lang={lang} onApply={applyCalibration} />}
          </Card>
        )}

        {activeTab === 'streaming' && (
          <Card title={t(lang, 'streaming')}>
            <div className="metric-grid">
              <MetricCard label="State" value={streaming ? 'listening' : 'idle'} tone={streaming ? 'good' : ''} />
              <MetricCard label="Detections" value={streamEvents.length} />
              <MetricCard label="Last keyword" value={streamEvents[0]?.keyword || '-'} />
            </div>
            <button className={`btn ${streaming ? 'danger' : 'primary'}`} type="button" onClick={toggleStreaming}>
              {streaming ? t(lang, 'stopStreaming') : t(lang, 'startStreaming')}
            </button>
            <div className="case-list">
              {streamEvents.map((event, index) => (
                <DetectionResult key={index} result={event} lang={lang} compact />
              ))}
            </div>
          </Card>
        )}

        {activeTab === 'model' && (
          <Card title={t(lang, 'modelInfo')}>
            <div className="metric-grid">
              {(artifacts?.records || []).map((record) => (
                <MetricCard key={record.id} label={record.status} value={record.exists ? record.label : `${record.label} missing`} tone={record.exists ? 'good' : 'warn'} />
              ))}
            </div>
          </Card>
        )}

        {activeTab === 'reports' && (
          <Card title={t(lang, 'reports')}>
            <button className="btn primary" type="button" onClick={exportReport} disabled={busy === 'export'}>{busy === 'export' ? t(lang, 'running') : t(lang, 'exportReport')}</button>
            {artifacts && <ArtifactTable artifacts={artifacts} lang={lang} />}
          </Card>
        )}
      </main>

      {profileToSwitch && (
        <div className="modal-backdrop" role="presentation">
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="switch-title">
            <div className="panel-head">
              <h2 id="switch-title">{profileToSwitch.short_label || profileToSwitch.label}</h2>
              <button className="btn ghost" type="button" onClick={() => setProfileToSwitch(null)}>{t(lang, 'close')}</button>
            </div>
            <p>{profileText(profileToSwitch, lang, 'description')}</p>
            <div className="button-row">
              <button className="btn primary" type="button" onClick={() => switchModel('rebuild')}>{t(lang, 'rebuildEnrollment')}</button>
              <button className="btn ghost" type="button" onClick={() => switchModel('clear')}>{t(lang, 'clearEnrollment')}</button>
              <button className="btn ghost" type="button" onClick={() => setProfileToSwitch(null)}>{t(lang, 'cancel')}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function EnrollmentSummary({ enrollment, lang }: { enrollment: EnrollmentStatus | null; lang: Lang }) {
  const words = Object.entries(enrollment?.enrolled || {});
  return (
    <div className="section-gap">
      <h3>{t(lang, 'enrolledKeywords')}</h3>
      {!words.length && <p className="empty">{t(lang, 'noEnrollment')}</p>}
      <div className="chip-list">
        {words.map(([word, item]) => (
          <span className="chip" key={word}>{word}<small>{item.count} samples · thr {item.threshold ?? '-'}</small></span>
        ))}
      </div>
    </div>
  );
}

function DetectionResult({ result, lang, compact = false }: { result: DetectResult; lang: Lang; compact?: boolean }) {
  return (
    <article className={`detect-card ${result.detected ? 'ok' : 'miss'} ${compact ? 'compact' : ''}`}>
      <div className="detect-head">
        <div>
          <span className="eyebrow">{result.detected ? 'Detected' : 'Rejected'}</span>
          <h3>{result.keyword || 'unknown'}</h3>
        </div>
        <span className={`badge ${result.detected ? 'success' : 'danger'}`}>{result.detected ? 'OK' : 'UNKNOWN'}</span>
      </div>
      <div className="metric-grid tight">
        <MetricCard label="L2" value={num(result.distance, 4)} />
        <MetricCard label={t(lang, 'threshold')} value={num(result.threshold, 3)} />
        <MetricCard label="Margin" value={num(result.margin, 4)} />
      </div>
      <div className="candidate-box">
        <strong>{t(lang, 'topCandidates')}</strong>
        <TopCandidates items={result.top_3} />
      </div>
      {!compact && <PolicyCards settings={result.settings} lang={lang} />}
    </article>
  );
}

type TimingMatchRow = { timing: Timing; segment?: LongSegment; ok: boolean; reason: string };

type LongDetectionRow = {
  index: number;
  segment: LongSegment;
  predicted: string;
  expected?: Timing;
  ok?: boolean;
  status: 'OK' | 'ERR' | 'UNKNOWN' | 'EXTRA';
  reason: string;
  overlap: number;
};

function findBestTiming(segment: LongSegment, timings: Timing[]): { timing?: Timing; overlapAmount: number } {
  let timing: Timing | undefined;
  let overlapAmount = 0;
  for (const item of timings) {
    const amount = overlap(item.start_sec, item.end_sec, segment.t0, segment.t1);
    if (amount > overlapAmount) {
      timing = item;
      overlapAmount = amount;
    }
  }
  return { timing, overlapAmount };
}

function detectionReason(segment: LongSegment, predicted: string, expected: Timing | undefined, ok: boolean | undefined, lang: Lang): string {
  if (!expected) return t(lang, 'noTimingOverlap');
  if (ok) return 'OK';
  if (!segment.detected) {
    if (Number(segment.distance) > Number(segment.threshold)) return t(lang, 'rejectedByThreshold');
    if (Number(segment.margin || 0) < Number(segment.accept_margin || 0)) return t(lang, 'rejectedByGuard');
    return t(lang, 'rejectedAsUnknown');
  }
  return `${t(lang, 'predicted')}: ${predicted}`;
}

function formatTiming(item?: Timing): string {
  if (!item) return '-';
  return `${item.label} (${num(item.start_sec, 2)}s-${num(item.end_sec, 2)}s)`;
}

function SequenceStrip({ title, words, lang }: { title: string; words: string[]; lang: Lang }) {
  const [expanded, setExpanded] = useState(false);
  const limit = 42;
  const visible = expanded ? words : words.slice(0, limit);
  if (!words.length) return null;
  return (
    <section className="sequence-strip">
      <header>
        <h3>{title}</h3>
        <span className="badge neutral">{words.length}</span>
      </header>
      <div className="sequence-chips">
        {visible.map((word, index) => (
          <span className={`sequence-chip ${word === 'unknown' ? 'unknown' : ''}`} key={`${word}-${index}`}>
            <small>{index + 1}</small>
            {word}
          </span>
        ))}
      </div>
      {words.length > limit && (
        <button className="link-button" type="button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? t(lang, 'collapse') : `${t(lang, 'showAll')} ${words.length}`}
        </button>
      )}
    </section>
  );
}

function LongReviewCard({ row, lang }: { row: TimingMatchRow; lang: Lang }) {
  const segment = row.segment;
  const predicted = segment ? (segment.detected ? segment.keyword : 'unknown') : '-';
  return (
    <article className="review-card miss">
      <div className="review-card-head">
        <div>
          <span className="eyebrow">#{row.timing.label}</span>
          <h3>{num(row.timing.start_sec, 2)}s - {num(row.timing.end_sec, 2)}s</h3>
        </div>
        <span className="badge danger">MISS</span>
      </div>
      <div className="review-pair">
        <div>
          <span className="eyebrow">{t(lang, 'predicted')}</span>
          <strong>{predicted}</strong>
        </div>
        <div>
          <span className="eyebrow">{t(lang, 'expected')}</span>
          <strong>{row.timing.label}</strong>
        </div>
      </div>
      <p className="alert warn">{row.reason}</p>
      {segment && (
        <div className="review-meta">
          <MetricCard label="L2" value={num(segment.distance, 4)} />
          <MetricCard label={t(lang, 'threshold')} value={num(segment.threshold, 3)} />
          <MetricCard label="Margin" value={num(segment.margin, 4)} />
          <MetricCard label={t(lang, 'acceptMargin')} value={num(segment.accept_margin, 4)} />
        </div>
      )}
      {segment && (
        <div className="candidate-box">
          <strong>{t(lang, 'topCandidates')}</strong>
          <TopCandidates items={segment.top_3} />
        </div>
      )}
    </article>
  );
}

function LongDetectionTable({ rows, lang }: { rows: LongDetectionRow[]; lang: Lang }) {
  if (!rows.length) return null;
  return (
    <div className="compact-result-table">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>{t(lang, 'time')}</th>
            <th>{t(lang, 'predicted')}</th>
            <th>{t(lang, 'expected')}</th>
            <th>{t(lang, 'match')}</th>
            <th>L2</th>
            <th>{t(lang, 'threshold')}</th>
            <th>Margin</th>
            <th>{t(lang, 'status')}</th>
            <th>{t(lang, 'details')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr className={`result-row ${row.status.toLowerCase()}`} key={`${row.segment.t0}-${row.index}`}>
              <td>{row.index}</td>
              <td>{num(row.segment.t0, 2)}s - {num(row.segment.t1, 2)}s</td>
              <td><strong>{row.predicted}</strong></td>
              <td>{formatTiming(row.expected)}</td>
              <td>
                <span className={`badge ${row.status === 'OK' ? 'success' : row.status === 'EXTRA' ? 'neutral' : 'danger'}`}>
                  {row.status}
                </span>
              </td>
              <td>{num(row.segment.distance, 4)}</td>
              <td>{num(row.segment.threshold, 3)}</td>
              <td>{num(row.segment.margin, 4)}</td>
              <td>{row.reason}</td>
              <td>
                <details className="row-details">
                  <summary>{t(lang, 'details')}</summary>
                  <TopCandidates items={row.segment.top_3} />
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LongResultView({ result, labels, timings, matches, lang }: {
  result: LongResult;
  labels: string[];
  timings: Timing[];
  matches: { matched: number; rows: TimingMatchRow[] };
  lang: Lang;
}) {
  const expectedSequence = timings.length ? timings.map((item) => item.label) : labels;
  const expectedCount = expectedSequence.length || 0;
  const allAcc = timings.length ? matches.matched / timings.length : undefined;
  const detectedSequence = result.sequence?.length
    ? result.sequence
    : result.results.map((segment) => (segment.detected ? segment.keyword : 'unknown'));
  const detectedTimeline = result.results.map((segment) => ({
    label: segment.detected ? segment.keyword : 'unknown',
    start_sec: segment.t0,
    end_sec: segment.t1,
    ok: segment.detected
  }));
  const tableRows: LongDetectionRow[] = result.results.map((segment, index) => {
    const { timing, overlapAmount } = findBestTiming(segment, timings);
    const predicted = segment.detected ? segment.keyword : 'unknown';
    const ok = timing ? predicted === timing.label : undefined;
    const status: LongDetectionRow['status'] = timing
      ? ok ? 'OK' : segment.detected ? 'ERR' : 'UNKNOWN'
      : segment.detected ? 'EXTRA' : 'UNKNOWN';
    return {
      index: index + 1,
      segment,
      predicted,
      expected: overlapAmount > 0 ? timing : undefined,
      ok,
      status,
      reason: detectionReason(segment, predicted, overlapAmount > 0 ? timing : undefined, ok, lang),
      overlap: overlapAmount
    };
  });
  const missedRows = matches.rows.filter((row) => !row.ok);

  return (
    <div className="long-result-layout">
      <div className="metric-grid long-summary-grid">
        <MetricCard label="Duration" value={`${num(result.duration, 1)}s`} />
        <MetricCard label="Expected" value={expectedCount || '-'} />
        <MetricCard label="Detected" value={result.results.length} />
        <MetricCard label="Matched" value={timings.length ? `${matches.matched}/${timings.length}` : '-'} tone={allAcc && allAcc > 0.8 ? 'good' : 'warn'} />
        <MetricCard label="Accuracy" value={allAcc === undefined ? '-' : pct(allAcc)} tone={metricTone(allAcc)} />
      </div>
      <PolicyCards settings={result.settings} lang={lang} />
      <div className="sequence-grid">
        <SequenceStrip title={t(lang, 'expectedSequence')} words={expectedSequence} lang={lang} />
        <SequenceStrip title={t(lang, 'detectedSequence')} words={detectedSequence} lang={lang} />
      </div>
      {timings.length > 0 && <Timeline title={t(lang, 'expectedTimeline')} timings={timings} duration={result.duration} color="expected" />}
      <Timeline title={t(lang, 'detectedTimeline')} timings={detectedTimeline} duration={result.duration} color="detected" />
      <section className="result-review-section">
        <div className="section-title-row">
          <h3>{t(lang, 'missedExpected')}</h3>
          <span className={`badge ${missedRows.length ? 'danger' : 'success'}`}>{missedRows.length}</span>
        </div>
        {missedRows.length ? (
          <div className="review-list">
            {missedRows.map((row, index) => (
              <LongReviewCard row={row} lang={lang} key={`${row.timing.label}-${index}`} />
            ))}
          </div>
        ) : (
          <p className="alert info">{t(lang, 'noIssues')}</p>
        )}
      </section>
      <section className="result-table-section">
        <div className="section-title-row">
          <h3>{t(lang, 'allDetections')}</h3>
          <span className="badge neutral">{tableRows.length}</span>
        </div>
        <LongDetectionTable rows={tableRows} lang={lang} />
      </section>
    </div>
  );
}

function OpenSetView({ result, lang }: { result: OpenSetResult; lang: Lang }) {
  const s = result.summary;
  return (
    <div className="section-gap">
      <div className="metric-grid">
        <MetricCard label="Known" value={s.known_tested} />
        <MetricCard label="Unknown" value={s.unknown_tested} />
        <MetricCard label="Candidates" value={result.candidate_words.length} />
        <MetricCard label="Balanced" value={pct(s.balanced_score)} tone={metricTone(s.balanced_score, 0.65)} />
        <MetricCard label="Open-set ACC" value={pct(s.open_set_acc)} tone={metricTone(s.open_set_acc, 0.65)} />
        <MetricCard label="KW-ACC" value={pct(s.keyword_acc)} tone={metricTone(s.keyword_acc, 0.65)} />
        <MetricCard label="Unknown reject" value={pct(s.unknown_reject_acc)} tone={metricTone(s.unknown_reject_acc, 0.65)} />
        <MetricCard label="FAR" value={pct(s.false_accept_rate)} tone={s.false_accept_rate > 0.3 ? 'warn' : 'good'} />
        <MetricCard label="False reject" value={pct(s.false_reject_rate)} tone={s.false_reject_rate > 0.3 ? 'warn' : 'good'} />
      </div>
      <PolicyCards settings={result.settings} lang={lang} />
      <WordSet title={t(lang, 'knownWords')} words={result.known_words} />
      <WordSet title={t(lang, 'unknownWords')} words={result.unknown_words} />
      <CaseSection title="False accepts" items={result.false_accepts} />
      <CaseSection title="Known misses" items={result.known_misses} />
    </div>
  );
}

function CalibrationView({ data, lang, onApply }: { data: CalibrationResult; lang: Lang; onApply: (row: CalibrationRow) => void }) {
  const options = [
    [t(lang, 'bestBalanced'), data.best_balanced],
    [t(lang, 'bestReject'), data.best_open_set],
    [t(lang, 'bestKeyword'), data.best_keyword]
  ] as const;
  return (
    <div className="section-gap">
      <h3>Calibration</h3>
      <div className="card-grid">
        {options.map(([label, row]) => (
          <article className="calibration-card" key={label}>
            <h4>{label}</h4>
            <div className="metric-grid tight">
              <MetricCard label="Balanced" value={pct(row.balanced_score)} />
              <MetricCard label="Threshold" value={num(row.threshold, 2)} />
              <MetricCard label="Guard" value={row.close_word_guard ? 'ON' : 'OFF'} />
              <MetricCard label="Per-class" value={row.use_per_class ? 'ON' : 'OFF'} />
              <MetricCard label="Margin" value={num(row.accept_margin, 4)} />
            </div>
            <button className="btn ghost" type="button" onClick={() => onApply(row)}>{t(lang, 'applySettings')}</button>
          </article>
        ))}
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>#</th><th>Balanced</th><th>KW</th><th>Reject</th><th>FAR</th><th>Thr</th><th>Guard</th><th>Per-class</th><th>Margin</th></tr>
          </thead>
          <tbody>
            {data.rows.slice(0, 20).map((row, index) => (
              <tr key={`${row.threshold}-${row.accept_margin}-${row.use_per_class}-${index}`}>
                <td>{index + 1}</td><td>{pct(row.balanced_score)}</td><td>{pct(row.keyword_acc)}</td><td>{pct(row.unknown_reject_acc)}</td><td>{pct(row.false_accept_rate)}</td><td>{num(row.threshold, 2)}</td><td>{row.close_word_guard ? 'ON' : 'OFF'}</td><td>{row.use_per_class ? 'ON' : 'OFF'}</td><td>{num(row.accept_margin, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function WordSet({ title, words }: { title: string; words: string[] }) {
  return (
    <div className="word-box">
      <strong>{title}</strong>
      <div className="chip-list">{words.map((word) => <span className="chip small" key={word}>{word}</span>)}</div>
    </div>
  );
}

function openSetCaseTitle(item: OpenSetCase): string {
  if (item.status === 'false_accept' || (item.expected === 'unknown' && item.predicted !== 'unknown')) {
    return 'False accept';
  }
  if (item.status === 'false_reject' || item.predicted === 'unknown') {
    return 'False reject';
  }
  if (item.status === 'wrong_keyword') {
    return 'Wrong keyword';
  }
  return item.status || 'Check';
}

function openSetCaseReason(item: OpenSetCase): string {
  const distance = num(item.distance, 4);
  const threshold = num(item.threshold, 3);
  const margin = num(item.margin, 4);
  const acceptMargin = num(item.accept_margin, 4);
  if (item.status === 'false_accept' || (item.expected === 'unknown' && item.predicted !== 'unknown')) {
    return `This is an unknown sample for "${item.word}", but the model accepted it as "${item.predicted}" because L2 ${distance} is below threshold ${threshold} and margin ${margin} passed accept margin ${acceptMargin}.`;
  }
  if (item.status === 'false_reject' || item.predicted === 'unknown') {
    return `This is a known keyword "${item.word}", but it was rejected as unknown. Check whether L2 ${distance} is above threshold ${threshold}, or whether margin ${margin} is below accept margin ${acceptMargin}.`;
  }
  if (item.status === 'wrong_keyword') {
    return `This is a known keyword "${item.word}", but the nearest accepted label was "${item.predicted}".`;
  }
  return 'This case needs manual review.';
}

function CaseSection({ title, items }: { title: string; items: OpenSetCase[] }) {
  if (!items?.length) return null;
  return (
    <div className="case-list">
      <h3>{title}</h3>
      {items.slice(0, 20).map((item, index) => (
        <article className="detect-card miss" key={`${item.word}-${index}`}>
          <div className="detect-head">
            <div>
              <span className="eyebrow">{item.file || item.path || item.word}</span>
              <h3>{openSetCaseTitle(item)}</h3>
            </div>
            <span className="badge danger">{item.status || 'CHECK'}</span>
          </div>
          <div className="case-meta-grid">
            <div>
              <span className="eyebrow">True word</span>
              <strong>{item.word || '-'}</strong>
            </div>
            <div>
              <span className="eyebrow">Expected</span>
              <strong>{item.expected || '-'}</strong>
            </div>
            <div>
              <span className="eyebrow">Predicted</span>
              <strong>{item.predicted || '-'}</strong>
            </div>
          </div>
          <div className="metric-grid tight">
            <MetricCard label="L2" value={num(item.distance, 4)} />
            <MetricCard label="Thr" value={num(item.threshold, 3)} />
            <MetricCard label="Margin" value={num(item.margin, 4)} />
            <MetricCard label="Accept margin" value={num(item.accept_margin, 4)} />
          </div>
          <p className="case-reason">{openSetCaseReason(item)}</p>
          <TopCandidates items={item.top_3} />
        </article>
      ))}
    </div>
  );
}

function ArtifactTable({ artifacts, lang }: { artifacts: ArtifactStatus; lang: Lang }) {
  return (
    <div className="table-scroll">
      <table>
        <thead><tr><th>Artifact</th><th>Status</th><th>Role</th><th>Evidence</th><th>Notes</th></tr></thead>
        <tbody>
          {artifacts.records.map((record) => (
            <tr key={record.id}>
              <td>{record.label}</td>
              <td><span className={`badge ${record.exists ? 'success' : 'danger'}`}>{record.status}</span></td>
              <td>{record.role}</td>
              <td>{record.exists ? 'yes' : 'missing'}</td>
              <td>{lang === 'vi' ? record.notes_vi : record.notes_en}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
