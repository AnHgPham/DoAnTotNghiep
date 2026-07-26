import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  AlertCircle,
  AudioLines,
  AudioWaveform,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  CircleStop,
  CircleX,
  Cpu,
  Download,
  FileText,
  FileUp,
  FlaskConical,
  Info,
  Lightbulb,
  Mic,
  Microscope,
  Play,
  Radar,
  RadioTower,
  RefreshCw,
  SearchCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
  UserCheck,
  X,
  Zap,
  type LucideIcon
} from 'lucide-react';
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
type Tone = 'primary' | 'success' | 'neutral' | 'danger';
type MetricTone = 'good' | 'warn' | '';

const GSC_17_KNOWN = 'yes,stop,happy,bird,dog,tree,marvin,four,learn,wow,sheila,zero,down,left,right,off,three';
const GSC_17_UNKNOWN = 'no,go,up,on,one,two,five,six,seven,eight,nine,bed,cat,house,backward,forward,follow';

const tabs: { id: Tab; labelKey: TextKey; icon: string }[] = [
  { id: 'enroll', labelKey: 'enrollment', icon: 'how_to_reg' },
  { id: 'single', labelKey: 'singleDetect', icon: 'search_check' },
  { id: 'long', labelKey: 'longAudio', icon: 'graphic_eq' },
  { id: 'openset', labelKey: 'openSet', icon: 'biotech' },
  { id: 'streaming', labelKey: 'streaming', icon: 'settings_input_antenna' },
  { id: 'model', labelKey: 'modelInfo', icon: 'memory' },
  { id: 'reports', labelKey: 'reports', icon: 'assessment' }
];

const inputClass =
  'w-full bg-surface-container-low border border-outline-variant rounded-lg px-4 py-2.5 text-on-surface font-body-md text-body-md placeholder:text-outline focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition';
const btnBase =
  'inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg font-title-md text-[14px] font-medium transition-colors disabled:opacity-60 disabled:cursor-progress';
const btnPrimary = `${btnBase} bg-primary text-on-primary hover:bg-on-primary-fixed-variant shadow-sm`;
const btnGhost = `${btnBase} border border-primary text-primary hover:bg-primary/5`;
const btnDanger = `${btnBase} bg-error/10 text-error border border-error/20 hover:bg-error/20`;

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

function metricTone(value: number | undefined, goodAt = 0.8): MetricTone {
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

/* ------------------------------ Primitives ------------------------------ */

const iconMap: Record<string, LucideIcon> = {
  assessment: BarChart3,
  biotech: Microscope,
  bolt: Zap,
  cancel: CircleX,
  check_circle: CheckCircle2,
  close: X,
  delete: Trash2,
  description: FileText,
  download: Download,
  error: AlertCircle,
  graphic_eq: AudioLines,
  how_to_reg: UserCheck,
  info: Info,
  lightbulb: Lightbulb,
  memory: Cpu,
  mic: Mic,
  model_training: BrainCircuit,
  play_arrow: Play,
  radar: Radar,
  record_voice_over: AudioWaveform,
  refresh: RefreshCw,
  science: FlaskConical,
  search_check: SearchCheck,
  settings_input_antenna: RadioTower,
  stop_circle: CircleStop,
  sync: RefreshCw,
  tune: SlidersHorizontal,
  upload: Upload,
  upload_file: FileUp
};

function Icon({ name, className = '', fill = false }: { name: string; className?: string; fill?: boolean }) {
  const Component = iconMap[name] ?? Info;
  return (
    <Component
      aria-hidden="true"
      className={className}
      size="1em"
      strokeWidth={fill ? 2.5 : 2}
    />
  );
}

function Card({ title, actions, icon, children, className = '' }: {
  title?: string;
  actions?: ReactNode;
  icon?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`glass-card min-w-0 rounded-xl p-5 md:p-stack-lg ${className}`}>
      {(title || actions) && (
        <div className="flex items-center justify-between gap-4 mb-stack-md flex-wrap">
          {title && (
            <h2 className="font-title-md text-title-md text-on-surface flex items-center gap-2">
              {icon && <Icon name={icon} className="text-primary text-[22px]" />}
              {title}
            </h2>
          )}
          {actions && <div className="flex items-center gap-2 flex-wrap">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  const map: Record<Tone, string> = {
    primary: 'bg-primary/10 text-primary border-primary/20',
    success: 'bg-success/10 text-success border-success/20',
    neutral: 'bg-surface-container-high text-on-surface-variant border-outline-variant/60',
    danger: 'bg-error/10 text-error border-error/20'
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full border font-metric-value text-metric-value whitespace-nowrap ${map[tone]}`}>
      {children}
    </span>
  );
}

function Metric({ label, value, tone = '' }: { label: string; value: ReactNode; tone?: MetricTone }) {
  const valueColor = tone === 'good' ? 'text-primary' : tone === 'warn' ? 'text-error' : 'text-on-surface';
  const border = tone === 'good' ? 'border-primary/30 bg-primary/5' : tone === 'warn' ? 'border-error/30 bg-error/5' : 'border-outline-variant bg-surface-container-lowest';
  return (
    <div className={`border ${border} rounded-lg p-4 hover:border-primary transition-colors min-w-0`}>
      <span className="font-metric-label text-metric-label uppercase text-on-surface-variant block mb-1 truncate">{label}</span>
      <span className={`font-metric-value text-[20px] leading-tight font-semibold break-words ${valueColor}`}>{value}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-2 min-w-0">
      <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">{label}</span>
      {children}
    </label>
  );
}

function Checkbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="inline-flex items-center gap-3 min-h-[46px] px-4 rounded-lg border border-outline-variant bg-surface-container-lowest cursor-pointer hover:border-primary transition-colors">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="w-5 h-5 accent-[#004ac6]" />
      <span className="font-body-md text-body-md text-on-surface">{label}</span>
    </label>
  );
}

function PolicyToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className={`flex items-center gap-3 min-h-[58px] px-4 rounded-lg border cursor-pointer transition-colors ${checked ? 'border-primary/40 bg-primary/5' : 'border-outline-variant bg-surface-container-lowest'}`}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="w-5 h-5 accent-[#004ac6]" />
      <span className="grid">
        <strong className="font-body-md text-body-md text-on-surface">{label}</strong>
        <small className={`font-metric-label text-metric-label ${checked ? 'text-primary' : 'text-on-surface-variant'}`}>{checked ? 'ON' : 'OFF'}</small>
      </span>
    </label>
  );
}

function TopCandidates({ items }: { items?: TopCandidate[] }) {
  if (!items?.length) return <p className="text-on-surface-variant font-body-sm text-body-sm m-0">-</p>;
  return (
    <ol className="flex flex-col gap-1.5 mt-2 list-none p-0 m-0">
      {items.slice(0, 3).map((item, index) => (
        <li key={`${item.word}-${index}`} className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-surface-container-lowest border border-outline-variant/60">
          <span className="font-body-md text-body-md text-on-surface font-medium flex items-center gap-2">
            <span className="font-metric-value text-metric-value text-on-surface-variant w-4">{index + 1}</span>
            {item.word}
          </span>
          <span className="font-metric-value text-metric-value text-on-surface-variant">{num(item.dist, 4)}</span>
        </li>
      ))}
    </ol>
  );
}

function PolicyCards({ settings, lang }: { settings?: DetectResult['settings']; lang: Lang }) {
  if (!settings) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      <Metric label={t(lang, 'threshold')} value={num(settings.threshold, 2)} />
      <Metric label={t(lang, 'perClass')} value={settings.use_per_class ? 'ON' : 'OFF'} tone={settings.use_per_class ? 'good' : 'warn'} />
      <Metric label={t(lang, 'closeGuard')} value={settings.close_word_guard ? 'ON' : 'OFF'} tone={settings.close_word_guard ? 'good' : 'warn'} />
      <Metric label={t(lang, 'acceptMargin')} value={num(settings.accept_margin, 4)} />
      <Metric label="Engine" value={settings.engine} />
      <Metric label="Model" value={settings.model_label || settings.model_profile || '-'} />
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
      aria-current={active ? 'true' : undefined}
      disabled={!profile.exists}
      onClick={onSelect}
      className={`text-left grid w-full min-w-0 overflow-hidden gap-3 min-h-[196px] p-5 rounded-xl border bg-surface-container-lowest transition-all hover-lift disabled:opacity-60 disabled:hover:translate-y-0 disabled:hover:shadow-none ${active ? 'border-primary ring-2 ring-primary/25' : 'border-outline-variant'} ${profile.exists ? '' : 'border-dashed'}`}
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <Badge tone={active ? 'primary' : profile.exists ? 'success' : 'danger'}>
          <Icon name={active ? 'check_circle' : profile.exists ? 'bolt' : 'error'} className="text-[14px]" />
          {active ? t(lang, 'active') : profile.exists ? t(lang, 'ready') : t(lang, 'missing')}
        </Badge>
        <span className="inline-flex min-w-0 max-w-[55%] overflow-hidden items-center gap-1 px-2.5 py-1 rounded-full border border-outline-variant/60 bg-surface-container-high text-on-surface-variant font-metric-value text-metric-value" title={profile.checkpoint_name || 'checkpoint'}>
          <span className="truncate">{profile.checkpoint_name || 'checkpoint'}</span>
        </span>
      </div>
      <h3 className="font-headline-lg text-[18px] leading-tight text-on-surface truncate" title={profile.short_label || profile.label}>{profile.short_label || profile.label}</h3>
      <p className="font-body-sm text-body-sm text-on-surface-variant line-clamp-2" title={profileText(profile, lang, 'description') || profileText(profile, lang, 'notes')}>{profileText(profile, lang, 'notes') || profileText(profile, lang, 'description')}</p>
      <div className="flex flex-wrap gap-2 mt-auto">
        {(profile.metrics || []).slice(0, 3).map((metric) => (
          <span key={metric.label} className="inline-flex min-w-0 max-w-full items-baseline gap-1 rounded-full border border-outline-variant bg-surface-container-low px-2.5 py-1 font-metric-value text-metric-value text-on-surface-variant">
            <strong className="text-primary">{metric.value}</strong>{metric.label}
          </span>
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
    <div className="grid gap-2">
      <div className="flex items-center justify-between">
        <div className="font-metric-label text-metric-label uppercase text-on-surface-variant">{title}</div>
        <Badge tone="neutral">{timings.length}</Badge>
      </div>
      <div className="overflow-x-auto border border-outline-variant rounded-lg bg-surface-container-low p-3 custom-scrollbar" tabIndex={0}>
        <div className="relative min-h-[74px]" style={{ minWidth }}>
          <div className="absolute left-0 right-0 bottom-[10px] h-[3px] rounded-full bg-outline-variant" aria-hidden="true" />
          {timings.map((item, index) => {
            const left = Math.max(0, Math.min(100, (item.start_sec / Math.max(duration, 0.001)) * 100));
            const width = Math.max(1.8, ((item.end_sec - item.start_sec) / Math.max(duration, 0.001)) * 100);
            const bg = color === 'expected' ? 'bg-tertiary' : item.ok === false ? 'bg-error' : 'bg-primary';
            return (
              <div
                key={`${item.label}-${index}`}
                className={`absolute top-3 h-9 min-w-[34px] flex items-center justify-center px-2 rounded-lg text-white font-metric-value text-[12px] overflow-hidden whitespace-nowrap shadow-sm ${bg}`}
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

/* -------------------------------- App ---------------------------------- */

export default function App() {
  const [lang, setLang] = useState<Lang>('vi');
  const [activeTab, setActiveTab] = useState<Tab>('enroll');
  const [profiles, setProfiles] = useState<ModelProfile[]>([]);
  const [activeProfile, setActiveProfile] = useState('');
  const [showAllModels, setShowAllModels] = useState(false);
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

  const [customPath, setCustomPath] = useState('');
  const [customFamily, setCustomFamily] = useState('auto');
  const [customFile, setCustomFile] = useState<File | null>(null);

  const [streaming, setStreaming] = useState(false);
  const [streamEvents, setStreamEvents] = useState<DetectResult[]>([]);
  const streamRefs = useRef<{ ws?: WebSocket; ctx?: AudioContext; source?: MediaStreamAudioSourceNode; processor?: ScriptProcessorNode; media?: MediaStream }>({});

  const activeModel = useMemo(
    () => profiles.find((profile) => profile.id === activeProfile) || null,
    [profiles, activeProfile]
  );
  const visibleProfiles = useMemo(
    () => showAllModels
      ? profiles
      : profiles.filter((profile) => profile.featured || profile.id === activeProfile),
    [activeProfile, profiles, showAllModels]
  );

  const activeTabMeta = tabs.find((tab) => tab.id === activeTab) || tabs[0];

  // unused helper kept to satisfy splitWords import usage
  void splitWords;

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

  async function discoverModels() {
    const data = await runTask('discover', () => apiPostForm<ModelProfilesResponse>('/api/model/discover', new FormData()));
    if (data) {
      setProfiles(data.profiles || []);
      setActiveProfile(data.active);
    }
  }

  async function loadCustomCheckpoint() {
    if (!customPath.trim()) {
      setError(t(lang, 'noFile'));
      return;
    }
    const result = await runTask('loadCustom', () => apiPostForm('/api/model/select', formFromObject({
      checkpoint_path: customPath.trim(),
      model_family: customFamily,
      enrollment_policy: 'clear'
    })));
    if (result) {
      setCustomPath('');
      await refreshAll();
    }
  }

  async function uploadCustomCheckpoint() {
    if (!customFile) {
      setError(t(lang, 'noFile'));
      return;
    }
    const result = await runTask('uploadCustom', () => apiPostForm('/api/model/upload', formFromObject({
      checkpoint: customFile,
      model_family: customFamily,
      enrollment_policy: 'clear'
    })));
    if (result) {
      setCustomFile(null);
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
      preset: 'manual',
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
    <div className="min-h-screen flex flex-col md:flex-row bg-background text-on-surface">
      {/* Sidebar */}
      <aside className="md:fixed md:left-0 md:top-0 md:h-full md:w-[280px] w-full flex flex-col bg-[#0F172A]/90 backdrop-blur-xl border-b md:border-b-0 md:border-r border-white/10 z-50">
        <div className="px-5 md:px-6 py-6 md:py-stack-lg flex flex-col h-full gap-6 md:gap-stack-lg">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shrink-0">
              <Icon name="science" className="text-white" fill />
            </div>
            <div className="min-w-0">
              <h1 className="font-headline-lg text-[20px] leading-tight font-bold text-white truncate">KWS Research</h1>
              <p className="font-body-sm text-body-sm text-inverse-primary/80 truncate">{t(lang, 'appTitle')}</p>
            </div>
          </div>

          <nav className="flex md:flex-col gap-1.5 overflow-x-auto md:overflow-visible custom-scrollbar" aria-label="Workflow">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 shrink-0 border-l-[3px] ${isActive ? 'bg-primary/15 text-white border-primary' : 'text-white/70 hover:text-white hover:bg-white/5 border-transparent'}`}
                >
                  <Icon name={tab.icon} fill={isActive} className="text-[22px]" />
                  <span className="font-body-md text-body-md font-medium whitespace-nowrap">{t(lang, tab.labelKey)}</span>
                </button>
              );
            })}
          </nav>

          <div className="mt-auto grid gap-3">
            <div className="grid grid-cols-2 gap-2" role="group" aria-label="Language">
              {(['vi', 'en'] as Lang[]).map((code) => (
                <button
                  key={code}
                  type="button"
                  onClick={() => setLang(code)}
                  className={`py-2.5 rounded-lg border text-sm font-semibold uppercase transition-colors ${lang === code ? 'border-primary-fixed-dim bg-primary/25 text-white' : 'border-white/10 bg-white/5 text-white/70 hover:bg-white/10'}`}
                >
                  {code}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => refreshAll().catch((err: Error) => setError(err.message))}
              className="w-full py-3 px-4 bg-white/5 hover:bg-white/10 text-white rounded-lg border border-white/10 transition-colors flex items-center justify-center gap-2 font-body-md text-body-md font-medium"
            >
              <Icon name="refresh" className="text-[18px]" />
              {t(lang, 'reload')}
            </button>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0 md:ml-[280px] min-h-screen">
        <header className="sticky top-0 z-40 h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant flex items-center justify-between px-4 md:px-margin-x">
          <div className="flex items-center gap-4 min-w-0">
            <span className="font-headline-lg text-[20px] font-black text-primary hidden sm:block">KWS Dashboard</span>
            {activeModel && (
              <span className="hidden md:inline-flex">
                <Badge tone="primary"><Icon name="model_training" className="text-[14px]" />{activeModel.short_label || activeModel.label}</Badge>
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            {busy && (
              <span className="font-metric-label text-metric-label uppercase text-on-surface-variant hidden sm:flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />{t(lang, 'running')}
              </span>
            )}
            <button type="button" onClick={() => refreshAll().catch((err: Error) => setError(err.message))} className={btnGhost}>
              <Icon name="refresh" className="text-[18px]" />
              <span className="hidden sm:inline">{t(lang, 'reload')}</span>
            </button>
          </div>
        </header>

        <main className="flex-1 p-4 md:p-margin-x max-w-container-max mx-auto w-full flex flex-col gap-gutter">
          {/* Page header */}
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-primary/10 text-primary flex items-center justify-center shrink-0">
              <Icon name={activeTabMeta.icon} fill className="text-[24px]" />
            </div>
            <div>
              <h2 className="font-display-lg text-headline-lg-mobile md:text-display-lg text-on-surface leading-tight">{t(lang, activeTabMeta.labelKey)}</h2>
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-1 max-w-2xl">{t(lang, 'appSubtitle')}</p>
            </div>
          </div>

          {/* Active model */}
          <Card
            title={t(lang, 'activeModel')}
            icon="memory"
            actions={
              <button type="button" onClick={discoverModels} disabled={busy === 'discover'} className={btnGhost}>
                <Icon name="sync" className="text-[18px]" />
                {busy === 'discover' ? t(lang, 'running') : (lang === 'vi' ? 'Quét lại checkpoint' : 'Rescan checkpoints')}
              </button>
            }
          >
            <div className="grid min-w-0 grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-gutter">
              {visibleProfiles.map((profile) => (
                <ModelCard
                  key={profile.id}
                  profile={profile}
                  active={profile.id === activeProfile}
                  lang={lang}
                  onSelect={() => setProfileToSwitch(profile)}
                />
              ))}
            </div>
            {profiles.length > visibleProfiles.length && (
              <div className="mt-4 flex justify-center">
                <button type="button" onClick={() => setShowAllModels(true)} className={btnGhost}>
                  {t(lang, 'showAll')} {profiles.length}
                </button>
              </div>
            )}
            {showAllModels && profiles.some((profile) => !profile.featured) && (
              <div className="mt-4 flex justify-center">
                <button type="button" onClick={() => setShowAllModels(false)} className={btnGhost}>
                  {t(lang, 'collapse')}
                </button>
              </div>
            )}
            {activeModel && <p className="font-body-sm text-body-sm text-on-surface-variant mt-stack-md">{profileText(activeModel, lang, 'description')}</p>}

            <div className="mt-stack-md rounded-xl border border-outline-variant bg-surface-container-low p-4">
              <div className="flex items-center gap-2 mb-3">
                <Icon name="upload_file" className="text-primary text-[20px]" />
                <h3 className="font-title-md text-title-md text-on-surface">{lang === 'vi' ? 'Nạp checkpoint theo đường dẫn' : 'Load a checkpoint by path'}</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] gap-3">
                <input
                  type="text"
                  value={customPath}
                  onChange={(event) => setCustomPath(event.target.value)}
                  placeholder="checkpoints/....pt"
                  className={inputClass}
                />
                <select value={customFamily} onChange={(event) => setCustomFamily(event.target.value)} className={inputClass}>
                  <option value="auto">auto</option>
                  <option value="dscnn">dscnn</option>
                  <option value="edgespot_full">edgespot_full</option>
                </select>
                <button type="button" onClick={loadCustomCheckpoint} disabled={busy === 'loadCustom'} className={btnPrimary}>
                  <Icon name="download" className="text-[18px]" />
                  {busy === 'loadCustom' ? t(lang, 'running') : (lang === 'vi' ? 'Nạp' : 'Load')}
                </button>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-2">
                {lang === 'vi'
                  ? 'Đường dẫn tương đối tới gốc dự án hoặc đường dẫn tuyệt đối tới file .pt. Thêm file mới vào thư mục checkpoints/ rồi bấm "Quét lại checkpoint".'
                  : 'Relative to the project root or an absolute path to a .pt file. Drop new files into checkpoints/ then click "Rescan checkpoints".'}
              </p>

              <div className="flex items-center gap-3 my-3">
                <div className="h-px flex-1 bg-outline-variant/60" />
                <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">{lang === 'vi' ? 'hoặc chọn file từ máy' : 'or pick a file from your computer'}</span>
                <div className="h-px flex-1 bg-outline-variant/60" />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3">
                <input
                  type="file"
                  accept=".pt"
                  onChange={(event) => setCustomFile(event.target.files?.[0] || null)}
                  className={`${inputClass} file:mr-3 file:rounded-md file:border-0 file:bg-primary file:text-on-primary file:px-3 file:py-1.5 file:font-medium file:cursor-pointer`}
                />
                <button type="button" onClick={uploadCustomCheckpoint} disabled={busy === 'uploadCustom'} className={btnPrimary}>
                  <Icon name="upload" className="text-[18px]" />
                  {busy === 'uploadCustom' ? t(lang, 'running') : (lang === 'vi' ? 'Tải lên & nạp' : 'Upload & load')}
                </button>
              </div>
              {customFile && (
                <p className="font-body-sm text-body-sm text-on-surface-variant mt-2 flex items-center gap-2">
                  <Icon name="description" className="text-primary text-[16px]" />
                  <span className="truncate" title={customFile.name}>{customFile.name}</span>
                  <span className="text-outline">({(customFile.size / 1_048_576).toFixed(1)} MB)</span>
                </p>
              )}
            </div>
          </Card>

          {error && (
            <div className="rounded-lg border border-error/30 bg-error/10 text-on-error-container px-4 py-3 font-body-md text-body-md flex items-center gap-2" role="alert">
              <Icon name="error" className="text-error" />
              <strong>{t(lang, 'error')}:</strong> {error}
            </div>
          )}

          {activeTab === 'enroll' && (
            <Card title={t(lang, 'enrollment')} icon="how_to_reg">
              <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-gutter">
                <Field label={t(lang, 'customWords')}>
                  <textarea value={enrollWords} onChange={(event) => setEnrollWords(event.target.value)} rows={4} className={`${inputClass} resize-y min-h-[110px]`} />
                </Field>
                <Field label={t(lang, 'samplesPerWord')}>
                  <input type="number" min={1} max={20} value={enrollK} onChange={(event) => setEnrollK(Number(event.target.value))} className={inputClass} />
                </Field>
              </div>
              <div className="flex flex-wrap gap-3 mt-stack-md">
                <button className={btnPrimary} type="button" onClick={enrollGsc} disabled={busy === 'enroll'}>
                  <Icon name="how_to_reg" className="text-[18px]" />{busy === 'enroll' ? t(lang, 'running') : t(lang, 'enroll')}
                </button>
                <button className={btnGhost} type="button" onClick={() => setEnrollWords(GSC_17_KNOWN)}>GSC 17 known</button>
                <button className={btnDanger} type="button" onClick={clearEnrollment}>
                  <Icon name="delete" className="text-[18px]" />{t(lang, 'clearAll')}
                </button>
              </div>
              <EnrollmentSummary enrollment={enrollment} lang={lang} />
              {presets && Object.keys(presets.presets || {}).length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-stack-md">
                  {Object.entries(presets.presets || {}).map(([name, words]) => (
                    <button key={name} type="button" onClick={() => setEnrollWords(words)} className="px-3 py-2.5 rounded-lg border border-outline-variant bg-surface-container-low text-on-surface-variant font-body-sm text-body-sm hover:border-primary hover:text-primary transition-colors text-left truncate">
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </Card>
          )}

          {activeTab === 'single' && (
            <Card title={t(lang, 'singleDetect')} icon="search_check">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter">
                <Field label={t(lang, 'uploadAudio')}>
                  <input type="file" accept="audio/*" onChange={(event) => setSingleFile(event.target.files?.[0] || null)} className={`${inputClass} file:mr-3 file:rounded-md file:border-0 file:bg-primary file:text-on-primary file:px-3 file:py-1.5 file:font-medium`} />
                </Field>
                <Field label={t(lang, 'threshold')}>
                  <input type="number" step={0.01} value={singleThreshold} onChange={(event) => setSingleThreshold(Number(event.target.value))} className={inputClass} />
                </Field>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 my-stack-md">
                <Checkbox label={t(lang, 'perClass')} checked={singlePerClass} onChange={setSinglePerClass} />
                <Checkbox label={t(lang, 'closeGuard')} checked={singleGuard} onChange={setSingleGuard} />
              </div>
              <button className={btnPrimary} type="button" onClick={detectSingle} disabled={busy === 'single'}>
                <Icon name="radar" className="text-[18px]" />{busy === 'single' ? t(lang, 'running') : t(lang, 'detect')}
              </button>
              {singleResult && <div className="mt-stack-md"><DetectionResult result={singleResult} lang={lang} /></div>}
            </Card>
          )}

          {activeTab === 'long' && (
            <Card title={t(lang, 'longAudio')} icon="graphic_eq">
              <div className="grid gap-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Field label={t(lang, 'uploadAudio')}>
                    <input type="file" accept="audio/*" onChange={(event) => setLongFile(event.target.files?.[0] || null)} className={`${inputClass} file:mr-3 file:rounded-md file:border-0 file:bg-primary file:text-on-primary file:px-3 file:py-1.5`} />
                  </Field>
                  <Field label={t(lang, 'labels')}>
                    <input type="file" accept=".txt,.csv" onChange={(event) => setLabelFile(event.target.files?.[0] || null)} className={`${inputClass} file:mr-3 file:rounded-md file:border-0 file:bg-surface-container-high file:text-on-surface file:px-3 file:py-1.5`} />
                  </Field>
                  <Field label={t(lang, 'timings')}>
                    <input type="file" accept=".json" onChange={(event) => setTimingFile(event.target.files?.[0] || null)} className={`${inputClass} file:mr-3 file:rounded-md file:border-0 file:bg-surface-container-high file:text-on-surface file:px-3 file:py-1.5`} />
                  </Field>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Field label={t(lang, 'threshold')}>
                    <input type="number" step={0.01} value={longThreshold} onChange={(event) => setLongThreshold(Number(event.target.value))} className={inputClass} />
                  </Field>
                  <Field label={t(lang, 'segmentation')}>
                    <select value={longSeg} onChange={(event) => setLongSeg(event.target.value)} className={inputClass}>
                      <option>Energy</option>
                      <option>Silero VAD</option>
                    </select>
                  </Field>
                  <Field label={t(lang, 'minDuration')}>
                    <input type="number" min={80} max={5000} value={longMinDur} onChange={(event) => setLongMinDur(Number(event.target.value))} className={inputClass} />
                  </Field>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
                  <PolicyToggle label={t(lang, 'perClass')} checked={longPerClass} onChange={setLongPerClass} />
                  <PolicyToggle label={t(lang, 'closeGuard')} checked={longGuard} onChange={setLongGuard} />
                  <button className={`${btnPrimary} min-h-[58px]`} type="button" onClick={detectLong} disabled={busy === 'long'}>
                    <Icon name="play_arrow" className="text-[20px]" />{busy === 'long' ? t(lang, 'running') : t(lang, 'runLongDetect')}
                  </button>
                </div>
              </div>
              {longResult && (
                <LongResultView result={longResult} labels={longLabels} timings={longTimings} matches={timingMatches} lang={lang} />
              )}
            </Card>
          )}

          {activeTab === 'openset' && (
            <Card title={t(lang, 'openSet')} icon="biotech">
              <div className="rounded-lg border border-primary/25 bg-primary/5 text-on-primary-fixed-variant px-4 py-3 font-body-sm text-body-sm flex items-start gap-2 mb-stack-md">
                <Icon name="lightbulb" className="text-primary text-[18px] mt-0.5" />
                <span><strong>{t(lang, 'recommended')}:</strong> {t(lang, 'guardRecommendation')}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter">
                <Field label={t(lang, 'knownWords')}>
                  <textarea value={openKnown} rows={3} onChange={(event) => setOpenKnown(event.target.value)} className={`${inputClass} resize-y min-h-[90px]`} />
                </Field>
                <Field label={t(lang, 'unknownWords')}>
                  <textarea value={openUnknown} rows={3} onChange={(event) => setOpenUnknown(event.target.value)} className={`${inputClass} resize-y min-h-[90px]`} />
                </Field>
                <Field label={t(lang, 'samplesPerWord')}>
                  <input type="number" min={1} max={10} value={openK} onChange={(event) => setOpenK(Number(event.target.value))} className={inputClass} />
                </Field>
                <Field label={t(lang, 'threshold')}>
                  <input type="number" step={0.01} value={openThreshold} onChange={(event) => setOpenThreshold(Number(event.target.value))} className={inputClass} />
                </Field>
                <Field label={t(lang, 'acceptMargin')}>
                  <input type="number" min={0} max={0.1} step={0.01} value={openMargin} onChange={(event) => setOpenMargin(Number(event.target.value))} className={inputClass} />
                </Field>
                <Field label="Seed">
                  <input type="number" value={openSeed} onChange={(event) => setOpenSeed(Number(event.target.value))} className={inputClass} />
                </Field>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 my-stack-md">
                <Checkbox label={t(lang, 'perClass')} checked={openPerClass} onChange={setOpenPerClass} />
                <Checkbox label={t(lang, 'closeGuard')} checked={openGuard} onChange={setOpenGuard} />
              </div>
              <div className="flex flex-wrap gap-3">
                <button className={btnPrimary} type="button" onClick={runOpenSet} disabled={busy === 'openset'}>
                  <Icon name="science" className="text-[18px]" />{busy === 'openset' ? t(lang, 'running') : t(lang, 'runOpenSet')}
                </button>
                <button className={btnGhost} type="button" onClick={runCalibration} disabled={busy === 'calibration'}>
                  <Icon name="tune" className="text-[18px]" />{busy === 'calibration' ? t(lang, 'running') : t(lang, 'runCalibration')}
                </button>
              </div>
              {openResult && <OpenSetView result={openResult} lang={lang} />}
              {calibration && <CalibrationView data={calibration} lang={lang} onApply={applyCalibration} />}
            </Card>
          )}

          {activeTab === 'streaming' && (
            <Card title={t(lang, 'streaming')} icon="settings_input_antenna">
              <div className="rounded-xl border border-outline-variant bg-surface-container-lowest relative overflow-hidden h-[200px] flex items-center justify-center mb-stack-md">
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-surface/90 backdrop-blur-md border border-outline-variant px-3 py-1.5 rounded-full">
                  <span className={`w-2.5 h-2.5 rounded-full ${streaming ? 'bg-error recording-dot' : 'bg-outline'}`} />
                  <span className="font-metric-label text-metric-label uppercase text-on-surface font-bold">{streaming ? 'Live Listening' : 'Idle'}</span>
                </div>
                <div className="flex items-center gap-1.5 h-24">
                  {Array.from({ length: 18 }).map((_, i) => (
                    <div
                      key={i}
                      className={`w-1 rounded-full bg-primary ${streaming ? 'animate-pulse' : 'opacity-30'}`}
                      style={{ height: `${10 + ((i * 37) % 70)}%`, animationDelay: `${(i % 6) * 0.12}s` }}
                    />
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-stack-md">
                <Metric label="State" value={streaming ? 'listening' : 'idle'} tone={streaming ? 'good' : ''} />
                <Metric label="Detections" value={streamEvents.length} />
                <Metric label="Last keyword" value={streamEvents[0]?.keyword || '-'} />
              </div>
              <button className={streaming ? btnDanger : btnPrimary} type="button" onClick={toggleStreaming}>
                <Icon name={streaming ? 'stop_circle' : 'mic'} className="text-[18px]" />
                {streaming ? t(lang, 'stopStreaming') : t(lang, 'startStreaming')}
              </button>
              <div className="grid gap-3 mt-stack-md">
                {streamEvents.map((event, index) => (
                  <DetectionResult key={index} result={event} lang={lang} compact />
                ))}
              </div>
            </Card>
          )}

          {activeTab === 'model' && (
            <Card title={t(lang, 'modelInfo')} icon="info">
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {(artifacts?.records || []).map((record) => (
                  <Metric
                    key={record.id}
                    label={record.status}
                    value={record.exists ? record.label : `${record.label} (missing)`}
                    tone={record.exists ? 'good' : 'warn'}
                  />
                ))}
              </div>
            </Card>
          )}

          {activeTab === 'reports' && (
            <Card title={t(lang, 'reports')} icon="assessment">
              <button className={btnPrimary} type="button" onClick={exportReport} disabled={busy === 'export'}>
                <Icon name="download" className="text-[18px]" />{busy === 'export' ? t(lang, 'running') : t(lang, 'exportReport')}
              </button>
              {artifacts && <div className="mt-stack-md"><ArtifactTable artifacts={artifacts} lang={lang} /></div>}
            </Card>
          )}
        </main>
      </div>

      {profileToSwitch && (
        <div className="fixed inset-0 z-50 grid place-items-center p-6 bg-[#0b1c30]/45 backdrop-blur-sm" role="presentation" onClick={() => setProfileToSwitch(null)}>
          <div className="w-[min(640px,100%)] max-h-[calc(100vh-48px)] overflow-auto bg-surface-container-lowest border border-outline-variant rounded-xl shadow-2xl p-6" role="dialog" aria-modal="true" aria-labelledby="switch-title" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-center justify-between gap-4 mb-4">
              <h2 id="switch-title" className="font-headline-lg text-[20px] text-on-surface">{profileToSwitch.short_label || profileToSwitch.label}</h2>
              <button className="text-on-surface-variant hover:text-primary transition-colors" type="button" onClick={() => setProfileToSwitch(null)} aria-label={t(lang, 'close')}>
                <Icon name="close" />
              </button>
            </div>
            <p className="font-body-md text-body-md text-on-surface-variant mb-stack-md">{profileText(profileToSwitch, lang, 'description')}</p>
            <div className="flex flex-wrap gap-3">
              <button className={btnPrimary} type="button" onClick={() => switchModel('rebuild')}>{t(lang, 'rebuildEnrollment')}</button>
              <button className={btnGhost} type="button" onClick={() => switchModel('clear')}>{t(lang, 'clearEnrollment')}</button>
              <button className={btnGhost} type="button" onClick={() => setProfileToSwitch(null)}>{t(lang, 'cancel')}</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------ Sub-views ------------------------------ */

function EnrollmentSummary({ enrollment, lang }: { enrollment: EnrollmentStatus | null; lang: Lang }) {
  const words = Object.entries(enrollment?.enrolled || {});
  return (
    <div className="grid gap-3 mt-stack-md">
      <h3 className="font-title-md text-title-md text-on-surface">{t(lang, 'enrolledKeywords')}</h3>
      {!words.length && <p className="font-body-md text-body-md text-on-surface-variant m-0">{t(lang, 'noEnrollment')}</p>}
      <div className="flex flex-wrap gap-2.5">
        {words.map(([word, item]) => (
          <span key={word} className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary-fixed/40 px-3.5 py-2 text-on-primary-fixed-variant">
            <Icon name="record_voice_over" className="text-[16px]" />
            <span className="font-body-md text-body-md font-semibold">{word}</span>
            <small className="font-metric-label text-metric-label text-secondary">{item.count} · thr {item.threshold ?? '-'}</small>
          </span>
        ))}
      </div>
    </div>
  );
}

function DetectionResult({ result, lang, compact = false }: { result: DetectResult; lang: Lang; compact?: boolean }) {
  const ok = result.detected;
  return (
    <article className={`grid gap-4 rounded-xl border bg-surface-container-lowest border-l-4 ${ok ? 'border-l-success border-outline-variant' : 'border-l-error border-error/30 bg-error/5'} ${compact ? 'p-4' : 'p-5'}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">{ok ? 'Detected' : 'Rejected'}</span>
          <h3 className="font-headline-lg text-[24px] leading-tight text-on-surface mt-1 break-words">{result.keyword || 'unknown'}</h3>
        </div>
        <Badge tone={ok ? 'success' : 'danger'}>
          <Icon name={ok ? 'check_circle' : 'cancel'} className="text-[14px]" />{ok ? 'OK' : 'UNKNOWN'}
        </Badge>
      </div>
      <div className={`grid gap-3 ${result.timing_ms ? 'grid-cols-2 md:grid-cols-4' : 'grid-cols-3'}`}>
        <Metric label="L2" value={num(result.distance, 4)} />
        <Metric label={t(lang, 'threshold')} value={num(result.threshold, 3)} />
        <Metric label="Margin" value={num(result.margin, 4)} />
        {result.timing_ms && <Metric label={t(lang, 'latency')} value={`${num(result.timing_ms.total, 0)} ms`} tone="good" />}
      </div>
      <div className="rounded-lg border border-outline-variant bg-surface-container-low p-4">
        <strong className="font-metric-label text-metric-label uppercase text-primary">{t(lang, 'topCandidates')}</strong>
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
    <section className="grid gap-3 rounded-lg border border-outline-variant bg-surface-container-low p-4 min-w-0">
      <header className="flex items-center justify-between gap-3">
        <h3 className="font-title-md text-title-md text-on-surface">{title}</h3>
        <Badge tone="neutral">{words.length}</Badge>
      </header>
      <div className="flex flex-wrap gap-2 max-h-[142px] overflow-auto custom-scrollbar pr-1">
        {visible.map((word, index) => (
          <span key={`${word}-${index}`} className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-metric-value text-metric-value ${word === 'unknown' ? 'border-error/25 bg-error/10 text-error' : 'border-primary/20 bg-primary-fixed/40 text-on-primary-fixed-variant'}`}>
            <small className="text-secondary font-semibold">{index + 1}</small>
            {word}
          </span>
        ))}
      </div>
      {words.length > limit && (
        <button className="justify-self-start text-primary font-body-sm text-body-sm font-semibold hover:underline" type="button" onClick={() => setExpanded((value) => !value)}>
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
    <article className="grid gap-3 rounded-xl border border-outline-variant border-l-4 border-l-error bg-error/5 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">#{row.timing.label}</span>
          <h3 className="font-title-md text-title-md text-on-surface">{num(row.timing.start_sec, 2)}s - {num(row.timing.end_sec, 2)}s</h3>
        </div>
        <Badge tone="danger">MISS</Badge>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">{t(lang, 'predicted')}</span>
          <strong className="block mt-1 font-body-md text-body-md text-on-surface">{predicted}</strong>
        </div>
        <div>
          <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">{t(lang, 'expected')}</span>
          <strong className="block mt-1 font-body-md text-body-md text-on-surface">{row.timing.label}</strong>
        </div>
      </div>
      <p className="rounded-lg border border-warn/0 bg-surface-container-high px-3 py-2 font-body-sm text-body-sm text-on-surface-variant m-0">{row.reason}</p>
      {segment && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Metric label="L2" value={num(segment.distance, 4)} />
          <Metric label={t(lang, 'threshold')} value={num(segment.threshold, 3)} />
          <Metric label="Margin" value={num(segment.margin, 4)} />
          <Metric label={t(lang, 'acceptMargin')} value={num(segment.accept_margin, 4)} />
        </div>
      )}
      {segment && (
        <div className="rounded-lg border border-outline-variant bg-surface-container-low p-4">
          <strong className="font-metric-label text-metric-label uppercase text-primary">{t(lang, 'topCandidates')}</strong>
          <TopCandidates items={segment.top_3} />
        </div>
      )}
    </article>
  );
}

function LongDetectionTable({ rows, lang }: { rows: LongDetectionRow[]; lang: Lang }) {
  if (!rows.length) return null;
  return (
    <div className="w-full overflow-x-auto border border-outline-variant rounded-xl bg-surface-container-lowest custom-scrollbar">
      <table className="w-full border-collapse min-w-[1080px]">
        <thead>
          <tr className="bg-surface-container-low">
            {['#', t(lang, 'time'), t(lang, 'predicted'), t(lang, 'expected'), t(lang, 'match'), 'L2', t(lang, 'threshold'), 'Margin', t(lang, 'status'), t(lang, 'details')].map((head) => (
              <th key={head} className="sticky top-0 bg-surface-container-low text-left px-3.5 py-3 font-metric-label text-metric-label uppercase text-on-surface-variant border-b border-outline-variant">{head}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.segment.t0}-${row.index}`} className={`border-b border-outline-variant/40 ${row.status === 'ERR' || row.status === 'UNKNOWN' ? 'bg-warn/0' : ''} hover:bg-surface-container-low`}>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface-variant">{row.index}</td>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface">{num(row.segment.t0, 2)}s - {num(row.segment.t1, 2)}s</td>
              <td className="px-3.5 py-3 font-body-md text-body-md text-on-surface font-medium">{row.predicted}</td>
              <td className="px-3.5 py-3 font-body-sm text-body-sm text-on-surface-variant">{formatTiming(row.expected)}</td>
              <td className="px-3.5 py-3">
                <Badge tone={row.status === 'OK' ? 'success' : row.status === 'EXTRA' ? 'neutral' : 'danger'}>{row.status}</Badge>
              </td>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface">{num(row.segment.distance, 4)}</td>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface">{num(row.segment.threshold, 3)}</td>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface">{num(row.segment.margin, 4)}</td>
              <td className="px-3.5 py-3 font-body-sm text-body-sm text-on-surface-variant">{row.reason}</td>
              <td className="px-3.5 py-3">
                <details>
                  <summary className="cursor-pointer text-primary font-body-sm text-body-sm font-semibold">{t(lang, 'details')}</summary>
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
  const processingSpeed = result.timing_ms?.total
    ? (result.duration * 1000) / result.timing_ms.total
    : undefined;

  return (
    <div className="grid gap-stack-md mt-stack-md">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <Metric label="Duration" value={`${num(result.duration, 1)}s`} />
        <Metric label="Expected" value={expectedCount || '-'} />
        <Metric label="Detected" value={result.results.length} />
        <Metric label="Matched" value={timings.length ? `${matches.matched}/${timings.length}` : '-'} tone={allAcc && allAcc > 0.8 ? 'good' : 'warn'} />
        <Metric label="Accuracy" value={allAcc === undefined ? '-' : pct(allAcc)} tone={metricTone(allAcc)} />
        <Metric label={t(lang, 'processingSpeed')} value={processingSpeed ? `${num(processingSpeed, 1)}x` : '-'} tone={processingSpeed && processingSpeed >= 1 ? 'good' : 'warn'} />
      </div>
      <PolicyCards settings={result.settings} lang={lang} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <SequenceStrip title={t(lang, 'expectedSequence')} words={expectedSequence} lang={lang} />
        <SequenceStrip title={t(lang, 'detectedSequence')} words={detectedSequence} lang={lang} />
      </div>
      {timings.length > 0 && <Timeline title={t(lang, 'expectedTimeline')} timings={timings} duration={result.duration} color="expected" />}
      <Timeline title={t(lang, 'detectedTimeline')} timings={detectedTimeline} duration={result.duration} color="detected" />
      <section className="grid gap-3">
        <div className="flex items-center justify-between gap-3">
          <h3 className="font-title-md text-title-md text-on-surface">{t(lang, 'missedExpected')}</h3>
          <Badge tone={missedRows.length ? 'danger' : 'success'}>{missedRows.length}</Badge>
        </div>
        {missedRows.length ? (
          <div className="grid gap-3 max-h-[760px] overflow-auto custom-scrollbar pr-1">
            {missedRows.map((row, index) => (
              <LongReviewCard row={row} lang={lang} key={`${row.timing.label}-${index}`} />
            ))}
          </div>
        ) : (
          <p className="rounded-lg border border-primary/25 bg-primary/5 text-on-primary-fixed-variant px-4 py-3 font-body-md text-body-md m-0">{t(lang, 'noIssues')}</p>
        )}
      </section>
      <section className="grid gap-3">
        <div className="flex items-center justify-between gap-3">
          <h3 className="font-title-md text-title-md text-on-surface">{t(lang, 'allDetections')}</h3>
          <Badge tone="neutral">{tableRows.length}</Badge>
        </div>
        <LongDetectionTable rows={tableRows} lang={lang} />
      </section>
    </div>
  );
}

function OpenSetView({ result, lang }: { result: OpenSetResult; lang: Lang }) {
  const s = result.summary;
  return (
    <div className="grid gap-stack-md mt-stack-md">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        <Metric label="Known" value={s.known_tested} />
        <Metric label="Unknown" value={s.unknown_tested} />
        <Metric label="Candidates" value={result.candidate_words.length} />
        <Metric label="Balanced" value={pct(s.balanced_score)} tone={metricTone(s.balanced_score, 0.65)} />
        <Metric label="Open-set ACC" value={pct(s.open_set_acc)} tone={metricTone(s.open_set_acc, 0.65)} />
        <Metric label="KW-ACC" value={pct(s.keyword_acc)} tone={metricTone(s.keyword_acc, 0.65)} />
        <Metric label="Unknown reject" value={pct(s.unknown_reject_acc)} tone={metricTone(s.unknown_reject_acc, 0.65)} />
        <Metric label="FAR" value={pct(s.false_accept_rate)} tone={s.false_accept_rate > 0.3 ? 'warn' : 'good'} />
        <Metric label="False reject" value={pct(s.false_reject_rate)} tone={s.false_reject_rate > 0.3 ? 'warn' : 'good'} />
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
    <div className="grid gap-stack-md mt-stack-md">
      <h3 className="font-title-md text-title-md text-on-surface">Calibration</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-gutter">
        {options.map(([label, row]) => (
          <article key={label} className="rounded-xl border border-outline-variant bg-surface-container-lowest p-5 grid gap-3 hover-lift">
            <h4 className="font-title-md text-title-md text-on-surface">{label}</h4>
            <div className="grid grid-cols-2 gap-3">
              <Metric label="Balanced" value={pct(row.balanced_score)} />
              <Metric label="Threshold" value={num(row.threshold, 2)} />
              <Metric label="Guard" value={row.close_word_guard ? 'ON' : 'OFF'} />
              <Metric label="Per-class" value={row.use_per_class ? 'ON' : 'OFF'} />
              <Metric label="Margin" value={num(row.accept_margin, 4)} />
            </div>
            <button className={btnGhost} type="button" onClick={() => onApply(row)}>{t(lang, 'applySettings')}</button>
          </article>
        ))}
      </div>
      <div className="w-full overflow-x-auto border border-outline-variant rounded-xl bg-surface-container-lowest custom-scrollbar">
        <table className="w-full border-collapse min-w-[760px]">
          <thead>
            <tr className="bg-surface-container-low">
              {['#', 'Balanced', 'KW', 'Reject', 'FAR', 'Thr', 'Guard', 'Per-class', 'Margin'].map((head) => (
                <th key={head} className="text-left px-3.5 py-3 font-metric-label text-metric-label uppercase text-on-surface-variant border-b border-outline-variant">{head}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.slice(0, 20).map((row, index) => (
              <tr key={`${row.threshold}-${row.accept_margin}-${row.use_per_class}-${index}`} className="border-b border-outline-variant/40 hover:bg-surface-container-low font-metric-value text-metric-value text-on-surface">
                <td className="px-3.5 py-2.5 text-on-surface-variant">{index + 1}</td>
                <td className="px-3.5 py-2.5">{pct(row.balanced_score)}</td>
                <td className="px-3.5 py-2.5">{pct(row.keyword_acc)}</td>
                <td className="px-3.5 py-2.5">{pct(row.unknown_reject_acc)}</td>
                <td className="px-3.5 py-2.5">{pct(row.false_accept_rate)}</td>
                <td className="px-3.5 py-2.5">{num(row.threshold, 2)}</td>
                <td className="px-3.5 py-2.5">{row.close_word_guard ? 'ON' : 'OFF'}</td>
                <td className="px-3.5 py-2.5">{row.use_per_class ? 'ON' : 'OFF'}</td>
                <td className="px-3.5 py-2.5">{num(row.accept_margin, 4)}</td>
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
    <div className="rounded-lg border border-outline-variant bg-surface-container-lowest p-4">
      <strong className="block mb-2.5 font-title-md text-title-md text-on-surface">{title}</strong>
      <div className="flex flex-wrap gap-2">
        {words.map((word) => (
          <span key={word} className="inline-flex items-center rounded-full border border-outline-variant bg-surface-container-low px-2.5 py-1 font-metric-value text-metric-value text-on-surface-variant">{word}</span>
        ))}
      </div>
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
    <div className="grid gap-3">
      <h3 className="font-title-md text-title-md text-on-surface">{title}</h3>
      {items.slice(0, 20).map((item, index) => (
        <article key={`${item.word}-${index}`} className="grid gap-3 rounded-xl border border-outline-variant border-l-4 border-l-error bg-error/5 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <span className="font-metric-label text-metric-label uppercase text-on-surface-variant break-all">{item.file || item.path || item.word}</span>
              <h3 className="font-title-md text-title-md text-on-surface mt-1">{openSetCaseTitle(item)}</h3>
            </div>
            <Badge tone="danger">{item.status || 'CHECK'}</Badge>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">True word</span>
              <strong className="block mt-1 font-body-md text-body-md text-on-surface">{item.word || '-'}</strong>
            </div>
            <div>
              <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">Expected</span>
              <strong className="block mt-1 font-body-md text-body-md text-on-surface">{item.expected || '-'}</strong>
            </div>
            <div>
              <span className="font-metric-label text-metric-label uppercase text-on-surface-variant">Predicted</span>
              <strong className="block mt-1 font-body-md text-body-md text-on-surface">{item.predicted || '-'}</strong>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Metric label="L2" value={num(item.distance, 4)} />
            <Metric label="Thr" value={num(item.threshold, 3)} />
            <Metric label="Margin" value={num(item.margin, 4)} />
            <Metric label="Accept margin" value={num(item.accept_margin, 4)} />
          </div>
          <p className="rounded-lg border border-outline-variant bg-surface-container-high px-3 py-2 font-body-sm text-body-sm text-on-surface-variant m-0">{openSetCaseReason(item)}</p>
          <TopCandidates items={item.top_3} />
        </article>
      ))}
    </div>
  );
}

function ArtifactTable({ artifacts, lang }: { artifacts: ArtifactStatus; lang: Lang }) {
  return (
    <div className="w-full overflow-x-auto border border-outline-variant rounded-xl bg-surface-container-lowest custom-scrollbar">
      <table className="w-full border-collapse min-w-[760px]">
        <thead>
          <tr className="bg-surface-container-low">
            {['Artifact', 'Status', 'Role', 'Evidence', 'Notes'].map((head) => (
              <th key={head} className="text-left px-3.5 py-3 font-metric-label text-metric-label uppercase text-on-surface-variant border-b border-outline-variant">{head}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {artifacts.records.map((record) => (
            <tr key={record.id} className="border-b border-outline-variant/40 hover:bg-surface-container-low">
              <td className="px-3.5 py-3 font-body-md text-body-md text-on-surface font-medium">{record.label}</td>
              <td className="px-3.5 py-3"><Badge tone={record.exists ? 'success' : 'danger'}>{record.status}</Badge></td>
              <td className="px-3.5 py-3 font-body-sm text-body-sm text-on-surface-variant">{record.role}</td>
              <td className="px-3.5 py-3 font-metric-value text-metric-value text-on-surface-variant">{record.exists ? 'yes' : 'missing'}</td>
              <td className="px-3.5 py-3 font-body-sm text-body-sm text-on-surface-variant">{lang === 'vi' ? record.notes_vi : record.notes_en}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
