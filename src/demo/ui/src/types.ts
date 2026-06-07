export type Lang = 'vi' | 'en';

export type ModelMetric = {
  label: string;
  value: string;
};

export type ModelProfile = {
  id: string;
  label: string;
  short_label?: string;
  description?: string;
  description_vi?: string;
  description_en?: string;
  notes?: string;
  notes_vi?: string;
  notes_en?: string;
  checkpoint_name?: string;
  checkpoint_path?: string;
  model_family?: string;
  feature_type?: string;
  threshold_hint?: number | null;
  exists: boolean;
  metrics?: ModelMetric[];
};

export type ModelProfilesResponse = {
  active: string;
  can_rebuild_on_switch: boolean;
  profiles: ModelProfile[];
};

export type EnrollmentStatus = {
  enrolled: Record<string, {
    count: number;
    threshold?: number;
    profile?: string;
    qualities?: unknown[];
  }>;
  total: number;
  profile_version: number;
  streaming: string;
  can_rebuild_on_switch: boolean;
};

export type TopCandidate = {
  word: string;
  dist: number;
};

export type DetectionSettings = {
  threshold: number;
  use_per_class: boolean;
  close_word_guard: boolean;
  accept_margin: number;
  engine: string;
  model_profile?: string;
  model_label?: string;
};

export type DetectResult = {
  keyword: string;
  best_label?: string;
  detected: boolean;
  distance: number;
  threshold: number;
  margin?: number;
  confidence?: number;
  second_label?: string | null;
  top_3?: TopCandidate[];
  settings?: DetectionSettings;
};

export type LongSegment = DetectResult & {
  t0: number;
  t1: number;
  accept_margin?: number;
  close_word_guard?: boolean;
};

export type LongResult = {
  duration: number;
  segments: number;
  results: LongSegment[];
  sequence: string[];
  engine: string;
  settings?: DetectionSettings;
};

export type OpenSetSummary = {
  known_tested: number;
  unknown_tested: number;
  open_set_acc: number;
  keyword_acc: number;
  unknown_reject_acc: number;
  false_accept_rate: number;
  false_reject_rate: number;
  known_misses: number;
  balanced_score: number;
};

export type OpenSetCase = {
  kind?: 'known' | 'unknown';
  word: string;
  expected?: string;
  predicted: string;
  status?: 'correct' | 'false_accept' | 'false_reject' | 'wrong_keyword' | string;
  best_label?: string;
  second_label?: string | null;
  distance?: number;
  threshold?: number;
  margin?: number;
  accept_margin?: number;
  top_3?: TopCandidate[];
  file?: string;
  path?: string;
};

export type OpenSetResult = {
  settings: DetectionSettings;
  preset: string;
  known_words: string[];
  unknown_words: string[];
  heldout_words: string[];
  candidate_words: string[];
  summary: OpenSetSummary;
  false_accepts: OpenSetCase[];
  known_misses: OpenSetCase[];
  missing_known_words?: string[];
  missing_unknown_words?: string[];
};

export type CalibrationRow = OpenSetSummary & {
  threshold: number;
  use_per_class: boolean;
  close_word_guard: boolean;
  accept_margin: number;
};

export type CalibrationResult = {
  settings: Record<string, unknown>;
  preset: string;
  known_words: string[];
  unknown_words: string[];
  heldout_words: string[];
  candidate_words: string[];
  best_balanced: CalibrationRow;
  best_open_set: CalibrationRow;
  best_keyword: CalibrationRow;
  rows: CalibrationRow[];
};

export type PresetResponse = {
  presets: Record<string, string>;
  gsc_words: string[];
  open_set_presets: Record<string, {
    id: string;
    label: string;
    known_words: string[];
    unknown_words: string[];
    heldout_words: string[];
  }>;
};

export type ArtifactRecord = {
  id: string;
  label: string;
  status: string;
  role: string;
  path: string;
  exists: boolean;
  evidence_type: string;
  metrics: Record<string, number | null>;
  notes_vi: string;
  notes_en: string;
};

export type ArtifactStatus = {
  generated_from: string;
  records: ArtifactRecord[];
};
