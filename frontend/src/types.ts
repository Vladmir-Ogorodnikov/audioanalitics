export interface WordItem {
  text: string;
  start: number | null;
  end: number | null;
  score: number | null;
}

export interface UtteranceItem {
  id: number;
  seq_num: number;
  start_sec: number;
  end_sec: number;
  timecode: string;
  speaker: string;
  original_speaker: string;
  text: string;
  is_edited: boolean;
  words: WordItem[];
}

export interface TranscriptData {
  id: number;
  language: string;
  model_name: string;
  engine_used: string;
  word_count: number;
  censored_count: number;
  term_coverage: {
    terms_total?: number;
    terms_found?: number;
    total_occurrences?: number;
    recall?: number;
    hits?: Record<string, number>;
  };
  speaker_map: Record<string, string>;
  meta: {
    term_corrections?: Array<{ from: string; to: string }>;
    preprocessed?: boolean;
    duration_sec?: number;
    [key: string]: unknown;
  };
  speakers: string[];
  utterances: UtteranceItem[];
}

export interface RecordingSummary {
  id: number;
  title: string;
  original_filename: string;
  file_format: string;
  file_size_bytes: number;
  duration_sec: number;
  sample_rate: number;
  channels: number;
  status: string;
  priority: number;
  progress_pct: number;
  stage_message: string;
  checkpoint_stage: string;
  retry_count: number;
  error_message: string | null;
  estimated_wait_sec: number | null;
  created_at: string;
  completed_at: string | null;
  processing_time_sec: number | null;
  word_count: number;
  search_snippet: string | null;
  matched_timecode_sec: number | null;
}

export interface CheckpointEntry {
  stage: string;
  completed_chunks: number;
  total_chunks: number;
  note: string;
  timestamp: string;
}

export interface RecordingDetail extends RecordingSummary {
  language: string;
  requested_speakers: number | null;
  diarize_enabled: boolean;
  low_vram: boolean;
  pipeline_mode: string;
  checkpoint_history: CheckpointEntry[];
  transcript: TranscriptData | null;
}

export interface SystemStatus {
  version: string;
  database_dialect: string;
  data_dir: string;
  disk_free_gb: number;
  network_isolation: {
    configured_host: string;
    localhost_only: boolean;
    external_telemetry: boolean;
    offline_ready: boolean;
    audit_retention_days: number;
  };
  ffmpeg: {
    available: boolean;
    path: string | null;
  };
  whisperx_installed: boolean;
  gpu: {
    available: boolean;
    device_name: string;
    vram_total_mb: number;
    vram_allocated_mb: number;
    vram_reserved_mb: number;
    profile_recommendation: string;
  };
  queue: {
    queued_count: number;
    active_count: number;
    queued_audio_hours: number;
    rt_factor: number;
    estimated_total_wait_sec: number;
    eta_by_recording_id: Record<string, number>;
  };
  stats: {
    total_recordings: number;
    completed_recordings: number;
    failed_recordings: number;
  };
}

export interface ModelRegistryItem {
  id: number;
  component: string;
  model_name: string;
  version: string;
  source_url: string;
  license: string;
  quantization: string;
  vram_mb: number;
  metrics: Record<string, unknown>;
  is_active: boolean;
  registered_at: string | null;
}

export interface LexiconResponse {
  terms_text: string;
  badwords_text: string;
  terms_count: number;
  active_profanity_roots_count: number;
  default_profanity_roots: string[];
  safe_word_prefixes: string[];
}

export interface MetricsEvalResponse {
  processed_hypothesis: string;
  word_level_preview: string[];
  censored_count: number;
  term_corrections: Array<{ from: string; to: string }>;
  wer: {
    rate: number;
    percent: number;
    passes_threshold_12pct: boolean;
    errors: number;
    total_words: number;
    substitutions: number;
    deletions: number;
    insertions: number;
  };
  cer: {
    rate: number;
    percent: number;
    passes_threshold_5pct: boolean;
    errors: number;
    total_chars: number;
  };
  terms: {
    precision: number;
    recall: number;
    f1: number;
    passes_recall_85pct: boolean;
    passes_f1_088: boolean;
    ref_occurrences: number;
    hyp_occurrences: number;
    true_positives: number;
    per_term: Record<string, { ref: number; hyp: number; tp: number }>;
  };
}

export interface AuditLogItem {
  id: number;
  timestamp: string | null;
  event_type: string;
  entity_type: string | null;
  entity_id: number | null;
  client_ip: string;
  message: string;
  details: Record<string, unknown>;
}
