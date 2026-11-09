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
  speakers: string[];
  utterances: UtteranceItem[];
}

export interface KeyConceptItem {
  term: string;
  definition: string;
}

export interface ThesisItem {
  id: number;
  seq_num: number;
  text: string;
  timecode_sec: number;
  timecode_formatted: string;
  confidence: number;
  needs_review: boolean;
}

export interface SummaryData {
  id: number;
  model_name: string;
  goal: string;
  key_concepts: KeyConceptItem[];
  theses: ThesisItem[];
  tasks_decisions: string[];
  self_check_questions: string[];
  entities: Record<string, string[]>;
  updated_at: string | null;
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

export interface RecordingDetail extends RecordingSummary {
  language: string;
  requested_speakers: number | null;
  diarize_enabled: boolean;
  low_vram: boolean;
  speed_preset?: string;
  transcript: TranscriptData | null;
  summary: SummaryData | null;
}
