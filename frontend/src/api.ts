import type {
  AuditLogItem,
  LexiconResponse,
  MetricsEvalResponse,
  ModelRegistryItem,
  RecordingDetail,
  RecordingSummary,
  SystemStatus,
  UtteranceItem,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      if (errBody?.detail) {
        detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  getSystemStatus: () => requestJson<SystemStatus>("/api/v1/system/status"),

  getModels: () => requestJson<ModelRegistryItem[]>("/api/v1/models"),

  listRecordings: (q?: string, status?: string) => {
    const params = new URLSearchParams();
    if (q && q.trim()) params.set("q", q.trim());
    if (status && status !== "all") params.set("status", status);
    const qs = params.toString();
    return requestJson<RecordingSummary[]>(`/api/v1/recordings${qs ? `?${qs}` : ""}`);
  },

  getRecording: (id: number) => requestJson<RecordingDetail>(`/api/v1/recordings/${id}`),

  uploadRecording: (formData: FormData) =>
    requestJson<RecordingSummary>("/api/v1/recordings/upload", {
      method: "POST",
      body: formData,
    }),

  createDemoRecording: (title?: string, priority = 3) => {
    const params = new URLSearchParams({ priority: String(priority) });
    if (title) params.set("title", title);
    return requestJson<RecordingSummary>(`/api/v1/recordings/demo?${params.toString()}`, {
      method: "POST",
    });
  },

  retryRecording: (id: number) =>
    requestJson<RecordingSummary>(`/api/v1/recordings/${id}/retry`, {
      method: "POST",
    }),

  deleteRecording: (id: number) =>
    requestJson<{ deleted: boolean; deleted_in_ms: number; removed_files: string[] }>(
      `/api/v1/recordings/${id}`,
      { method: "DELETE" }
    ),

  renameSpeaker: (recordingId: number, oldName: string, newName: string) =>
    requestJson<{ updated_utterances: number }>(`/api/v1/recordings/${recordingId}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_name: oldName, new_name: newName }),
    }),

  updateUtterance: (utteranceId: number, text: string, speaker?: string) =>
    requestJson<UtteranceItem>(`/api/v1/utterances/${utteranceId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speaker, apply_censor: true }),
    }),

  getLexicon: () => requestJson<LexiconResponse>("/api/v1/lexicon"),

  saveLexicon: (termsText: string, badwordsText: string) =>
    requestJson<{ saved: boolean; terms_count: number; active_profanity_roots_count: number }>(
      "/api/v1/lexicon",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ terms_text: termsText, badwords_text: badwordsText }),
      }
    ),

  evaluateMetrics: (payload: {
    reference: string;
    hypothesis: string;
    apply_postprocessing_preview: boolean;
    case_sensitive: boolean;
  }) =>
    requestJson<MetricsEvalResponse>("/api/v1/metrics/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getAuditLogs: (eventType?: string) => {
    const params = new URLSearchParams({ limit: "300" });
    if (eventType && eventType !== "all") params.set("event_type", eventType);
    return requestJson<AuditLogItem[]>(`/api/v1/audit-logs?${params.toString()}`);
  },
};
