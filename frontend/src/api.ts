import type {
  RecordingDetail,
  RecordingSummary,
  SummaryData,
  UtteranceItem,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `Ошибка ${res.status}`;
    try {
      const errBody = await res.json();
      if (errBody?.detail) {
        detail =
          typeof errBody.detail === "string"
            ? errBody.detail
            : JSON.stringify(errBody.detail);
      }
    } catch {
      // ignore json parse error
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  listRecordings: (q?: string) => {
    const params = new URLSearchParams();
    if (q && q.trim()) params.set("q", q.trim());
    const qs = params.toString();
    return requestJson<RecordingSummary[]>(`/api/v1/recordings${qs ? `?${qs}` : ""}`);
  },

  getRecording: (id: number) =>
    requestJson<RecordingDetail>(`/api/v1/recordings/${id}`),

  uploadRecording: (formData: FormData) =>
    requestJson<RecordingSummary>("/api/v1/recordings/upload", {
      method: "POST",
      body: formData,
    }),

  createDemoRecording: () =>
    requestJson<RecordingSummary>("/api/v1/recordings/demo?priority=3", {
      method: "POST",
    }),

  retryRecording: (id: number) =>
    requestJson<RecordingSummary>(`/api/v1/recordings/${id}/retry`, {
      method: "POST",
    }),

  deleteRecording: (id: number) =>
    requestJson<{ deleted: boolean; deleted_in_ms: number }>(
      `/api/v1/recordings/${id}`,
      { method: "DELETE" }
    ),

  renameSpeaker: (recordingId: number, oldName: string, newName: string) =>
    requestJson<{ updated_utterances: number }>(
      `/api/v1/recordings/${recordingId}/speakers`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      }
    ),

  updateUtterance: (utteranceId: number, text: string, speaker?: string) =>
    requestJson<UtteranceItem>(`/api/v1/utterances/${utteranceId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speaker, apply_censor: true }),
    }),

  summarizeRecording: (recordingId: number) =>
    requestJson<SummaryData>(`/api/v1/recordings/${recordingId}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }),
};
