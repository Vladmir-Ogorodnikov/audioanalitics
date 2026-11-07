import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { LibraryView } from "./components/LibraryView";
import { PlayerView } from "./components/PlayerView";
import type { RecordingDetail, RecordingSummary } from "./types";

export const App: React.FC = () => {
  const [recordings, setRecordings] = useState<RecordingSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRecording, setSelectedRecording] = useState<RecordingDetail | null>(null);
  const [seekTarget, setSeekTarget] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const notify = (msg: string) => {
    setErrorMsg(null);
    setToast(msg);
  };

  const reportError = (err: string) => {
    setErrorMsg(err);
  };

  const fetchRecordings = useCallback(async () => {
    try {
      const recs = await api.listRecordings(searchQuery);
      setRecordings(recs);
    } catch (err) {
      console.error(err);
    }
  }, [searchQuery]);

  useEffect(() => {
    fetchRecordings();
  }, [fetchRecordings]);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [toast]);

  const activeCount = recordings.filter((r) =>
    ["queued", "preprocessing", "transcribing", "analyzing"].includes(r.status)
  ).length;

  useEffect(() => {
    const intervalMs = activeCount > 0 ? 1500 : 6000;
    const timer = window.setInterval(() => {
      fetchRecordings();
      if (selectedRecording && selectedRecording.status !== "completed") {
        api.getRecording(selectedRecording.id).then(setSelectedRecording).catch(() => {});
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [activeCount, selectedRecording, fetchRecordings]);

  const openRecording = async (id: number, seekSec?: number) => {
    try {
      const detail = await api.getRecording(id);
      setSelectedRecording(detail);
      setSeekTarget(seekSec ?? null);
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
  };

  const reloadSelected = async () => {
    if (!selectedRecording) return;
    try {
      const detail = await api.getRecording(selectedRecording.id);
      setSelectedRecording(detail);
      fetchRecordings();
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="app-shell">
      <header className="header">
        <div className="header-inner">
          <button
            type="button"
            className="brand-btn"
            onClick={() => {
              setSelectedRecording(null);
              setSeekTarget(null);
            }}
          >
            <span className="brand-dot" aria-hidden="true" />
            <span className="brand-name">АудиоАналитик</span>
          </button>

          {activeCount > 0 && (
            <div className="queue-indicator" role="status">
              <span className="pulse-dot" aria-hidden="true" />
              <span>В обработке: {activeCount}</span>
            </div>
          )}
        </div>
      </header>

      <main className="main-content">
        {toast && (
          <div className="notice" role="status">
            <span>{toast}</span>
            <button
              type="button"
              className="notice-close"
              aria-label="Закрыть уведомление"
              onClick={() => setToast(null)}
            >
              ✕
            </button>
          </div>
        )}

        {errorMsg && (
          <div className="notice notice-error" role="alert">
            <span>{errorMsg}</span>
            <button
              type="button"
              className="notice-close"
              aria-label="Закрыть ошибку"
              onClick={() => setErrorMsg(null)}
            >
              ✕
            </button>
          </div>
        )}

        {selectedRecording ? (
          <PlayerView
            recording={selectedRecording}
            initialSeekSec={seekTarget}
            onBack={() => {
              setSelectedRecording(null);
              setSeekTarget(null);
            }}
            onReload={reloadSelected}
            onNotify={notify}
            onError={reportError}
          />
        ) : (
          <LibraryView
            recordings={recordings}
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            onOpenRecording={openRecording}
            onRefresh={fetchRecordings}
            onNotify={notify}
            onError={reportError}
          />
        )}
      </main>
    </div>
  );
};
