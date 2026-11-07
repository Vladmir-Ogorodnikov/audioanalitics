import React, { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { AuditLogView } from "./components/AuditLogView";
import { LexiconMetricsView } from "./components/LexiconMetricsView";
import { LibraryView } from "./components/LibraryView";
import { PlayerView } from "./components/PlayerView";
import { QueueModelsView } from "./components/QueueModelsView";
import type {
  ModelRegistryItem,
  RecordingDetail,
  RecordingSummary,
  SystemStatus,
} from "./types";

type ActiveTab = "library" | "lexicon" | "system" | "audit";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ActiveTab>("library");
  const [recordings, setRecordings] = useState<RecordingSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRecording, setSelectedRecording] = useState<RecordingDetail | null>(null);
  const [seekTarget, setSeekTarget] = useState<number | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [toast, setToast] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const notify = (msg: string) => {
    setErrorMsg(null);
    setToast(msg);
  };

  const reportError = (err: string) => {
    setErrorMsg(err);
  };

  const fetchAll = useCallback(async () => {
    try {
      const [recs, sys] = await Promise.all([
        api.listRecordings(searchQuery),
        api.getSystemStatus(),
      ]);
      setRecordings(recs);
      setSystemStatus(sys);
    } catch (err) {
      // не блокируем интерфейс при кратковременной перезагрузке
      console.error(err);
    }
  }, [searchQuery]);

  useEffect(() => {
    fetchAll();
    api.getModels().then(setModels).catch(() => {});
  }, [fetchAll]);

  // Автообновление прогресса очереди каждые 2 секунды, пока есть активные задачи
  useEffect(() => {
    const hasPending = recordings.some((r) =>
      ["queued", "preprocessing", "transcribing", "analyzing"].includes(r.status)
    );
    const intervalMs = hasPending ? 1500 : 6000;
    const timer = window.setInterval(() => {
      fetchAll();
      if (selectedRecording && selectedRecording.status !== "completed") {
        api.getRecording(selectedRecording.id).then(setSelectedRecording).catch(() => {});
      }
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [recordings, selectedRecording, fetchAll]);

  const openRecording = async (id: number, seekSec?: number) => {
    try {
      const detail = await api.getRecording(id);
      setSelectedRecording(detail);
      setSeekTarget(seekSec ?? null);
      setActiveTab("library");
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
  };

  const reloadSelected = async () => {
    if (!selectedRecording) return;
    try {
      const detail = await api.getRecording(selectedRecording.id);
      setSelectedRecording(detail);
      fetchAll();
    } catch (err) {
      reportError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-badge">ААИИ-2026</span>
          <div>
            <h1 className="brand-title">АС «АудиоАналитик ИИ»</h1>
            <p className="brand-sub">Спринт 1 · Локальная транскрибация, диаризация и база знаний (152-ФЗ)</p>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Разделы системы">
          <button
            type="button"
            className={`nav-tab ${activeTab === "library" ? "active" : ""}`}
            onClick={() => {
              setSelectedRecording(null);
              setActiveTab("library");
            }}
          >
            Библиотека и плеер
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "lexicon" ? "active" : ""}`}
            onClick={() => setActiveTab("lexicon")}
          >
            Словари и метрики (Спринт 1)
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "system" ? "active" : ""}`}
            onClick={() => setActiveTab("system")}
          >
            Очередь, GPU и модели
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === "audit" ? "active" : ""}`}
            onClick={() => setActiveTab("audit")}
          >
            Журнал 152-ФЗ
          </button>
        </nav>

        <div className="status-pills">
          {systemStatus && (
            <>
              <span className="pill pill-dark" title="Активные и ожидающие задачи в очереди">
                Очередь: {systemStatus.queue.queued_count + systemStatus.queue.active_count}
              </span>
              <span className="pill pill-dark" title="Локальное хранилище и отсутствие телеметрии">
                🔒 Локальный режим
              </span>
            </>
          )}
        </div>
      </header>

      <main className="container">
        {toast && (
          <div className="banner-toast" role="status">
            <span>{toast}</span>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setToast(null)}>
              ✕
            </button>
          </div>
        )}

        {errorMsg && (
          <div className="banner-error" role="alert">
            <span>{errorMsg}</span>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setErrorMsg(null)}>
              ✕
            </button>
          </div>
        )}

        {activeTab === "library" &&
          (selectedRecording ? (
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
              onRefresh={fetchAll}
              onNotify={notify}
              onError={reportError}
            />
          ))}

        {activeTab === "lexicon" && (
          <LexiconMetricsView onNotify={notify} onError={reportError} />
        )}

        {activeTab === "system" && (
          <QueueModelsView status={systemStatus} models={models} onRefresh={fetchAll} />
        )}

        {activeTab === "audit" && <AuditLogView onError={reportError} />}
      </main>
    </div>
  );
};
