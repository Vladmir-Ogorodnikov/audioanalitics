import React, { useRef, useState } from "react";
import { api } from "../api";
import type { RecordingSummary } from "../types";

interface Props {
  recordings: RecordingSummary[];
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onOpenRecording: (id: number, seekSec?: number) => void;
  onRefresh: () => void;
  onNotify: (msg: string) => void;
  onError: (err: string) => void;
}

function fmtDuration(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("ru-RU", {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function renderStatus(rec: RecordingSummary) {
  if (rec.status === "completed") {
    return <span className="status-tag status-ready">Готово</span>;
  }
  if (rec.status === "queued") {
    return <span className="status-tag status-queued">В очереди</span>;
  }
  if (rec.status === "failed") {
    return <span className="status-tag status-error">Ошибка</span>;
  }
  return (
    <span className="status-tag status-progress">
      Обработка {Math.round(rec.progress_pct)}%
    </span>
  );
}

export const LibraryView: React.FC<Props> = ({
  recordings,
  searchQuery,
  onSearchChange,
  onOpenRecording,
  onRefresh,
  onNotify,
  onError,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [showOptions, setShowOptions] = useState(false);
  const [speakers, setSpeakers] = useState("");
  const [diarize, setDiarize] = useState(true);
  const [priority, setPriority] = useState(2);
  const [speedPreset, setSpeedPreset] = useState("auto");
  const [busy, setBusy] = useState(false);

  const pickFile = (picked: File | null) => {
    if (!picked) return;
    setFile(picked);
    if (!title) {
      setTitle(picked.name.replace(/\.[^.]+$/, ""));
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (title.trim()) fd.append("title", title.trim());
      fd.append("priority", String(priority));
      if (speakers.trim()) fd.append("num_speakers", speakers.trim());
      fd.append("diarize", String(diarize));
      fd.append("speed_preset", speedPreset);
      await api.uploadRecording(fd);
      setFile(null);
      setTitle("");
      setShowOptions(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      onNotify("Запись добавлена в обработку.");
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleDemo = async () => {
    setBusy(true);
    try {
      await api.createDemoRecording();
      onNotify("Демо-лекция добавлена в обработку.");
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, rec: RecordingSummary) => {
    e.stopPropagation();
    if (!window.confirm(`Удалить запись «${rec.title}»?`)) return;
    try {
      await api.deleteRecording(rec.id);
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRetry = async (e: React.MouseEvent, rec: RecordingSummary) => {
    e.stopPropagation();
    try {
      await api.retryRecording(rec.id);
      onNotify("Обработка возобновлена.");
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div>
      {/* Загрузка аудиозаписи */}
      <div
        className={`upload-box ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pickFile(e.dataTransfer.files?.[0] || null);
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".mp3,.wav,.flac,audio/*"
          style={{ display: "none" }}
          onChange={(e) => pickFile(e.target.files?.[0] || null)}
        />

        <div className="upload-row">
          <div>
            <h2 className="upload-title">Новая аудиозапись</h2>
            <p className="upload-hint">
              Перетащите файл сюда или выберите на компьютере · MP3, WAV, FLAC до 1 ГБ
            </p>
          </div>

          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              disabled={busy}
              onClick={handleDemo}
            >
              Демо-запись
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              Выбрать файл
            </button>
          </div>
        </div>

        {file && (
          <form onSubmit={handleUpload} className="selected-file-bar">
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.65rem", alignItems: "center" }}>
              <input
                type="text"
                className="input"
                style={{ flex: "1 1 240px" }}
                placeholder="Название записи"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
              <button
                type="button"
                className="btn btn-text btn-sm"
                onClick={() => setShowOptions((v) => !v)}
              >
                {showOptions ? "Скрыть параметры" : "Параметры"}
              </button>
              <button
                type="button"
                className="btn btn-text btn-sm"
                onClick={() => {
                  setFile(null);
                  setTitle("");
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
              >
                Отмена
              </button>
              <button type="submit" className="btn btn-primary btn-sm" disabled={busy}>
                {busy ? "Загрузка…" : "Расшифровать"}
              </button>
            </div>

            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Файл: {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} МБ)
            </div>

            {showOptions && (
              <div className="options-grid">
                <div>
                  <label className="field-label">Скорость и модель</label>
                  <select
                    className="select"
                    value={speedPreset}
                    onChange={(e) => setSpeedPreset(e.target.value)}
                  >
                    <option value="auto">Авто (быстро и точно)</option>
                    <option value="fast">Максимальная скорость (small)</option>
                    <option value="accurate">Максимум качества (large-v3)</option>
                  </select>
                </div>
                <div>
                  <label className="field-label">Число спикеров</label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    className="input"
                    placeholder="Автоматически"
                    value={speakers}
                    onChange={(e) => setSpeakers(e.target.value)}
                  />
                </div>
                <div>
                  <label className="field-label">Приоритет</label>
                  <select
                    className="select"
                    value={priority}
                    onChange={(e) => setPriority(Number(e.target.value))}
                  >
                    <option value={3}>Высокий</option>
                    <option value={2}>Обычный</option>
                    <option value={1}>Низкий</option>
                  </select>
                </div>
                <div style={{ display: "flex", alignItems: "flex-end", paddingBottom: "0.45rem" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.86rem", cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={diarize}
                      onChange={(e) => setDiarize(e.target.checked)}
                    />
                    Разделять по спикерам
                  </label>
                </div>
              </div>
            )}
          </form>
        )}
      </div>

      {/* Поиск */}
      <div className="search-wrap">
        <span className="search-icon" aria-hidden="true">⌕</span>
        <input
          type="search"
          className="search-input"
          aria-label="Поиск по записям"
          placeholder="Поиск по названиям и тексту расшифровок…"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
        {searchQuery && (
          <button
            type="button"
            className="search-clear"
            aria-label="Очистить поиск"
            onClick={() => onSearchChange("")}
          >
            ✕
          </button>
        )}
      </div>

      {/* Библиотека записей */}
      <div className="section-header">
        <h3 className="section-title">
          Записи {recordings.length > 0 ? `(${recordings.length})` : ""}
        </h3>
      </div>

      <div className="library-surface">
        {recordings.length === 0 ? (
          <div className="empty-state">
            {searchQuery
              ? `По запросу «${searchQuery}» ничего не найдено.`
              : "Записей пока нет. Загрузите аудиофайл или создайте демо-запись."}
          </div>
        ) : (
          recordings.map((rec) => {
            const isReady = rec.status === "completed";
            const isFailed = rec.status === "failed";
            const isRunning = !isReady && !isFailed;

            return (
              <div
                key={rec.id}
                className={`rec-item ${isReady ? "clickable" : ""}`}
                onClick={() => {
                  if (isReady) onOpenRecording(rec.id);
                }}
              >
                <div className="rec-main">
                  <div className="rec-title-row">
                    <span className="rec-name">{rec.title}</span>
                    {renderStatus(rec)}
                  </div>

                  <div className="rec-meta">
                    <span>{fmtDuration(rec.duration_sec)}</span>
                    {rec.word_count > 0 && <span>· {rec.word_count} слов</span>}
                    <span>· {rec.file_format.toUpperCase()}</span>
                    {rec.created_at && <span>· {fmtDate(rec.created_at)}</span>}
                    {isRunning && rec.estimated_wait_sec !== null && (
                      <span>· осталось ~{Math.ceil(rec.estimated_wait_sec)} с</span>
                    )}
                  </div>

                  {isRunning && (
                    <>
                      <div className="thin-progress" title={rec.stage_message}>
                        <div
                          className="thin-progress-bar"
                          style={{ width: `${Math.max(4, Math.min(100, rec.progress_pct))}%` }}
                        />
                      </div>
                      {rec.stage_message && (
                        <div style={{ marginTop: "0.28rem", fontSize: "0.78rem", color: "var(--text-muted)" }}>
                          {rec.stage_message}
                        </div>
                      )}
                    </>
                  )}

                  {isFailed && rec.error_message && (
                    <div style={{ marginTop: "0.35rem", fontSize: "0.82rem", color: "var(--danger)" }}>
                      {rec.error_message}
                    </div>
                  )}

                  {rec.search_snippet && (
                    <div
                      className="quote-match"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isReady) {
                          onOpenRecording(rec.id, rec.matched_timecode_sec ?? 0);
                        }
                      }}
                    >
                      <span>{rec.search_snippet}</span>
                      {rec.matched_timecode_sec !== null && (
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenRecording(rec.id, rec.matched_timecode_sec ?? 0);
                          }}
                        >
                          ▶ {fmtDuration(rec.matched_timecode_sec)}
                        </button>
                      )}
                    </div>
                  )}
                </div>

                <div className="rec-actions" onClick={(e) => e.stopPropagation()}>
                  {isReady && (
                    <>
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm"
                        onClick={() => onOpenRecording(rec.id)}
                      >
                        Открыть
                      </button>
                      <a
                        className="btn btn-text btn-sm"
                        href={`/api/v1/recordings/${rec.id}/export?format=txt`}
                        download
                        title="Скачать TXT"
                      >
                        TXT
                      </a>
                    </>
                  )}
                  {isFailed && (
                    <button
                      type="button"
                      className="btn btn-ghost btn-sm"
                      onClick={(e) => handleRetry(e, rec)}
                    >
                      Повторить
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-danger-text btn-sm"
                    onClick={(e) => handleDelete(e, rec)}
                    title="Удалить запись"
                    aria-label="Удалить запись"
                  >
                    Удалить
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
