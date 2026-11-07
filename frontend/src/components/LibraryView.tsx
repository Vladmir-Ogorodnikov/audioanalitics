import React, { useState } from "react";
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
    ? `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function statusBadge(status: string) {
  switch (status) {
    case "completed":
      return <span className="pill pill-green">Готово</span>;
    case "queued":
      return <span className="pill pill-amber">В очереди</span>;
    case "preprocessing":
      return <span className="pill pill-blue">Предобработка −16 LUFS</span>;
    case "transcribing":
      return <span className="pill pill-blue">Распознавание и диаризация</span>;
    case "failed":
      return <span className="pill pill-red">Сбой (есть чекпоинт)</span>;
    default:
      return <span className="pill pill-blue">{status}</span>;
  }
}

function priorityLabel(p: number): string {
  if (p >= 3) return "Высокий";
  if (p === 1) return "Низкий (фоновый)";
  return "Обычный";
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
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [priority, setPriority] = useState(2);
  const [speakers, setSpeakers] = useState("");
  const [diarize, setDiarize] = useState(true);
  const [lowVram, setLowVram] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      onError("Выберите аудиофайл формата MP3, WAV или FLAC (до 1 ГБ).");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (title.trim()) fd.append("title", title.trim());
      fd.append("priority", String(priority));
      if (speakers.trim()) fd.append("num_speakers", speakers.trim());
      fd.append("diarize", String(diarize));
      fd.append("low_vram", String(lowVram));
      const created = await api.uploadRecording(fd);
      setFile(null);
      setTitle("");
      onNotify(`Запись «${created.title}» принята и поставлена в очередь (#${created.id}).`);
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
      const created = await api.createDemoRecording();
      onNotify(
        `Демонстрационная лекция (#${created.id}) создана и запущена в обработку (ffmpeg −16 LUFS → чекпоинты → словари → «Спикер 1/2»).`
      );
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (rec: RecordingSummary) => {
    if (!window.confirm(`Удалить запись «${rec.title}» и все производные файлы с диска?`)) {
      return;
    }
    try {
      const res = await api.deleteRecording(rec.id);
      onNotify(
        `Запись «${rec.title}» и ${res.removed_files.length} файл(ов) полностью удалены за ${res.deleted_in_ms} мс (норматив FR-08: ≤ 20 000 мс).`
      );
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRetry = async (rec: RecordingSummary) => {
    try {
      await api.retryRecording(rec.id);
      onNotify(`Запись «${rec.title}» возвращена в очередь с чекпоинта «${rec.checkpoint_stage}» (FR-09).`);
      onRefresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
      <div className="grid-2">
        {/* Форма приёма файла FR-01 */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.5rem", marginBottom: "0.6rem" }}>
            <h2 style={{ margin: 0, fontSize: "1.12rem" }}>Загрузка аудиозаписи (FR-01, FR-10)</h2>
            <button
              type="button"
              className="btn btn-accent btn-sm"
              disabled={busy}
              onClick={handleDemo}
              title="Генерирует учебный WAV-файл 22 кГц и прогоняет весь конвейер Спринта 1"
            >
              ⚡ Создать демо-лекцию (1 клик)
            </button>
          </div>
          <p style={{ margin: "0 0 0.85rem", fontSize: "0.86rem", color: "var(--text-muted)" }}>
            Поддерживаются MP3, WAV, FLAC до 1 ГБ. Файл автоматически приводится к 16 кГц, моно, −16 LUFS (FR-02).
          </p>

          <form onSubmit={handleUpload}>
            <div style={{ marginBottom: "0.75rem" }}>
              <label className="label" htmlFor="audio-file-input">
                Аудиофайл (.mp3, .wav, .flac)
              </label>
              <input
                id="audio-file-input"
                type="file"
                accept=".mp3,.wav,.flac,audio/*"
                className="input"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </div>

            <div className="form-row">
              <div>
                <label className="label" htmlFor="rec-title-input">
                  Название записи (необязательно)
                </label>
                <input
                  id="rec-title-input"
                  type="text"
                  className="input"
                  placeholder="Например: Лекция по методам оптимизации"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>

              <div>
                <label className="label" htmlFor="rec-prio-select">
                  Приоритет в очереди (FR-10)
                </label>
                <select
                  id="rec-prio-select"
                  className="select"
                  value={priority}
                  onChange={(e) => setPriority(Number(e.target.value))}
                >
                  <option value={3}>Высокий (срочно)</option>
                  <option value={2}>Обычный</option>
                  <option value={1}>Низкий (фоновый пакет)</option>
                </select>
              </div>

              <div>
                <label className="label" htmlFor="rec-speakers-input">
                  Число спикеров (опц.)
                </label>
                <input
                  id="rec-speakers-input"
                  type="number"
                  min={1}
                  max={10}
                  className="input"
                  placeholder="Авто"
                  value={speakers}
                  onChange={(e) => setSpeakers(e.target.value)}
                />
              </div>
            </div>

            <div style={{ display: "flex", flexWrap: "wrap", gap: "1.2rem", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", gap: "1rem", fontSize: "0.88rem" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", cursor: "pointer" }}>
                  <input type="checkbox" checked={diarize} onChange={(e) => setDiarize(e.target.checked)} />
                  Диаризация «Спикер N» (FR-04)
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: "0.35rem", cursor: "pointer" }}>
                  <input type="checkbox" checked={lowVram} onChange={(e) => setLowVram(e.target.checked)} />
                  Профиль GPU 8 ГБ (CON-03)
                </label>
              </div>

              <button type="submit" className="btn btn-primary" disabled={busy || !file}>
                {busy ? "Загрузка..." : "Поставить в очередь"}
              </button>
            </div>
          </form>
        </div>

        {/* Полнотекстовый поиск FR-08 */}
        <div className="card" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
          <div>
            <h2 style={{ fontSize: "1.12rem", marginBottom: "0.45rem" }}>
              Полнотекстовый поиск по базе знаний (FR-08)
            </h2>
            <p style={{ margin: "0 0 0.85rem", fontSize: "0.86rem", color: "var(--text-muted)" }}>
              Мгновенный поиск цитат, терминов и лекций на русском языке с прямым переходом к таймкоду записи (≤ 3 действий, п. 4.8).
            </p>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <input
                type="search"
                className="input"
                aria-label="Полнотекстовый поиск по записям и транскриптам"
                placeholder="Введите термин или фразу: например, «регуляризация», «квадратов», «гомоскедастичность»..."
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
              />
              {searchQuery && (
                <button type="button" className="btn btn-secondary" onClick={() => onSearchChange("")}>
                  Сбросить
                </button>
              )}
            </div>
          </div>

          <div style={{ marginTop: "1rem", padding: "0.75rem", background: "var(--surface-alt)", borderRadius: "8px", fontSize: "0.84rem", color: "var(--text-muted)" }}>
            <strong>Что уже работает в Спринте 1:</strong> приём файлов до 1 ГБ, двухпроходный EBU R128 (−16 LUFS, 16 кГц, моно), чекпоинты каждые 5 мин с автовосстановлением (FR-09), диаризация «Спикер N» с переименованием, пословный плеер (±0,5 с), постобработка терминов, фильтр [НБЗ] без ложных срабатываний и журнал 152-ФЗ.
          </div>
        </div>
      </div>

      {/* Библиотека записей */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.12rem" }}>
            Библиотека записей ({recordings.length})
          </h2>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onRefresh}>
            Обновить список
          </button>
        </div>

        {recordings.length === 0 ? (
          <div style={{ textAlign: "center", padding: "2.2rem 1rem", color: "var(--text-muted)" }}>
            {searchQuery
              ? `По запросу «${searchQuery}» ничего не найдено.`
              : "Библиотека пока пуста. Загрузите аудиофайл выше или нажмите «⚡ Создать демо-лекцию (1 клик)» для мгновенной проверки!"}
          </div>
        ) : (
          <div className="rec-list">
            {recordings.map((rec) => (
              <div key={rec.id} className="rec-card">
                <div style={{ flex: "1 1 420px" }}>
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.55rem", marginBottom: "0.3rem" }}>
                    {statusBadge(rec.status)}
                    <strong style={{ fontSize: "1.02rem" }}>{rec.title}</strong>
                    <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                      #{rec.id} · {rec.file_format.toUpperCase()} · {(rec.file_size_bytes / (1024 * 1024)).toFixed(2)} МБ ·{" "}
                      {fmtDuration(rec.duration_sec)} · {rec.sample_rate} Гц
                    </span>
                  </div>

                  <div style={{ fontSize: "0.84rem", color: "var(--text-muted)", display: "flex", flexWrap: "wrap", gap: "0.9rem" }}>
                    <span>Приоритет: <strong>{priorityLabel(rec.priority)}</strong></span>
                    <span>Чекпоинт (FR-09): <code>{rec.checkpoint_stage}</code></span>
                    {rec.word_count > 0 && <span>Слов: <strong>{rec.word_count}</strong></span>}
                    {rec.processing_time_sec !== null && (
                      <span>Обработано за: <strong>{rec.processing_time_sec.toFixed(1)} с</strong></span>
                    )}
                    {rec.estimated_wait_sec !== null && rec.status !== "completed" && (
                      <span>Расчётное ожидание (ETA): <strong>~{Math.ceil(rec.estimated_wait_sec)} с</strong></span>
                    )}
                  </div>

                  {rec.status !== "completed" && (
                    <div style={{ marginTop: "0.55rem" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.2rem" }}>
                        <span>{rec.stage_message}</span>
                        <strong>{rec.progress_pct}%</strong>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${Math.min(100, rec.progress_pct)}%` }} />
                      </div>
                    </div>
                  )}

                  {rec.error_message && (
                    <div style={{ marginTop: "0.4rem", color: "var(--danger)", fontSize: "0.84rem" }}>
                      Ошибка: {rec.error_message}
                    </div>
                  )}

                  {rec.search_snippet && (
                    <div className="snippet-box">
                      <div><strong>Найденная цитата:</strong> {rec.search_snippet}</div>
                      {rec.matched_timecode_sec !== null && (
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          style={{ marginTop: "0.4rem" }}
                          onClick={() => onOpenRecording(rec.id, rec.matched_timecode_sec ?? 0)}
                        >
                          ▶ Открыть на таймкоде {fmtDuration(rec.matched_timecode_sec)}
                        </button>
                      )}
                    </div>
                  )}
                </div>

                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
                  {rec.status === "completed" && (
                    <>
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={() => onOpenRecording(rec.id)}
                      >
                        Открыть транскрипт и плеер
                      </button>
                      <a
                        className="btn btn-secondary btn-sm"
                        href={`/api/v1/recordings/${rec.id}/export?format=txt`}
                        download
                      >
                        Экспорт TXT
                      </a>
                    </>
                  )}
                  {rec.status === "failed" && (
                    <button
                      type="button"
                      className="btn btn-accent btn-sm"
                      onClick={() => handleRetry(rec)}
                    >
                      Возобновить с чекпоинта
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDelete(rec)}
                  >
                    Удалить
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
