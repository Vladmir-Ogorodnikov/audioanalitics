import React, { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { RecordingDetail, UtteranceItem } from "../types";

interface Props {
  recording: RecordingDetail;
  initialSeekSec?: number | null;
  onBack: () => void;
  onReload: () => void;
  onNotify: (msg: string) => void;
  onError: (err: string) => void;
}

function fmtPrecise(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

export const PlayerView: React.FC<Props> = ({
  recording,
  initialSeekSec,
  onBack,
  onReload,
  onNotify,
  onError,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);

  // Редактирование отдельной реплики (FR-07)
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [editSpeaker, setEditSpeaker] = useState("");

  // Переименование спикера по всей записи (FR-04)
  const [renameTarget, setRenameTarget] = useState<string>("");
  const [renameValue, setRenameValue] = useState<string>("");

  useEffect(() => {
    if (initialSeekSec !== null && initialSeekSec !== undefined && audioRef.current) {
      audioRef.current.currentTime = initialSeekSec;
      setCurrentTime(initialSeekSec);
    }
  }, [initialSeekSec, recording.id]);

  const seekTo = (sec: number | null) => {
    if (sec === null || sec === undefined || !audioRef.current) return;
    const target = Math.max(0, sec);
    audioRef.current.currentTime = target;
    setCurrentTime(target);
    audioRef.current.play().catch(() => {
      // autoplay может требовать явного действия браузера
    });
  };

  const changeRate = (rate: number) => {
    setPlaybackRate(rate);
    if (audioRef.current) {
      audioRef.current.playbackRate = rate;
    }
  };

  const startEditUtterance = (u: UtteranceItem) => {
    setEditingId(u.id);
    setEditText(u.text);
    setEditSpeaker(u.speaker);
  };

  const saveEditUtterance = async (u: UtteranceItem) => {
    try {
      await api.updateUtterance(u.id, editText, editSpeaker);
      setEditingId(null);
      onNotify(`Реплика ${u.timecode} сохранена (FR-07).`);
      onReload();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRenameSpeaker = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renameTarget || !renameValue.trim()) return;
    try {
      const res = await api.renameSpeaker(recording.id, renameTarget, renameValue.trim());
      onNotify(
        `Метка «${renameTarget}» заменена на «${renameValue.trim()}» в ${res.updated_utterances} репликах (FR-04).`
      );
      setRenameTarget("");
      setRenameValue("");
      onReload();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const tr = recording.transcript;
  const corrections = tr?.meta?.term_corrections || [];

  return (
    <div>
      {/* Верхняя панель навигации и экспорта */}
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onBack}>
            ← К списку записей
          </button>
          <div>
            <h2 style={{ margin: 0, fontSize: "1.2rem" }}>{recording.title}</h2>
            <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
              Параметры FR-02: {recording.sample_rate} Гц, {recording.channels === 1 ? "моно" : "стерео"}, −16 LUFS ·{" "}
              Движок: {tr?.engine_used || "—"}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "0.5rem" }}>
          <a
            className="btn btn-primary btn-sm"
            href={`/api/v1/recordings/${recording.id}/export?format=txt`}
            download
          >
            ⬇ Экспорт TXT
          </a>
          <a
            className="btn btn-secondary btn-sm"
            href={`/api/v1/recordings/${recording.id}/export?format=json`}
            download
          >
            ⬇ Экспорт JSON
          </a>
        </div>
      </div>

      {/* Липкий аудиоплеер с поддержкой перемотки по словам (±0,5 с, FR-07) */}
      <div className="sticky-player">
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "1rem" }}>
          <audio
            ref={audioRef}
            controls
            preload="metadata"
            src={`/api/v1/recordings/${recording.id}/audio`}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            style={{ flex: "1 1 380px", height: "40px" }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.84rem" }}>
            <span style={{ color: "var(--text-muted)" }}>Скорость:</span>
            {[0.75, 1.0, 1.25, 1.5, 2.0].map((r) => (
              <button
                key={r}
                type="button"
                className={`btn btn-sm ${playbackRate === r ? "btn-primary" : "btn-secondary"}`}
                onClick={() => changeRate(r)}
              >
                {r}x
              </button>
            ))}
          </div>
          <span className="pill pill-blue" title="Текущая позиция воспроизведения">
            Позиция: {fmtPrecise(currentTime)}
          </span>
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.4rem" }}>
          Нажмите на <strong>любое слово</strong> в тексте ниже для мгновенного перехода плеера на его таймкод (точность ±0,1 с, норматив FR-07: ±0,5 с).
        </div>
      </div>

      {!tr ? (
        <div className="card">Транскрипт ещё формируется. Пожалуйста, дождитесь завершения очереди.</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 2.2fr) minmax(290px, 1fr)", gap: "1.15rem", alignItems: "start" }}>
          {/* Колонка реплик с кликабельными словами */}
          <div>
            {tr.utterances.map((u) => {
              const isActive = currentTime >= u.start_sec && currentTime <= u.end_sec + 0.15;
              const isEditing = editingId === u.id;

              return (
                <div key={u.id} className={`utterance-card ${isActive ? "active" : ""}`}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.45rem", gap: "0.5rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => seekTo(u.start_sec)}
                        title="Перейти к началу реплики"
                      >
                        ▶ {u.timecode}
                      </button>
                      <strong style={{ color: "var(--primary)" }}>{u.speaker}</strong>
                      {u.is_edited && <span className="pill pill-amber">отредактировано</span>}
                    </div>

                    {!isEditing && (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => startEditUtterance(u)}
                      >
                        Править реплику
                      </button>
                    )}
                  </div>

                  {isEditing ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem", marginTop: "0.5rem" }}>
                      <div className="form-row" style={{ marginBottom: 0 }}>
                        <div>
                          <label className="label">Спикер реплики</label>
                          <input
                            type="text"
                            className="input"
                            value={editSpeaker}
                            onChange={(e) => setEditSpeaker(e.target.value)}
                          />
                        </div>
                      </div>
                      <div>
                        <label className="label">Текст реплики (фильтр [НБЗ] применяется автоматически)</label>
                        <textarea
                          rows={3}
                          className="textarea"
                          style={{ fontFamily: "inherit", fontSize: "0.95rem" }}
                          value={editText}
                          onChange={(e) => setEditText(e.target.value)}
                        />
                      </div>
                      <div style={{ display: "flex", gap: "0.5rem" }}>
                        <button type="button" className="btn btn-primary btn-sm" onClick={() => saveEditUtterance(u)}>
                          Сохранить
                        </button>
                        <button type="button" className="btn btn-secondary btn-sm" onClick={() => setEditingId(null)}>
                          Отмена
                        </button>
                      </div>
                    </div>
                  ) : u.words && u.words.length > 0 ? (
                    <div>
                      {u.words.map((w, idx) => {
                        const wordActive =
                          w.start !== null &&
                          w.end !== null &&
                          currentTime >= w.start &&
                          currentTime <= w.end + 0.08;
                        const isNbz = w.text.includes("[НБЗ]");
                        return (
                          <button
                            key={`${u.id}-${idx}`}
                            type="button"
                            className={`word-chip ${wordActive ? "current" : ""} ${isNbz ? "nbz" : ""}`}
                            onClick={() => seekTo(w.start ?? u.start_sec)}
                            title={
                              w.start !== null
                                ? `Таймкод: ${fmtPrecise(w.start)} с`
                                : `Реплика: ${fmtPrecise(u.start_sec)} с`
                            }
                          >
                            {w.text}
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <p style={{ margin: 0 }}>{u.text}</p>
                  )}
                </div>
              );
            })}
          </div>

          {/* Правая панель: спикеры FR-04, статистика терминов FR-03 и чекпоинты FR-09 */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="card">
              <h3 style={{ fontSize: "1rem" }}>Управление спикерами (FR-04)</h3>
              <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: 0 }}>
                Метки диаризации автоматически приведены к виду «Спикер N». Вы можете переименовать любого участника:
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.75rem" }}>
                {tr.speakers.map((sp) => (
                  <button
                    key={sp}
                    type="button"
                    className={`btn btn-sm ${renameTarget === sp ? "btn-primary" : "btn-secondary"}`}
                    onClick={() => {
                      setRenameTarget(sp);
                      setRenameValue(sp);
                    }}
                  >
                    👤 {sp}
                  </button>
                ))}
              </div>
              {renameTarget && (
                <form onSubmit={handleRenameSpeaker} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  <label className="label">Новое имя для «{renameTarget}»:</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="Например: Доцент Долженкова М. Л."
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                  />
                  <div style={{ display: "flex", gap: "0.4rem" }}>
                    <button type="submit" className="btn btn-primary btn-sm">
                      Переименовать везде
                    </button>
                    <button type="button" className="btn btn-secondary btn-sm" onClick={() => setRenameTarget("")}>
                      Отмена
                    </button>
                  </div>
                </form>
              )}
            </div>

            <div className="card">
              <h3 style={{ fontSize: "1rem" }}>Лингвистический контроль (FR-03)</h3>
              <div style={{ fontSize: "0.86rem", display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                <div>Всего слов: <strong>{tr.word_count}</strong></div>
                <div>Маркировано [НБЗ]: <strong>{tr.censored_count}</strong></div>
                {tr.term_coverage?.terms_total !== undefined && (
                  <div>
                    Найдено терминов словаря:{" "}
                    <strong>
                      {tr.term_coverage.terms_found} из {tr.term_coverage.terms_total} (
                      {((tr.term_coverage.recall || 0) * 100).toFixed(0)}%)
                    </strong>
                  </div>
                )}
              </div>

              {corrections.length > 0 && (
                <div style={{ marginTop: "0.75rem" }}>
                  <div className="label">Автоисправленные опечатки терминов:</div>
                  <ul style={{ margin: "0.25rem 0 0", paddingLeft: "1.1rem", fontSize: "0.83rem" }}>
                    {corrections.map((c, idx) => (
                      <li key={idx}>
                        <code>{c.from}</code> → <strong>{c.to}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="card">
              <h3 style={{ fontSize: "1rem" }}>Журнал чекпоинтов задачи (FR-09)</h3>
              {recording.checkpoint_history.length === 0 ? (
                <div style={{ fontSize: "0.84rem", color: "var(--text-muted)" }}>Чекпоинты отсутствуют.</div>
              ) : (
                <ul style={{ margin: 0, paddingLeft: "1.1rem", fontSize: "0.82rem", display: "flex", flexDirection: "column", gap: "0.3rem" }}>
                  {recording.checkpoint_history.map((ck, idx) => (
                    <li key={idx}>
                      <strong>{ck.stage}</strong>: {ck.note}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
