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

const RATES = [1, 1.25, 1.5, 2, 0.75];

function fmtClock(sec: number): string {
  const s = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
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
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(recording.duration_sec || 0);
  const [rate, setRate] = useState(1);

  // Редактирование реплики
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [editSpeaker, setEditSpeaker] = useState("");

  // Быстрое переименование спикера в одну строку
  const [renamingSpeaker, setRenamingSpeaker] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState("");

  useEffect(() => {
    setDuration(recording.duration_sec || 0);
  }, [recording.duration_sec]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el || initialSeekSec === null || initialSeekSec === undefined) return;
    const applySeek = () => {
      el.currentTime = Math.max(0, initialSeekSec);
      setCurrentTime(el.currentTime);
    };
    if (el.readyState >= 1) {
      applySeek();
    } else {
      el.addEventListener("loadedmetadata", applySeek, { once: true });
    }
  }, [initialSeekSec, recording.id]);

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(() => {});
    } else {
      el.pause();
    }
  };

  const seekTo = (sec: number | null, autoPlay = true) => {
    const el = audioRef.current;
    if (sec === null || sec === undefined || !el) return;
    const next = Math.max(0, sec);
    el.currentTime = next;
    setCurrentTime(next);
    if (autoPlay) {
      el.play().catch(() => {});
    }
  };

  const cycleRate = () => {
    const idx = RATES.indexOf(rate);
    const next = RATES[(idx + 1) % RATES.length];
    setRate(next);
    if (audioRef.current) {
      audioRef.current.playbackRate = next;
    }
  };

  const startEdit = (u: UtteranceItem) => {
    setEditingId(u.id);
    setEditText(u.text);
    setEditSpeaker(u.speaker);
  };

  const saveEdit = async (u: UtteranceItem) => {
    try {
      await api.updateUtterance(u.id, editText, editSpeaker);
      setEditingId(null);
      onReload();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const submitSpeakerRename = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renamingSpeaker || !renameVal.trim() || renameVal.trim() === renamingSpeaker) {
      setRenamingSpeaker(null);
      return;
    }
    try {
      await api.renameSpeaker(recording.id, renamingSpeaker, renameVal.trim());
      setRenamingSpeaker(null);
      onNotify("Имя спикера обновлено.");
      onReload();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const tr = recording.transcript;
  const totalSec = Math.max(duration, recording.duration_sec || 0, 0.1);

  return (
    <div>
      {/* Шапка записи */}
      <div className="player-topbar">
        <div>
          <button type="button" className="btn btn-text btn-sm" onClick={onBack} style={{ marginLeft: "-0.55rem" }}>
            ← Все записи
          </button>
          <h1 className="player-heading">{recording.title}</h1>
          <div className="player-submeta">
            {fmtClock(totalSec)}
            {tr ? ` · ${tr.word_count} слов` : ""}
            {tr && tr.speakers.length > 0 ? ` · спикеров: ${tr.speakers.length}` : ""}
          </div>
        </div>

        <div style={{ display: "flex", gap: "0.45rem" }}>
          <a
            className="btn btn-primary btn-sm"
            href={`/api/v1/recordings/${recording.id}/export?format=txt`}
            download
          >
            Скачать TXT
          </a>
          <a
            className="btn btn-ghost btn-sm"
            href={`/api/v1/recordings/${recording.id}/export?format=json`}
            download
          >
            JSON
          </a>
        </div>
      </div>

      {/* Минималистичный аудиоплеер */}
      <div className="player-bar">
        <audio
          ref={audioRef}
          preload="metadata"
          src={`/api/v1/recordings/${recording.id}/audio`}
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => {
            if (e.currentTarget.duration && Number.isFinite(e.currentTarget.duration)) {
              setDuration(e.currentTarget.duration);
            }
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />

        <div className="player-controls">
          <button
            type="button"
            className="play-btn"
            onClick={togglePlay}
            aria-label={playing ? "Пауза" : "Воспроизвести"}
          >
            {playing ? "❚❚" : "▶"}
          </button>

          <span className="time-label">{fmtClock(currentTime)}</span>

          <input
            type="range"
            className="scrubber"
            min={0}
            max={totalSec}
            step={0.1}
            value={Math.min(currentTime, totalSec)}
            aria-label="Позиция воспроизведения"
            onChange={(e) => seekTo(Number(e.target.value), false)}
          />

          <span className="time-label">{fmtClock(totalSec)}</span>

          <button
            type="button"
            className="speed-btn"
            onClick={cycleRate}
            title="Скорость воспроизведения"
          >
            {rate}x
          </button>
        </div>
      </div>

      {/* Компактная полоска переименования спикеров */}
      {tr && tr.speakers.length > 0 && (
        <div className="speakers-strip">
          {tr.speakers.map((sp) =>
            renamingSpeaker === sp ? (
              <form key={sp} onSubmit={submitSpeakerRename} className="speaker-rename-inline">
                <input
                  type="text"
                  autoFocus
                  value={renameVal}
                  placeholder="Имя спикера"
                  onChange={(e) => setRenameVal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") setRenamingSpeaker(null);
                  }}
                />
                <button type="submit" className="btn btn-primary btn-sm" style={{ borderRadius: "999px", padding: "0.18rem 0.6rem" }}>
                  ОК
                </button>
                <button
                  type="button"
                  className="btn btn-text btn-sm"
                  style={{ borderRadius: "999px", padding: "0.18rem 0.45rem" }}
                  onClick={() => setRenamingSpeaker(null)}
                >
                  ✕
                </button>
              </form>
            ) : (
              <button
                key={sp}
                type="button"
                className="speaker-pill"
                title="Нажмите, чтобы переименовать спикера"
                onClick={() => {
                  setRenamingSpeaker(sp);
                  setRenameVal(sp);
                }}
              >
                <span>{sp}</span>
                <span style={{ opacity: 0.45 }}>✎</span>
              </button>
            )
          )}
        </div>
      )}

      {/* Поток реплик транскрипта */}
      {!tr ? (
        <div className="library-surface">
          <div className="empty-state">Расшифровка ещё выполняется…</div>
        </div>
      ) : (
        <div className="transcript-stream">
          {tr.utterances.map((u) => {
            const isActive = currentTime >= u.start_sec && currentTime <= u.end_sec + 0.15;
            const isEditing = editingId === u.id;

            return (
              <div key={u.id} className={`turn-block ${isActive ? "active" : ""}`}>
                <div className="turn-header">
                  <div className="turn-meta">
                    <button
                      type="button"
                      className="timecode-btn"
                      onClick={() => seekTo(u.start_sec)}
                      title="Перейти к началу реплики"
                    >
                      {fmtClock(u.start_sec)}
                    </button>
                    <span className="turn-speaker">{u.speaker}</span>
                    {u.is_edited && <span className="edited-dot">· изменено</span>}
                  </div>

                  {!isEditing && (
                    <button
                      type="button"
                      className="btn btn-text btn-sm turn-edit-trigger"
                      onClick={() => startEdit(u)}
                    >
                      Изменить
                    </button>
                  )}
                </div>

                {isEditing ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem", marginTop: "0.35rem" }}>
                    <input
                      type="text"
                      className="input"
                      style={{ maxWidth: "240px" }}
                      placeholder="Спикер"
                      value={editSpeaker}
                      onChange={(e) => setEditSpeaker(e.target.value)}
                    />
                    <textarea
                      rows={3}
                      className="textarea"
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                    />
                    <div style={{ display: "flex", gap: "0.45rem" }}>
                      <button type="button" className="btn btn-primary btn-sm" onClick={() => saveEdit(u)}>
                        Сохранить
                      </button>
                      <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEditingId(null)}>
                        Отмена
                      </button>
                    </div>
                  </div>
                ) : u.words && u.words.length > 0 ? (
                  <div className="turn-words">
                    {u.words.map((w, idx) => {
                      const isSpoken =
                        w.start !== null &&
                        w.end !== null &&
                        currentTime >= w.start &&
                        currentTime <= w.end + 0.08;
                      const isNbz = w.text.includes("[НБЗ]");
                      return (
                        <React.Fragment key={`${u.id}-${idx}`}>
                          <button
                            type="button"
                            className={`word-token ${isSpoken ? "spoken" : ""} ${isNbz ? "censored" : ""}`}
                            onClick={() => seekTo(w.start ?? u.start_sec)}
                          >
                            {w.text}
                          </button>{" "}
                        </React.Fragment>
                      );
                    })}
                  </div>
                ) : (
                  <div className="turn-words">{u.text}</div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
