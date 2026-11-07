import React from "react";
import type { ModelRegistryItem, SystemStatus } from "../types";

interface Props {
  status: SystemStatus | null;
  models: ModelRegistryItem[];
  onRefresh: () => void;
}

export const QueueModelsView: React.FC<Props> = ({ status, models, onRefresh }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
      {status && (
        <>
          <div className="grid-4">
            <div className="stat-card">
              <div className="stat-label">Очередь задач (FR-10)</div>
              <div className="stat-value">
                {status.queue.queued_count} в очереди / {status.queue.active_count} в работе
              </div>
              <div className="stat-sub">
                Объём: {status.queue.queued_audio_hours} ч · ETA ~{Math.ceil(status.queue.estimated_total_wait_sec)} с
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Вычислитель / VRAM (CON-03)</div>
              <div className="stat-value" style={{ fontSize: "1.15rem" }}>
                {status.gpu.available ? `${status.gpu.vram_total_mb} МБ VRAM` : "CPU / Демо"}
              </div>
              <div className="stat-sub">{status.gpu.device_name}</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Хранилище и СУБД</div>
              <div className="stat-value">{status.disk_free_gb} ГБ своб.</div>
              <div className="stat-sub">СУБД: {status.database_dialect.toUpperCase()}</div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Изоляция 152-ФЗ (NFR-04)</div>
              <div className="stat-value" style={{ fontSize: "1.15rem" }}>
                {status.network_isolation.configured_host}
              </div>
              <div className="stat-sub">
                Внешний трафик данных: 0 байт · Аудит ≥ {status.network_isolation.audit_retention_days} дн.
              </div>
            </div>
          </div>

          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.6rem" }}>
              <h3 style={{ margin: 0 }}>Диагностика локальной среды эксплуатации (п. 3.4, 4.6)</h3>
              <button type="button" className="btn btn-secondary btn-sm" onClick={onRefresh}>
                Обновить состояние
              </button>
            </div>
            <div style={{ fontSize: "0.88rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "0.6rem" }}>
              <div>
                <strong>Папка пользователя (NFR-04.1):</strong> <code>{status.data_dir}</code>
              </div>
              <div>
                <strong>FFmpeg (−16 LUFS):</strong>{" "}
                {status.ffmpeg.available ? (
                  <span className="pill pill-green">Доступен</span>
                ) : (
                  <span className="pill pill-red">Не найден</span>
                )}{" "}
                <code style={{ fontSize: "0.78rem" }}>{status.ffmpeg.path || ""}</code>
              </div>
              <div>
                <strong>WhisperX / PyTorch:</strong>{" "}
                {status.whisperx_installed ? (
                  <span className="pill pill-green">Установлен (боевой режим)</span>
                ) : (
                  <span className="pill pill-amber">Демо-симулятор ASR (ffmpeg + чекпоинты реальные)</span>
                )}
              </div>
              <div>
                <strong>Статистика обработки:</strong> всего {status.stats.total_recordings}, успешно{" "}
                {status.stats.completed_recordings}, сбоев {status.stats.failed_recordings}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Реестр моделей п. 4.9, 4.11 */}
      <div className="card">
        <h2 style={{ fontSize: "1.12rem", marginBottom: "0.35rem" }}>
          Реестр открытых ИИ-моделей и лицензий (п. 4.9, п. 4.11, CON-02)
        </h2>
        <p style={{ margin: "0 0 0.85rem", fontSize: "0.86rem", color: "var(--text-muted)" }}>
          Все модели являются открытыми (open-source), работают локально без облачных API и допускают образовательное использование.
        </p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Подсистема</th>
                <th>Модель</th>
                <th>Версия / Квантование</th>
                <th>Пик VRAM</th>
                <th>Лицензия</th>
                <th>Статус</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id}>
                  <td><code>{m.component}</code></td>
                  <td>
                    <strong>{m.model_name}</strong>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                      {JSON.stringify(m.metrics)}
                    </div>
                  </td>
                  <td>
                    {m.version} · <code>{m.quantization}</code>
                  </td>
                  <td>~{m.vram_mb} МБ</td>
                  <td><span className="pill pill-blue">{m.license}</span></td>
                  <td>
                    {m.is_active ? (
                      <span className="pill pill-green">Активна (Этап 2)</span>
                    ) : (
                      <span className="pill pill-amber">План: Спринт 2</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
