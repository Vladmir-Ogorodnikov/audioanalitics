import React, { useEffect, useState } from "react";
import { api } from "../api";
import type { AuditLogItem } from "../types";

interface Props {
  onError: (err: string) => void;
}

export const AuditLogView: React.FC<Props> = ({ onError }) => {
  const [logs, setLogs] = useState<AuditLogItem[]>([]);
  const [filter, setFilter] = useState("all");

  const loadLogs = async (ev = filter) => {
    try {
      const rows = await api.getAuditLogs(ev);
      setLogs(rows);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    loadLogs(filter);
  }, [filter]);

  return (
    <div className="card">
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "0.75rem", marginBottom: "0.85rem" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.12rem" }}>
            Журнал операций информационной безопасности (NFR-04.3, 152-ФЗ)
          </h2>
          <p style={{ margin: "0.2rem 0 0", fontSize: "0.84rem", color: "var(--text-muted)" }}>
            Срок хранения записей — не менее 90 суток. Журналируются загрузка, очередь, чекпоинты, правка, переименование спикеров, экспорт и удаление.
          </p>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "center" }}>
          <select
            className="select"
            style={{ width: "auto" }}
            aria-label="Фильтр типа события"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">Все события</option>
            <option value="upload">upload (загрузка)</option>
            <option value="queue_start">queue_start (старт)</option>
            <option value="complete">complete (завершение)</option>
            <option value="checkpoint_resumed">checkpoint_resumed (FR-09)</option>
            <option value="speaker_rename">speaker_rename (FR-04)</option>
            <option value="utterance_edit">utterance_edit (FR-07)</option>
            <option value="export">export (выгрузка)</option>
            <option value="delete">delete (удаление FR-08)</option>
            <option value="lexicon_update">lexicon_update (словари)</option>
          </select>

          <button type="button" className="btn btn-secondary btn-sm" onClick={() => loadLogs(filter)}>
            Обновить
          </button>
          <a className="btn btn-primary btn-sm" href="/api/v1/audit-logs/export?format=csv" download>
            ⬇ Выгрузить CSV
          </a>
          <a className="btn btn-secondary btn-sm" href="/api/v1/audit-logs/export?format=json" download>
            ⬇ Выгрузить JSON
          </a>
        </div>
      </div>

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Дата и время (UTC)</th>
              <th>Тип события</th>
              <th>Объект</th>
              <th>IP</th>
              <th>Описание операции</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: "center", color: "var(--text-muted)", padding: "1.5rem" }}>
                  Записей в журнале по выбранному фильтру пока нет.
                </td>
              </tr>
            ) : (
              logs.map((item) => (
                <tr key={item.id}>
                  <td>#{item.id}</td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {item.timestamp ? new Date(item.timestamp).toLocaleString("ru-RU") : "—"}
                  </td>
                  <td><span className="pill pill-blue">{item.event_type}</span></td>
                  <td>
                    {item.entity_type ? `${item.entity_type}${item.entity_id ? ` #${item.entity_id}` : ""}` : "—"}
                  </td>
                  <td><code>{item.client_ip}</code></td>
                  <td>{item.message}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
