"""Журнал операций безопасности (NFR-04.3): запись, хранение ≥ 90 дней и выгрузка CSV/JSON."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db_models import AuditLog


def log_audit_event(
    db: Session,
    event_type: str,
    message: str,
    *,
    entity_type: str | None = None,
    entity_id: int | None = None,
    details: dict | None = None,
    client_ip: str = "127.0.0.1",
    commit: bool = True,
) -> AuditLog:
    entry = AuditLog(
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        message=message,
        details_json=json.dumps(details or {}, ensure_ascii=False),
        client_ip=client_ip or "127.0.0.1",
    )
    db.add(entry)
    if commit:
        db.commit()
        db.refresh(entry)
    return entry


def cleanup_expired_audit_logs(db: Session, retention_days: int = 90) -> int:
    """Удаляет записи старше retention_days (не менее 90 дней по NFR-04.3)."""
    days = max(90, int(retention_days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    res = db.execute(delete(AuditLog).where(AuditLog.timestamp < cutoff))
    db.commit()
    return int(res.rowcount or 0)


def export_audit_csv(logs: list[AuditLog]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "event_type", "entity_type", "entity_id", "client_ip", "message", "details"])
    for item in logs:
        writer.writerow([
            item.id,
            item.timestamp.isoformat() if item.timestamp else "",
            item.event_type,
            item.entity_type or "",
            item.entity_id if item.entity_id is not None else "",
            item.client_ip,
            item.message,
            item.details_json,
        ])
    return buf.getvalue()


def export_audit_json(logs: list[AuditLog]) -> str:
    payload = [
        {
            "id": item.id,
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "event_type": item.event_type,
            "entity_type": item.entity_type,
            "entity_id": item.entity_id,
            "client_ip": item.client_ip,
            "message": item.message,
            "details": json.loads(item.details_json or "{}"),
        }
        for item in logs
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def fetch_audit_logs(db: Session, *, event_type: str | None = None, limit: int = 200) -> list[AuditLog]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(max(limit, 1), 5000))
    if event_type:
        stmt = stmt.where(AuditLog.event_type == event_type)
    return list(db.scalars(stmt).all())
