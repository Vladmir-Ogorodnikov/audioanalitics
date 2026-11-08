"""Добавляет профиль скорости обработки (speed_preset) в таблицу recordings.

Revision ID: 0002_speed_preset
Revises: 0001_sprint1
Create Date: 2026-11-07
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002_speed_preset"
down_revision: str | None = "0001_sprint1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("recordings")}
    if "speed_preset" not in cols:
        op.add_column(
            "recordings",
            sa.Column("speed_preset", sa.String(32), nullable=False, server_default="auto"),
        )


def downgrade() -> None:
    with op.batch_alter_table("recordings") as batch_op:
        batch_op.drop_column("speed_preset")
