"""quotes, holds and bookings

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quotes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("clinic_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_json", sa.Text, nullable=False),
    )
    op.create_table(
        "holds",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("slot_id", sa.String(36), nullable=False),
        sa.Column("clinic_id", sa.String(64), nullable=False),
        sa.Column("dentist_id", sa.String(64), nullable=False),
        sa.Column("start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state", sa.String(12), nullable=False),
    )
    op.create_index(
        "ux_active_hold_per_slot",
        "holds",
        ["slot_id"],
        unique=True,
        sqlite_where=sa.text("state = 'active'"),
        postgresql_where=sa.text("state = 'active'"),
    )
    op.create_table(
        "bookings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("slot_id", sa.String(36), nullable=False),
        sa.Column("clinic_id", sa.String(64), nullable=False),
        sa.Column("dentist_id", sa.String(64), nullable=False),
        sa.Column("start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quote_id", sa.String(36), nullable=False),
        sa.Column("hold_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("tickets_json", sa.Text, nullable=False),
        sa.Column("shared_analysis_id", sa.String(36), nullable=True),
        sa.Column("snapshot_json", sa.Text, nullable=True),
        sa.Column("snapshot_purge_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ux_confirmed_booking_per_slot",
        "bookings",
        ["slot_id"],
        unique=True,
        sqlite_where=sa.text("status = 'confirmed'"),
        postgresql_where=sa.text("status = 'confirmed'"),
    )


def downgrade() -> None:
    op.drop_index("ux_confirmed_booking_per_slot", "bookings")
    op.drop_table("bookings")
    op.drop_index("ux_active_hold_per_slot", "holds")
    op.drop_table("holds")
    op.drop_table("quotes")
