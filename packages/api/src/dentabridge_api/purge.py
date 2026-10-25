"""Consents and results live 24 hours; expired rows are removed whenever one is touched."""

from datetime import datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from dentabridge_api.db.models import AnalysisRow, ConsentRow


def purge_expired(session: Session, now: datetime) -> None:
    session.execute(delete(AnalysisRow).where(AnalysisRow.expires_at <= now))
    session.execute(delete(ConsentRow).where(ConsentRow.expires_at <= now))
