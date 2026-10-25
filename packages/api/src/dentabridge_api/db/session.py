"""Engine and session factory. SQLite is the default; Postgres works through db_url."""

from sqlalchemy import Engine, StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

from dentabridge_api.db.models import Base


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        in_memory = url in ("sqlite://", "sqlite:///:memory:")
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if in_memory else None,
        )
    return create_engine(url)


def init_schema(engine: Engine) -> None:
    """Create the tables if they are missing (demo and tests); deployments use alembic."""
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
