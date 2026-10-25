from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from dentabridge_api.db.models import Base

API = Path(__file__).resolve().parents[1]


def _config(url: str) -> Config:
    config = Config(str(API / "alembic.ini"))
    config.set_main_option("script_location", str(API / "alembic"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def test_alembic_upgrade_head_on_empty_sqlite(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'fresh.db'}"
    command.upgrade(_config(url), "head")
    engine = create_engine(url)
    assert set(inspect(engine).get_table_names()) == {*Base.metadata.tables, "alembic_version"}


def test_migrations_and_models_describe_the_same_schema(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'drift.db'}"
    command.upgrade(_config(url), "head")
    with create_engine(url).connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
