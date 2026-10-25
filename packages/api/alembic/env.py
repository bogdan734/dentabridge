import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from dentabridge_api.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

url = config.get_main_option("sqlalchemy.url") or os.environ.get("DENTABRIDGE_DB_URL", "")
if not url:
    raise SystemExit("set sqlalchemy.url or DENTABRIDGE_DB_URL")

with create_engine(url).connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
