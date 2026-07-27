from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.settings import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_runtime_schema() -> None:
    """Adds columns introduced after the first `upload_jobs` table was
    created, without wiping the existing SQLite dev database. Only needed
    on SQLite -- a fresh database (or Postgres, once that's supported)
    already gets these columns from Base.metadata.create_all()."""
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.connect() as connection:
        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(upload_jobs)")).all()
        }
    missing_columns = {
        "queued_at": "DATETIME",
        "processing_started_at": "DATETIME",
        "rq_job_id": "VARCHAR(255)",
        "last_error": "TEXT",
    }

    with engine.begin() as connection:
        for column_name, column_type in missing_columns.items():
            if column_name not in columns:
                connection.execute(
                    text(f"ALTER TABLE upload_jobs ADD COLUMN {column_name} {column_type}")
                )
