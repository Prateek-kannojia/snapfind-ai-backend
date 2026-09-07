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


def ensure_pgvector_extension() -> None:
    """Enables the pgvector extension on Postgres. Must run before
    Base.metadata.create_all(), since EventPhoto.embedding is a
    Vector(512) column on Postgres (see db/orm_models.py) and Postgres
    can't create that column type until the extension exists. No-op on
    SQLite, which doesn't have this concept at all.
    """
    if settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_runtime_schema() -> None:
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
        "zip_object_key": "VARCHAR(500)",
        "zip_upload_id": "VARCHAR(255)",
    }

    with engine.begin() as connection:
        for column_name, column_type in missing_columns.items():
            if column_name not in columns:
                connection.execute(
                    text(f"ALTER TABLE upload_jobs ADD COLUMN {column_name} {column_type}")
                )
