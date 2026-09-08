from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


class JobStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class UploadJob(Base):
    __tablename__ = "upload_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    status: Mapped[JobStatus] = mapped_column(
        SqlEnum(JobStatus), default=JobStatus.pending, nullable=False
    )
    selfie_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    selfie_object_key: Mapped[str] = mapped_column(String(500), nullable=False)
    # Object key of the uploaded zip, and the multipart upload id needed to
    # resume/complete it. Nullable so existing rows survive the migration.
    zip_object_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    zip_upload_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    event_photo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rq_job_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    event_photos: Mapped[list["EventPhoto"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    matched_photos: Mapped[list["MatchedPhoto"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class EventPhoto(Base):
    __tablename__ = "event_photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("upload_jobs.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(500), nullable=False)
    # A real pgvector column on Postgres (production/Docker), a JSON-text
    # fallback on SQLite (zero-setup local dev — see core/settings.py; the
    # `vector` extension and type don't exist there). Both paths store the
    # same 512 ArcFace numbers; services/face_matcher.py handles the two
    # representations via _serialize_embedding()/_deserialize_embedding().
    embedding: Mapped[Any | None] = mapped_column(
        Text().with_variant(Vector(512), "postgresql"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    job: Mapped[UploadJob] = relationship(back_populates="event_photos")
    matched_photos: Mapped[list["MatchedPhoto"]] = relationship(
        back_populates="event_photo", cascade="all, delete-orphan"
    )


class MatchedPhoto(Base):
    __tablename__ = "matched_photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("upload_jobs.id"), nullable=False)
    event_photo_id: Mapped[int] = mapped_column(ForeignKey("event_photos.id"), nullable=False)
    match_distance: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    job: Mapped[UploadJob] = relationship(back_populates="matched_photos")
    event_photo: Mapped[EventPhoto] = relationship(back_populates="matched_photos")
