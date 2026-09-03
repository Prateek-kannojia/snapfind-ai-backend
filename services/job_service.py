from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete as sql_delete, or_, select, update
from sqlalchemy.orm import Session, selectinload

from api.schemas import (
    JobSummaryResponse,
    MatchListResponse,
    MatchedPhotoResponse,
)
from core.errors import AppError
from core.settings import settings
from db.database import SessionLocal
from db.orm_models import JobStatus, MatchedPhoto, UploadJob
from services.face_matcher import build_matches_for_job
from services.queue_service import enqueue_face_matching_job


class JobServiceError(AppError):
    """Base for job-domain errors. Named subclasses below cover every case
    this module actually raises — nothing here builds an error from a bare
    message + status_code at the call site anymore, so `isinstance` checks
    and greps both work, and the HTTP mapping lives with the error, not
    scattered across call sites."""

    def __init__(self, message: str, *, status_code: int = 400, error_code: str = "job_error") -> None:
        super().__init__(message, status_code=status_code, error_code=error_code)


class JobNotFoundError(JobServiceError):
    def __init__(self, message: str = "Job not found") -> None:
        super().__init__(message, status_code=404, error_code="job_not_found")


class MatchNotFoundError(JobServiceError):
    def __init__(self, message: str = "Matched photo not found") -> None:
        super().__init__(message, status_code=404, error_code="match_not_found")


class MatchFileMissingError(JobServiceError):
    def __init__(self, message: str = "Matched photo file not found on disk") -> None:
        super().__init__(message, status_code=404, error_code="match_file_missing")


class JobAlreadyQueuedError(JobServiceError):
    def __init__(self, message: str = "Job is already queued for processing") -> None:
        super().__init__(message, status_code=409, error_code="job_already_queued")


class JobAlreadyProcessingError(JobServiceError):
    def __init__(self, message: str = "Job is currently being processed") -> None:
        super().__init__(message, status_code=409, error_code="job_already_processing")


class NoEventPhotosError(JobServiceError):
    def __init__(self, message: str = "Job has no event photos to process") -> None:
        super().__init__(message, status_code=400, error_code="no_event_photos")


class QueueUnavailableError(JobServiceError):
    def __init__(self, message: str = "Could not enqueue job for processing. Is Redis running?") -> None:
        super().__init__(message, status_code=503, error_code="queue_unavailable")


def _load_job(db: Session, job_id: str) -> UploadJob | None:
    statement = (
        select(UploadJob)
        .options(
            selectinload(UploadJob.event_photos),
            selectinload(UploadJob.matched_photos).selectinload(MatchedPhoto.event_photo),
        )
        .where(UploadJob.id == job_id)
    )
    return db.execute(statement).scalar_one_or_none()


def _load_match(db: Session, job_id: str, match_id: int) -> MatchedPhoto | None:
    statement = (
        select(MatchedPhoto)
        .options(selectinload(MatchedPhoto.event_photo))
        .where(MatchedPhoto.job_id == job_id, MatchedPhoto.id == match_id)
    )
    return db.execute(statement).scalar_one_or_none()


def _get_job_or_raise(db: Session, job_id: str) -> UploadJob:
    job = _load_job(db, job_id)
    if job is None:
        raise JobNotFoundError()
    return job


def _get_match_or_raise(db: Session, job_id: str, match_id: int) -> MatchedPhoto:
    match = _load_match(db, job_id, match_id)
    if match is None:
        raise MatchNotFoundError()
    return match


def _set_job_status(db: Session, job: UploadJob, status: JobStatus) -> None:
    job.status = status
    db.flush()


def _replace_job_matches(db: Session, job: UploadJob, matches: list[MatchedPhoto]) -> None:
    db.execute(sql_delete(MatchedPhoto).where(MatchedPhoto.job_id == job.id))
    db.flush()
    db.add_all(matches)


def _build_job_summary_response(job: UploadJob) -> JobSummaryResponse:
    return JobSummaryResponse(
        job_id=job.id,
        status=job.status.value,
        selfie_filename=job.selfie_filename,
        event_photo_count=job.event_photo_count,
        matched_photo_count=len(job.matched_photos),
        created_at=job.created_at,
    )


def _build_match_list_response(job: UploadJob) -> MatchListResponse:
    matches = sorted(job.matched_photos, key=lambda item: item.match_distance)
    return MatchListResponse(
        job_id=job.id,
        status=job.status.value,
        match_count=len(matches),
        matches=[
            MatchedPhotoResponse(
                id=match.id,
                event_photo_id=match.event_photo_id,
                filename=match.event_photo.original_filename,
                match_distance=match.match_distance,
                download_url=f"/jobs/{job.id}/matches/{match.id}/download",
                created_at=match.created_at,
            )
            for match in matches
        ],
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stale_cutoff() -> datetime:
    return _utc_now() - timedelta(seconds=settings.job_stale_after_seconds)


def _mark_job_failed(db: Session, job_id: str, error: str | None = None) -> None:
    db.rollback()
    job = _load_job(db, job_id)
    if job is None:
        return
    job.status = JobStatus.failed
    job.last_error = error
    db.commit()


def get_job_detail(db: Session, job_id: str) -> JobSummaryResponse:
    job = _get_job_or_raise(db, job_id)
    return _build_job_summary_response(job)


def enqueue_job_processing(db: Session, job_id: str, threshold: float) -> JobSummaryResponse:
    """Mark a job as queued, enqueue it in RQ, and return immediately."""
    now = _utc_now()
    stale_cutoff = _stale_cutoff()
    result = db.execute(
        update(UploadJob)
        .where(
            UploadJob.id == job_id,
            or_(
                UploadJob.status.not_in([JobStatus.queued, JobStatus.processing]),
                (
                    (UploadJob.status == JobStatus.queued)
                    & (
                        (UploadJob.queued_at.is_(None))
                        | (UploadJob.queued_at < stale_cutoff)
                    )
                ),
                (
                    (UploadJob.status == JobStatus.processing)
                    & (
                        (UploadJob.processing_started_at.is_(None))
                        | (UploadJob.processing_started_at < stale_cutoff)
                    )
                ),
            ),
        )
        .values(
            status=JobStatus.queued,
            queued_at=now,
            processing_started_at=None,
            rq_job_id=None,
            last_error=None,
        )
    )
    db.flush()
    db.expire_all()

    if result.rowcount == 0:
        job = _load_job(db, job_id)
        if job is None:
            raise JobNotFoundError()
        if job.status == JobStatus.queued:
            raise JobAlreadyQueuedError()
        raise JobAlreadyProcessingError()

    job = _get_job_or_raise(db, job_id)
    if not job.event_photos:
        db.rollback()
        raise NoEventPhotosError()

    db.commit()
    try:
        rq_job_id = enqueue_face_matching_job(job_id=job_id, threshold=threshold)
    except Exception as exc:
        db.rollback()
        job = _get_job_or_raise(db, job_id)
        job.status = JobStatus.pending
        job.queued_at = None
        job.rq_job_id = None
        db.commit()
        raise QueueUnavailableError() from exc

    job = _get_job_or_raise(db, job_id)
    job.rq_job_id = rq_job_id
    db.commit()
    db.refresh(job)
    return _build_job_summary_response(job)


def _claim_queued_job(db: Session, job_id: str) -> UploadJob | None:
    result = db.execute(
        update(UploadJob)
        .where(UploadJob.id == job_id, UploadJob.status == JobStatus.queued)
        .values(status=JobStatus.processing, processing_started_at=_utc_now())
    )
    db.flush()
    db.expire_all()

    if result.rowcount == 0:
        return None

    job = _get_job_or_raise(db, job_id)
    if not job.event_photos:
        job.status = JobStatus.failed
        job.last_error = "Job has no event photos to process"
        db.commit()
        return None

    db.commit()
    return _get_job_or_raise(db, job_id)


def run_job_processing(job_id: str, threshold: float) -> None:
    """RQ worker task. Owns its own DB session and runs outside the API process."""
    db = SessionLocal()
    try:
        job = _claim_queued_job(db, job_id)
        if job is None:
            return
        matches = build_matches_for_job(db, job, list(job.event_photos), threshold)
        _replace_job_matches(db, job, matches)
        job.last_error = None
        _set_job_status(db, job, JobStatus.completed)
        db.commit()
    except Exception as exc:
        _mark_job_failed(db, job_id, str(exc))
    finally:
        db.close()


def get_job_matches(db: Session, job_id: str) -> MatchListResponse:
    job = _get_job_or_raise(db, job_id)
    return _build_match_list_response(job)


def get_match_file(db: Session, job_id: str, match_id: int) -> tuple[Path, str]:
    match = _get_match_or_raise(db, job_id, match_id)
    file_path = Path(match.event_photo.storage_path)
    if not file_path.exists():
        raise MatchFileMissingError()
    return file_path, match.event_photo.original_filename
