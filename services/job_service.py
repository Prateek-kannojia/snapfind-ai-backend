from __future__ import annotations

import tempfile
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
from db.orm_models import EventPhoto, JobStatus, MatchedPhoto, UploadJob
from services import storage_service as storage
from services.face_matcher import build_matches_for_job
from services.queue_service import enqueue_face_matching_job
from services.storage_service import extract_zip_file
from services.upload_service import uploaded_part_numbers


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
    def __init__(self, message: str = "Matched photo file is no longer available") -> None:
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


class UploadNotCompleteError(JobServiceError):
    def __init__(self, message: str = "Upload is not complete for this job") -> None:
        super().__init__(message, status_code=409, error_code="upload_not_complete")


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
        uploaded_parts=uploaded_part_numbers(job),
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
                # Presigned, so the client fetches straight from object
                # storage. Expires — re-fetch this endpoint for fresh URLs.
                download_url=storage.presign_download(match.event_photo.object_key),
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


def load_job_or_raise(db: Session, job_id: str) -> UploadJob:
    """The job row itself, for routes that need more than the summary."""
    return _get_job_or_raise(db, job_id)


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
    # Photos don't exist yet — the worker extracts them. What must be true
    # here is that the zip upload actually finished.
    if not job.zip_object_key or job.zip_upload_id is not None:
        db.rollback()
        raise UploadNotCompleteError()

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

    # No event-photo check here any more: at claim time the zip hasn't been
    # extracted yet. _extract_photos_if_needed() handles an empty zip.
    db.commit()
    return _get_job_or_raise(db, job_id)


def _extract_photos_if_needed(db: Session, job: UploadJob) -> None:
    """Unpack the uploaded zip into one object per photo.

    Idempotent: a retry after a failed match run skips straight past this
    rather than re-downloading and re-uploading everything again.
    """
    if job.event_photos:
        return
    if not job.zip_object_key:
        raise NoEventPhotosError("Job has no uploaded zip")

    zip_bytes = storage.get_object(job.zip_object_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        zip_path = tmp_root / "archive.zip"
        zip_path.write_bytes(zip_bytes)
        extracted = extract_zip_file(zip_path, tmp_root / "extracted")

        images = [
            path
            for path in extracted
            if path.suffix.lower() in settings.allowed_image_extensions
        ]
        if not images:
            raise NoEventPhotosError("Zip file does not contain any supported image files")
        if len(images) > settings.max_event_photos:
            raise NoEventPhotosError(
                f"Zip contains more than {settings.max_event_photos} supported images"
            )

        for path in images:
            key = storage.build_object_key(
                job.id, "event_photos", storage.unique_filename(path.name)
            )
            storage.put_object(key, path.read_bytes())
            db.add(
                EventPhoto(job_id=job.id, original_filename=path.name, object_key=key)
            )

    job.event_photo_count = len(images)
    db.commit()


def run_job_processing(job_id: str, threshold: float) -> None:
    """RQ worker task. Owns its own DB session and runs outside the API process."""
    db = SessionLocal()
    try:
        job = _claim_queued_job(db, job_id)
        if job is None:
            return
        _extract_photos_if_needed(db, job)
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


def get_match_download_url(db: Session, job_id: str, match_id: int) -> str:
    """Presigned URL for one matched photo. The route redirects to it, so the
    photo bytes never pass through this server."""
    match = _get_match_or_raise(db, job_id, match_id)
    if not storage.object_exists(match.event_photo.object_key):
        raise MatchFileMissingError()
    return storage.presign_download(match.event_photo.object_key)
