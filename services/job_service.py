from __future__ import annotations

from pathlib import Path

from sqlalchemy import delete as sql_delete, select
from sqlalchemy.orm import Session, selectinload

from api.schemas import (
    JobSummaryResponse,
    MatchListResponse,
    MatchedPhotoResponse,
)
from db.orm_models import JobStatus, MatchedPhoto, UploadJob
from services.face_matcher import build_matches_for_job


class JobServiceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class JobNotFoundError(JobServiceError):
    def __init__(self, message: str = "Job not found") -> None:
        super().__init__(message, status_code=404)


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
        raise JobServiceError("Matched photo not found", status_code=404)
    return match


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


def get_job_detail(db: Session, job_id: str) -> JobSummaryResponse:
    job = _get_job_or_raise(db, job_id)
    return _build_job_summary_response(job)


def process_job(db: Session, job_id: str, threshold: float) -> JobSummaryResponse:
    """Runs face matching synchronously, inside the request. Fine for a
    handful of test photos; a bigger album will block the HTTP connection
    for the whole run."""
    job = _get_job_or_raise(db, job_id)
    if not job.event_photos:
        raise JobServiceError("Job has no event photos to process")

    job.status = JobStatus.processing
    db.commit()

    try:
        matches = build_matches_for_job(db, job, list(job.event_photos), threshold)
        db.execute(sql_delete(MatchedPhoto).where(MatchedPhoto.job_id == job.id))
        db.add_all(matches)
        job.status = JobStatus.completed
        db.commit()
    except Exception as exc:
        db.rollback()
        job = _get_job_or_raise(db, job_id)
        job.status = JobStatus.failed
        db.commit()
        raise JobServiceError(f"Face matching failed: {exc}") from exc

    db.refresh(job)
    return _build_job_summary_response(job)


def get_job_matches(db: Session, job_id: str) -> MatchListResponse:
    job = _get_job_or_raise(db, job_id)
    return _build_match_list_response(job)


def get_match_file(db: Session, job_id: str, match_id: int) -> tuple[Path, str]:
    match = _get_match_or_raise(db, job_id, match_id)
    file_path = Path(match.event_photo.storage_path)
    if not file_path.exists():
        raise JobServiceError("Matched photo file not found on disk", status_code=404)
    return file_path, match.event_photo.original_filename
