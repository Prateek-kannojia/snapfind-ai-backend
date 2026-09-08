from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from core.errors import AppError
from core.settings import settings
from db.orm_models import UploadJob
from api.schemas import PartUploadUrl, UploadInitResponse
from services import storage_service as storage


class UploadValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, error_code="upload_validation_error")


def _validate_image_filename(filename: str, label: str) -> None:
    name = (filename or "").strip()
    if not name:
        raise UploadValidationError(f"{label} must have a filename")
    if Path(name).suffix.lower() not in settings.allowed_image_extensions:
        allowed = ", ".join(sorted(settings.allowed_image_extensions))
        raise UploadValidationError(f"{label} must be one of: {allowed}")


def _validate_zip_filename(filename: str) -> None:
    name = (filename or "").strip()
    if not name:
        raise UploadValidationError("Event photos zip must have a filename")
    if Path(name).suffix.lower() != ".zip":
        raise UploadValidationError("Event photos upload must be a .zip file")


def _part_urls(key: str, upload_id: str, part_numbers: list[int]) -> list[PartUploadUrl]:
    return [
        PartUploadUrl(part_number=n, url=storage.presign_upload_part(key, upload_id, n))
        for n in part_numbers
    ]


def init_upload_job(
    db: Session, selfie_filename: str, zip_filename: str, part_count: int
) -> UploadInitResponse:
    """Create the job and hand back presigned URLs. No bytes touch this server —
    the client uploads straight to object storage, then calls complete."""
    _validate_image_filename(selfie_filename, "Selfie")
    _validate_zip_filename(zip_filename)

    job = UploadJob(
        selfie_filename=Path(selfie_filename).name,
        selfie_object_key="",
        event_photo_count=0,
    )
    db.add(job)
    db.flush()  # need job.id to build keys

    selfie_key = storage.build_object_key(
        job.id, "selfie", storage.unique_filename(selfie_filename)
    )
    zip_key = storage.build_object_key(
        job.id, "archive", storage.unique_filename(zip_filename)
    )

    try:
        upload_id = storage.create_multipart_upload(zip_key)
        selfie_url = storage.presign_put(selfie_key)
        part_urls = _part_urls(zip_key, upload_id, list(range(1, part_count + 1)))
    except storage.StorageError as exc:
        db.rollback()
        raise UploadValidationError(f"Could not start upload: {exc}") from exc

    job.selfie_object_key = selfie_key
    job.zip_object_key = zip_key
    job.zip_upload_id = upload_id
    db.commit()
    db.refresh(job)

    return UploadInitResponse(
        job_id=job.id,
        status=job.status.value,
        selfie_upload_url=selfie_url,
        zip_upload_urls=part_urls,
        created_at=job.created_at,
    )


def refresh_part_urls(db: Session, job: UploadJob, part_numbers: list[int]) -> list[PartUploadUrl]:
    """Fresh URLs for specific parts — used to resume, or when the originals expired."""
    if not job.zip_object_key or not job.zip_upload_id:
        raise UploadValidationError("This job has no upload in progress")
    return _part_urls(job.zip_object_key, job.zip_upload_id, part_numbers)


def uploaded_part_numbers(job: UploadJob) -> list[int]:
    """Which parts have landed. Empty once the upload has been completed."""
    if not job.zip_object_key or not job.zip_upload_id:
        return []
    try:
        parts = storage.list_uploaded_parts(job.zip_object_key, job.zip_upload_id)
    except storage.StorageError:
        return []  # upload already completed/aborted — nothing in progress
    return [part["PartNumber"] for part in parts]


def complete_upload_job(db: Session, job: UploadJob) -> UploadJob:
    """Assemble the uploaded parts into one object. The job stays `pending` —
    the client triggers matching with POST /jobs/{id}/process as before.
    Extraction runs in the worker, so event_photo_count stays 0 until then."""
    if not job.zip_object_key or not job.zip_upload_id:
        raise UploadValidationError("This job has no upload in progress")

    try:
        parts = storage.list_uploaded_parts(job.zip_object_key, job.zip_upload_id)
        if not parts:
            raise UploadValidationError("No uploaded parts found for this job")
        storage.complete_multipart_upload(job.zip_object_key, job.zip_upload_id, parts)
    except storage.StorageError as exc:
        raise UploadValidationError(f"Could not finish upload: {exc}") from exc

    if not storage.object_exists(job.selfie_object_key):
        raise UploadValidationError("Selfie was never uploaded")

    job.zip_upload_id = None  # upload is done; nothing left to resume
    db.commit()
    db.refresh(job)
    return job
