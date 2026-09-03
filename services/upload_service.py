from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from core.errors import AppError
from core.settings import settings
from db.orm_models import EventPhoto, UploadJob
from api.schemas import UploadJobResponse
from services.storage_service import (
    StorageError,
    ensure_upload_root,
    extract_zip_file,
    save_upload_file,
)


class UploadValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, error_code="upload_validation_error")


def _validate_image(upload_file: UploadFile, label: str) -> None:
    filename = upload_file.filename or ""
    extension = Path(filename).suffix.lower()

    if not filename.strip():
        raise UploadValidationError(f"{label} must have a filename")
    if extension not in settings.allowed_image_extensions:
        allowed = ", ".join(sorted(settings.allowed_image_extensions))
        raise UploadValidationError(f"{label} must be one of: {allowed}")


def _validate_zip(upload_file: UploadFile) -> None:
    filename = upload_file.filename or ""
    extension = Path(filename).suffix.lower()

    if not filename.strip():
        raise UploadValidationError("Event photos zip must have a filename")
    if extension != ".zip":
        raise UploadValidationError("Event photos upload must be a .zip file")


async def create_upload_job(
    db: Session, selfie: UploadFile, event_photos_zip: UploadFile
) -> UploadJobResponse:
    _validate_image(selfie, "Selfie")
    _validate_zip(event_photos_zip)

    ensure_upload_root()

    job = UploadJob(
        selfie_filename=Path(selfie.filename or "").name,
        selfie_storage_path="",
        event_photo_count=0,
    )
    db.add(job)
    db.flush()

    job_root = settings.upload_root / job.id
    selfie_dir = job_root / "selfie"
    event_dir = job_root / "event_photos"
    archive_dir = job_root / "archive"

    try:
        _, selfie_storage_path = await save_upload_file(selfie, selfie_dir)
        job.selfie_storage_path = selfie_storage_path

        _, zip_storage_path = await save_upload_file(event_photos_zip, archive_dir)

        try:
            extracted_files = extract_zip_file(Path(zip_storage_path), event_dir)
        except StorageError as exc:
            raise UploadValidationError(str(exc)) from exc
        except Exception as exc:
            raise UploadValidationError("Could not extract the uploaded zip file") from exc

        image_files = [
            file_path
            for file_path in extracted_files
            if file_path.suffix.lower() in settings.allowed_image_extensions
        ]

        if not image_files:
            raise UploadValidationError("Zip file does not contain any supported image files")
        if len(image_files) > settings.max_event_photos:
            raise UploadValidationError(
                f"Zip contains more than {settings.max_event_photos} supported images"
            )

        for file_path in image_files:
            db.add(
                EventPhoto(
                    job_id=job.id,
                    original_filename=file_path.name,
                    storage_path=str(file_path),
                )
            )

        job.event_photo_count = len(image_files)
        db.commit()
        db.refresh(job)

    except Exception:
        db.rollback()
        shutil.rmtree(job_root, ignore_errors=True)
        raise

    return UploadJobResponse(
        job_id=job.id,
        status=job.status.value,
        selfie_filename=job.selfie_filename,
        event_photo_count=job.event_photo_count,
        created_at=job.created_at,
    )
