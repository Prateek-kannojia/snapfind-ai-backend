from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.database import get_db
from api.schemas import JobSummaryResponse, MatchListResponse, UploadJobResponse
from services.job_service import (
    enqueue_job_processing,
    get_job_detail,
    get_job_matches,
    get_match_file,
)
from services.upload_service import create_upload_job

router = APIRouter()

# No try/except here for any of these routes: every error a service function
# can raise is an AppError (or subclass) with its own status_code/error_code
# baked in, and main.py's single @app.exception_handler(AppError) converts
# it to the right JSON response automatically. See core/errors.py.


@router.post("/jobs/upload", response_model=UploadJobResponse)
async def upload_event_photos(
    selfie: Annotated[UploadFile, File(...)],
    event_photos_zip: Annotated[UploadFile, File(...)],
    db: Session = Depends(get_db),
) -> UploadJobResponse:
    return await create_upload_job(db=db, selfie=selfie, event_photos_zip=event_photos_zip)


@router.get("/jobs/{job_id}", response_model=JobSummaryResponse)
def fetch_job(job_id: str, db: Session = Depends(get_db)) -> JobSummaryResponse:
    return get_job_detail(db=db, job_id=job_id)


@router.post("/jobs/{job_id}/process", response_model=JobSummaryResponse)
def process_uploaded_job(
    job_id: str,
    threshold: float = Query(0.68, gt=0, lt=1.0),
    db: Session = Depends(get_db),
) -> JobSummaryResponse:
    return enqueue_job_processing(db=db, job_id=job_id, threshold=threshold)


@router.get("/jobs/{job_id}/matches", response_model=MatchListResponse)
def fetch_job_matches(job_id: str, db: Session = Depends(get_db)) -> MatchListResponse:
    return get_job_matches(db=db, job_id=job_id)


@router.get("/jobs/{job_id}/matches/{match_id}/download")
def download_matched_photo(
    job_id: str, match_id: int, db: Session = Depends(get_db)
) -> FileResponse:
    file_path, filename = get_match_file(db=db, job_id=job_id, match_id=match_id)
    return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
