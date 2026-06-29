from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.database import get_db
from api.schemas import JobSummaryResponse, MatchListResponse, UploadJobResponse
from services.job_service import (
    JobServiceError,
    get_job_detail,
    get_job_matches,
    get_match_file,
    process_job,
)
from services.upload_service import UploadValidationError, create_upload_job

router = APIRouter()


@router.post("/jobs/upload", response_model=UploadJobResponse)
async def upload_event_photos(
    selfie: Annotated[UploadFile, File(...)],
    event_photos_zip: Annotated[UploadFile, File(...)],
    db: Session = Depends(get_db),
) -> UploadJobResponse:
    try:
        return await create_upload_job(
            db=db, selfie=selfie, event_photos_zip=event_photos_zip
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobSummaryResponse)
def fetch_job(job_id: str, db: Session = Depends(get_db)) -> JobSummaryResponse:
    try:
        return get_job_detail(db=db, job_id=job_id)
    except JobServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/process", response_model=JobSummaryResponse)
def process_uploaded_job(
    job_id: str,
    threshold: float = Query(0.68, gt=0, lt=1.0),
    db: Session = Depends(get_db),
) -> JobSummaryResponse:
    try:
        return process_job(db=db, job_id=job_id, threshold=threshold)
    except JobServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/matches", response_model=MatchListResponse)
def fetch_job_matches(job_id: str, db: Session = Depends(get_db)) -> MatchListResponse:
    try:
        return get_job_matches(db=db, job_id=job_id)
    except JobServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/matches/{match_id}/download")
def download_matched_photo(
    job_id: str, match_id: int, db: Session = Depends(get_db)
) -> FileResponse:
    try:
        file_path, filename = get_match_file(db=db, job_id=job_id, match_id=match_id)
        return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
    except JobServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
