from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from db.database import get_db
from api.schemas import (
    JobSummaryResponse,
    MatchListResponse,
    PartUrlsRequest,
    PartUrlsResponse,
    UploadInitRequest,
    UploadInitResponse,
)
from services.job_service import (
    enqueue_job_processing,
    get_job_detail,
    get_job_matches,
    get_match_download_url,
    load_job_or_raise,
)
from services.upload_service import complete_upload_job, init_upload_job, refresh_part_urls

router = APIRouter()

# No try/except in any route: service functions raise AppError subclasses and
# main.py's single handler turns them into responses. See core/errors.py.
#
# Uploads go straight from client to object storage via presigned URLs, so no
# photo bytes pass through this API. Clients must PUT with no Content-Type
# header — see services/storage_service.py.


@router.post("/jobs/upload/init", response_model=UploadInitResponse)
def start_upload(
    payload: UploadInitRequest, db: Session = Depends(get_db)
) -> UploadInitResponse:
    return init_upload_job(
        db=db,
        selfie_filename=payload.selfie_filename,
        zip_filename=payload.zip_filename,
        part_count=payload.part_count,
    )


@router.post("/jobs/{job_id}/upload/urls", response_model=PartUrlsResponse)
def get_part_urls(
    job_id: str, payload: PartUrlsRequest, db: Session = Depends(get_db)
) -> PartUrlsResponse:
    """Fresh URLs for specific parts — to resume, or when the originals expired."""
    job = load_job_or_raise(db, job_id)
    return PartUrlsResponse(
        job_id=job.id,
        zip_upload_urls=refresh_part_urls(db=db, job=job, part_numbers=payload.part_numbers),
    )


@router.post("/jobs/{job_id}/upload/complete", response_model=JobSummaryResponse)
def finish_upload(job_id: str, db: Session = Depends(get_db)) -> JobSummaryResponse:
    job = load_job_or_raise(db, job_id)
    complete_upload_job(db=db, job=job)
    return get_job_detail(db=db, job_id=job_id)


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
) -> RedirectResponse:
    """Kept for compatibility — redirects to a presigned URL. Clients can use
    the download_url from /matches directly and skip this hop."""
    return RedirectResponse(
        url=get_match_download_url(db=db, job_id=job_id, match_id=match_id), status_code=307
    )
