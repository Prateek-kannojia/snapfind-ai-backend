from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MatchedPhotoResponse(BaseModel):
    id: int
    event_photo_id: int
    filename: str
    match_distance: float
    download_url: str
    created_at: datetime


class UploadJobResponse(BaseModel):
    job_id: str = Field(..., description="Identifier for the upload job")
    status: str = Field(..., description="Current processing state")
    selfie_filename: str
    event_photo_count: int = Field(..., ge=0)
    created_at: datetime


class JobSummaryResponse(UploadJobResponse):
    matched_photo_count: int = Field(..., ge=0)


class MatchListResponse(BaseModel):
    job_id: str
    status: str
    match_count: int = Field(..., ge=0)
    matches: list[MatchedPhotoResponse]


class HealthResponse(BaseModel):
    message: str
