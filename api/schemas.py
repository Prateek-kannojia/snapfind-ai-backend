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
    # Which zip parts have landed. Used by the client to resume an
    # interrupted upload; empty once the upload is finished.
    uploaded_parts: list[int] = Field(default_factory=list)


class PartUploadUrl(BaseModel):
    part_number: int
    url: str


class UploadInitRequest(BaseModel):
    selfie_filename: str
    zip_filename: str
    part_count: int = Field(..., ge=1, description="How many parts the client will send")


class UploadInitResponse(BaseModel):
    job_id: str
    status: str
    # PUT these with NO Content-Type header — see services/storage_service.py
    selfie_upload_url: str
    zip_upload_urls: list[PartUploadUrl]
    created_at: datetime


class PartUrlsRequest(BaseModel):
    part_numbers: list[int] = Field(..., min_length=1)


class PartUrlsResponse(BaseModel):
    job_id: str
    zip_upload_urls: list[PartUploadUrl]


class MatchListResponse(BaseModel):
    job_id: str
    status: str
    match_count: int = Field(..., ge=0)
    matches: list[MatchedPhotoResponse]


class HealthResponse(BaseModel):
    message: str
