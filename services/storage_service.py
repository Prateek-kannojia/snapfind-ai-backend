from __future__ import annotations

import asyncio
import mimetypes
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import boto3
from fastapi import UploadFile

from core.settings import settings


# Deliberately a plain Exception, not an AppError: storage_service is a
# storage-only utility with no concept of "what HTTP status should this
# be" — that call belongs to whoever's using the storage. Today the only
# caller (upload_service.create_upload_job) always catches this and
# re-raises it as UploadValidationError, so it never reaches a route
# directly. If a future caller forgets to catch it, it becomes an
# uncaught-exception 500 by default, which is the correct behavior for an
# internal error nobody translated on purpose.
class StorageError(Exception):
    pass


# ---------------------------------------------------------------------------
# Object storage (MinIO / any S3-compatible service)
#
# Photos live here, addressed by *object key* — not by filesystem path. See
# DEEP_DIVE.md for the full mental model; the short version is that Postgres
# stores keys, presigned URLs are minted on demand and expire, and the app
# never opens a photo with open()/imread() again.
# ---------------------------------------------------------------------------

# Lazily-built singleton client. Note this defers the *client*, not the
# import: unlike _get_deepface() in face_matcher.py (where the point is to
# keep TensorFlow out of the api process entirely), boto3 is lightweight and
# both the api and worker processes genuinely use it, so there's nothing to
# gain by deferring the import. Deferring the client means settings are read
# when first needed rather than at import time, and one connection pool gets
# reused instead of a new client per call.
_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
    return _s3_client


def ensure_bucket() -> None:
    """Create the bucket if it isn't there yet.

    Called once at startup from main.py's lifespan — the same slot as
    ensure_pgvector_extension(), and for the same reason: make sure a
    backing service is actually ready before we start serving traffic.
    """
    client = _get_s3_client()
    try:
        existing = {bucket["Name"] for bucket in client.list_buckets().get("Buckets", [])}
        if settings.s3_bucket not in existing:
            client.create_bucket(Bucket=settings.s3_bucket)
    except Exception as exc:
        raise StorageError(
            f"Could not reach object storage at {settings.s3_endpoint_url}: {exc}"
        ) from exc


def unique_filename(original_filename: str) -> str:
    """A collision-proof name that keeps the original extension — same idea
    as the UUID filenames the on-disk version used, so two people uploading
    `IMG_1234.jpg` never overwrite each other."""
    return f"{uuid4().hex}{Path(original_filename or '').suffix.lower()}"


def build_object_key(job_id: str, category: str, filename: str) -> str:
    """`jobs/{job_id}/{category}/{filename}` — deliberately mirrors the old
    on-disk layout (selfie/, archive/, event_photos/) so the mapping from
    "where it used to live" to "what its key is" stays obvious."""
    return f"jobs/{job_id}/{category}/{filename}"


def put_object(key: str, data: bytes, content_type: str | None = None) -> None:
    if content_type is None:
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
    try:
        _get_s3_client().put_object(
            Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type
        )
    except Exception as exc:
        raise StorageError(f"Could not store object '{key}': {exc}") from exc


def get_object(key: str) -> bytes:
    try:
        response = _get_s3_client().get_object(Bucket=settings.s3_bucket, Key=key)
        return response["Body"].read()
    except Exception as exc:
        raise StorageError(f"Could not read object '{key}': {exc}") from exc


def object_exists(key: str) -> bool:
    try:
        _get_s3_client().head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except Exception:
        return False


def delete_prefix(prefix: str) -> None:
    """Delete every object under a prefix — the object-storage equivalent of
    the `shutil.rmtree(job_root)` cleanup the on-disk version did when an
    upload failed partway through."""
    client = _get_s3_client()
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
            keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if keys:
                client.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": keys})
    except Exception as exc:
        raise StorageError(f"Could not delete objects under '{prefix}': {exc}") from exc


# --- Presigned URLs --------------------------------------------------------
# The API mints these; the client uses them to talk to object storage
# directly, so photo bytes never pass through this server during upload or
# download. They expire (PRESIGNED_URL_TTL_SECONDS) and are therefore never
# stored in Postgres — only the key is.
#
# !! UPLOAD CONTRACT — verified against MinIO, not assumed !!
# A client PUTting to a presigned upload URL must send **no Content-Type
# header at all**. The signature covers whatever Content-Type the request
# carries, so sending one that wasn't signed fails with 403.
#
# You can't fix that by signing a Content-Type either: `upload_part` doesn't
# accept ContentType as a parameter at all (botocore rejects it outright),
# so parts can never have a signed content type. Omitting the header is the
# only contract that works for both parts and single PUTs.
#
# On Android: OkHttp's `RequestBody.create(null, bytes)` sends no
# Content-Type. Watch out for HTTP clients that add one automatically —
# Python's urllib does, which is exactly how this was found.
#
# Operational gotcha: a signature mismatch on a *large* body doesn't return
# a clean 403 — the connection hangs until it times out. If part uploads
# start hanging rather than failing, suspect a stray header before anything
# else.
#
# Content types still get set correctly on stored objects, just server-side
# via put_object() below (a normal signed boto3 call, where ContentType is
# fine) — which is what matters for matched photos rendering in the app.


def presign_download(key: str, ttl_seconds: int | None = None) -> str:
    try:
        return _get_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=ttl_seconds or settings.presigned_url_ttl_seconds,
        )
    except Exception as exc:
        raise StorageError(f"Could not presign download for '{key}': {exc}") from exc


def presign_put(key: str, ttl_seconds: int | None = None) -> str:
    """Single-shot upload URL. Used for the selfie — one small file, where
    multipart would be pure overhead.

    Deliberately signs no ContentType: see the upload contract above. The
    client must PUT with no Content-Type header.
    """
    try:
        return _get_s3_client().generate_presigned_url(
            "put_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=ttl_seconds or settings.presigned_url_ttl_seconds,
        )
    except Exception as exc:
        raise StorageError(f"Could not presign upload for '{key}': {exc}") from exc


# --- Multipart upload ------------------------------------------------------
# This is what makes uploads resumable. The client uploads parts directly to
# object storage; if the connection drops, list_uploaded_parts() says which
# parts already landed, so only the missing ones get re-sent. We don't track
# byte offsets ourselves — S3's protocol already does it.


def create_multipart_upload(key: str) -> str:
    """Returns the upload_id, which must be stored (upload_jobs.zip_upload_id)
    since every later part/complete/list call needs it."""
    try:
        response = _get_s3_client().create_multipart_upload(
            Bucket=settings.s3_bucket,
            Key=key,
            ContentType=mimetypes.guess_type(key)[0] or "application/octet-stream",
        )
        return response["UploadId"]
    except Exception as exc:
        raise StorageError(f"Could not start multipart upload for '{key}': {exc}") from exc


def presign_upload_part(
    key: str, upload_id: str, part_number: int, ttl_seconds: int | None = None
) -> str:
    try:
        return _get_s3_client().generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": settings.s3_bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=ttl_seconds or settings.presigned_url_ttl_seconds,
        )
    except Exception as exc:
        raise StorageError(
            f"Could not presign part {part_number} for '{key}': {exc}"
        ) from exc


def list_uploaded_parts(key: str, upload_id: str) -> list[dict]:
    """Which parts have actually landed — this is the resume mechanism.
    Returns [{"PartNumber": int, "ETag": str}, ...] ordered by part number."""
    client = _get_s3_client()
    try:
        parts: list[dict] = []
        paginator = client.get_paginator("list_parts")
        for page in paginator.paginate(
            Bucket=settings.s3_bucket, Key=key, UploadId=upload_id
        ):
            for part in page.get("Parts", []):
                parts.append({"PartNumber": part["PartNumber"], "ETag": part["ETag"]})
        return sorted(parts, key=lambda part: part["PartNumber"])
    except Exception as exc:
        raise StorageError(f"Could not list parts for '{key}': {exc}") from exc


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> None:
    """Tell object storage to assemble the parts into one object. `parts` is
    the {"PartNumber", "ETag"} list — normally straight from
    list_uploaded_parts(), so the client doesn't have to track ETags itself."""
    try:
        _get_s3_client().complete_multipart_upload(
            Bucket=settings.s3_bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception as exc:
        raise StorageError(f"Could not complete multipart upload for '{key}': {exc}") from exc


def abort_multipart_upload(key: str, upload_id: str) -> None:
    """Discards an unfinished upload and its parts. Without this, abandoned
    uploads keep consuming storage indefinitely."""
    try:
        _get_s3_client().abort_multipart_upload(
            Bucket=settings.s3_bucket, Key=key, UploadId=upload_id
        )
    except Exception as exc:
        raise StorageError(f"Could not abort multipart upload for '{key}': {exc}") from exc


# ---------------------------------------------------------------------------
# Local filesystem (still in use until the Phase 4 cutover, then deleted)
# ---------------------------------------------------------------------------


def ensure_upload_root() -> None:
    settings.upload_root.mkdir(parents=True, exist_ok=True)


async def save_upload_file(upload_file: UploadFile, destination_dir: Path) -> tuple[str, str]:
    original_name = Path(upload_file.filename or "").name
    extension = Path(original_name).suffix.lower()
    unique_name = f"{uuid4().hex}{extension}"

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / unique_name

    loop = asyncio.get_running_loop()
    with destination_path.open("wb") as buffer:
        await loop.run_in_executor(None, shutil.copyfileobj, upload_file.file, buffer)

    await upload_file.close()
    return original_name, str(destination_path.resolve())


def extract_zip_file(zip_path: Path, destination_dir: Path) -> list[Path]:
    """Extracts a zip to a directory, rejecting path-traversal entries.

    Survives the object-storage migration unchanged: after the cutover the
    worker downloads the zip to a temp directory and calls this exactly as
    before, then uploads each extracted photo as its own object. The
    traversal check below is the reason this is worth reusing rather than
    rewriting against a zip-in-memory API.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_root = destination_dir.resolve()
    extracted_files: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member.is_dir():
                continue

            target_path = (destination_root / member_path).resolve()
            if not target_path.is_relative_to(destination_root):
                raise StorageError("Zip file contains an invalid path")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files.append(target_path)

    return extracted_files
