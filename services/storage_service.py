from __future__ import annotations

import mimetypes
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

import boto3

from core.settings import settings


# Plain Exception, not an AppError: storage has no business deciding HTTP
# status codes. Its one caller catches and translates it.
class StorageError(Exception):
    pass


# --- Object storage (MinIO / any S3-compatible service) --------------------
# Photos are addressed by object key, not filesystem path. See DEEP_DIVE.md.

# Lazy singleton so settings are read on first use and one connection pool
# is reused. (Only the client is deferred — boto3 itself is cheap to import.)
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


# Separate client bound to the client-reachable endpoint. Used only to sign
# URLs — the server's own endpoint may be internal (minio:9000) and unusable
# by a phone or browser. Signing never opens a connection, so this is cheap.
_presign_client = None


def _get_presign_client():
    global _presign_client
    if settings.s3_public_endpoint_url == settings.s3_endpoint_url:
        return _get_s3_client()
    if _presign_client is None:
        _presign_client = boto3.client(
            "s3",
            endpoint_url=settings.s3_public_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
    return _presign_client


def ensure_bucket() -> None:
    """Create the bucket if missing. Called at startup, like ensure_pgvector_extension()."""
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
    """UUID name keeping the original extension, so uploads can't collide."""
    return f"{uuid4().hex}{Path(original_filename or '').suffix.lower()}"


def build_object_key(job_id: str, category: str, filename: str) -> str:
    """`jobs/{job_id}/{category}/{filename}` — mirrors the old on-disk layout."""
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
    """Delete everything under a prefix — the rmtree equivalent for cleanup."""
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
# Minted on demand, expire, never stored in Postgres — only keys are.
#
# Upload contract: the client must send NO Content-Type header. SigV2 signs
# it as a fixed field, so an unsigned one fails (403, or a hang on large
# bodies). Signing it isn't an option either — upload_part rejects the
# parameter. Full reasoning in DEEP_DIVE.md.


def presign_download(key: str, ttl_seconds: int | None = None) -> str:
    try:
        return _get_presign_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=ttl_seconds or settings.presigned_url_ttl_seconds,
        )
    except Exception as exc:
        raise StorageError(f"Could not presign download for '{key}': {exc}") from exc


def presign_put(key: str, ttl_seconds: int | None = None) -> str:
    """Single-shot upload URL, used for the selfie (multipart would be overkill)."""
    try:
        return _get_presign_client().generate_presigned_url(
            "put_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=ttl_seconds or settings.presigned_url_ttl_seconds,
        )
    except Exception as exc:
        raise StorageError(f"Could not presign upload for '{key}': {exc}") from exc


# --- Multipart upload ------------------------------------------------------
# What makes uploads resumable: list_uploaded_parts() says which parts
# already landed, so only missing ones are re-sent. S3 tracks offsets, not us.


def create_multipart_upload(key: str) -> str:
    """Returns the upload_id — stored on upload_jobs, needed by every later call."""
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
        return _get_presign_client().generate_presigned_url(
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
    """Parts that landed, as [{"PartNumber", "ETag"}, ...]. The resume mechanism."""
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
    """Assemble the parts into one object. `parts` comes from list_uploaded_parts()."""
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
    """Discard an unfinished upload; otherwise its parts occupy storage forever."""
    try:
        _get_s3_client().abort_multipart_upload(
            Bucket=settings.s3_bucket, Key=key, UploadId=upload_id
        )
    except Exception as exc:
        raise StorageError(f"Could not abort multipart upload for '{key}': {exc}") from exc


# --- Zip handling ----------------------------------------------------------


def extract_zip_file(zip_path: Path, destination_dir: Path) -> list[Path]:
    """Extract a zip, rejecting path-traversal entries.

    The worker downloads the zip to a temp dir, calls this, then uploads each
    photo as its own object. The temp dir is the only local disk we still use.
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
