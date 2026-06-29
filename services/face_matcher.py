from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from deepface import DeepFace
from sqlalchemy.orm import Session

from core.settings import settings
from db.orm_models import EventPhoto, MatchedPhoto, UploadJob

# Set once at import time so every DeepFace call uses the right home directory.
settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)

DEFAULT_MODEL = "ArcFace"


class FaceMatchError(Exception):
    pass


class SelfieFaceNotDetectedError(FaceMatchError):
    pass


class _Photo(NamedTuple):
    id: int
    storage_path: str
    embedding: str | None  # JSON string of 512 floats, or None if not yet computed


def _deserialize_embedding(raw: str) -> list[float]:
    return json.loads(raw)


def _serialize_embedding(embedding: list[float]) -> str:
    return json.dumps(embedding)


def _cosine_distance(source: list[float], target: list[float]) -> float:
    dot_product = sum(left * right for left, right in zip(source, target))
    source_norm = math.sqrt(sum(v * v for v in source))
    target_norm = math.sqrt(sum(v * v for v in target))

    if source_norm == 0 or target_norm == 0:
        raise FaceMatchError("Unable to compare face embeddings")

    similarity = dot_product / (source_norm * target_norm)
    return 1 - max(min(similarity, 1.0), -1.0)


def _validate_image_path(image_path: str) -> None:
    path = Path(image_path)
    if not path.exists():
        raise FaceMatchError(f"Image file not found: {path}")
    if path.suffix.lower() not in settings.allowed_image_extensions:
        raise FaceMatchError(
            "Image must have one of these extensions: "
            + ", ".join(sorted(settings.allowed_image_extensions))
        )


def _embedding_for_selfie(image_path: str) -> list[float]:
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=DEFAULT_MODEL,
            detector_backend=settings.selfie_detector,
            enforce_detection=True,
        )
    except Exception as exc:
        raise SelfieFaceNotDetectedError(
            "Could not detect a clear face in the uploaded selfie. "
            "Please upload a clearer front-facing photo."
        ) from exc

    if not result:
        raise SelfieFaceNotDetectedError(
            "Could not generate a face embedding from the uploaded selfie. "
            "Please upload a clearer front-facing photo."
        )
    return result[0]["embedding"]


def _embedding_for_event_photo(image_path: str) -> list[float]:
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=DEFAULT_MODEL,
            detector_backend=settings.event_photo_detector,
            enforce_detection=True,
        )
    except Exception as exc:
        raise FaceMatchError("Could not detect a face in one of the event photos") from exc

    if not result:
        raise FaceMatchError("Could not generate a face embedding for an event photo")
    return result[0]["embedding"]


def _embedding_for_image(image_path: str, *, is_selfie: bool) -> list[float]:
    _validate_image_path(image_path)
    if is_selfie:
        return _embedding_for_selfie(image_path)
    return _embedding_for_event_photo(image_path)


def _process_single_photo(
    selfie_embedding: list[float], photo: _Photo
) -> tuple[int, float | None, str | None]:
    try:
        if photo.embedding is not None:
            image_embedding = _deserialize_embedding(photo.embedding)
            new_embedding_to_save = None
        else:
            image_embedding = _embedding_for_image(photo.storage_path, is_selfie=False)
            new_embedding_to_save = _serialize_embedding(image_embedding)

        distance = _cosine_distance(selfie_embedding, image_embedding)
        return photo.id, distance, new_embedding_to_save
    except FaceMatchError:
        return photo.id, None, None


def build_matches_for_job(
    db: Session, job: UploadJob, event_photos: list[EventPhoto], threshold: float
) -> list[MatchedPhoto]:
    if threshold <= 0:
        raise FaceMatchError("Threshold must be greater than 0")

    selfie_embedding = _embedding_for_image(job.selfie_storage_path, is_selfie=True)

    photos = [_Photo(p.id, p.storage_path, p.embedding) for p in event_photos]
    orm_by_id = {p.id: p for p in event_photos}

    matches: list[MatchedPhoto] = []
    embeddings_to_save: dict[int, str] = {}
    worker_count = min(settings.face_match_workers, max(1, len(photos)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_single_photo, selfie_embedding, photo): photo
            for photo in photos
        }
        for future in as_completed(futures):
            photo_id, distance, new_embedding = future.result()
            if new_embedding is not None:
                embeddings_to_save[photo_id] = new_embedding
            if distance is not None and distance <= threshold:
                matches.append(
                    MatchedPhoto(
                        job_id=job.id,
                        event_photo_id=photo_id,
                        match_distance=round(distance, 4),
                    )
                )

    for photo_id, emb in embeddings_to_save.items():
        orm_by_id[photo_id].embedding = emb
    if embeddings_to_save:
        db.flush()

    matches.sort(key=lambda m: m.match_distance)
    return matches
