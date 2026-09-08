from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from core.errors import AppError
from core.settings import settings
from db.orm_models import EventPhoto, MatchedPhoto, UploadJob
from services import storage_service as storage

# Set once at import time so every DeepFace call uses the right home directory.
settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)
settings.insightface_home.mkdir(parents=True, exist_ok=True)
os.environ["INSIGHTFACE_HOME"] = str(settings.insightface_home)

DEFAULT_MODEL = "ArcFace"

# Lazily-built singleton. Built once (in build_matches_for_job, before the
# thread pool starts, same warm-up pattern as the selfie embedding) and then
# read-only from worker threads — onnxruntime InferenceSession.run() is
# documented thread-safe for concurrent calls, so sharing one instance across
# threads is safe once it's built.
_insightface_app = None


def _get_insightface_app():
    global _insightface_app
    if _insightface_app is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_sc",  # smallest/fastest pack: SCRFD-500MF detector
            allowed_modules=["detection"],  # we only use detection; embedding stays DeepFace/ArcFace
            root=str(settings.insightface_home),
        )
        app.prepare(ctx_id=-1, det_size=(settings.event_photo_max_dimension,) * 2)  # ctx_id=-1 = CPU
        _insightface_app = app
    return _insightface_app


# Same lazy-import reasoning as _get_insightface_app() above: deepface drags
# in TensorFlow (multi-second import, ~1GB memory). The `api` container's
# process imports this module too (main.py -> api/routes.py ->
# services/job_service.py -> here), but never actually calls anything that
# needs deepface — only the `worker` process does. A module-level `import
# deepface` would make every API server startup pay that cost for nothing.
# One accessor here instead of `from deepface import DeepFace` repeated in
# every function that needs it — same pattern, not duplicated twice.
_deepface_module = None


def _get_deepface():
    global _deepface_module
    if _deepface_module is None:
        from deepface import DeepFace

        _deepface_module = DeepFace
    return _deepface_module


class FaceMatchError(AppError):
    """Base for face-matching errors. In practice these are only ever raised
    inside the worker process (run_job_processing -> build_matches_for_job),
    which catches Exception broadly and stores the message on the job's
    last_error column rather than letting it become a live HTTP response —
    the client learns about it by polling GET /jobs/{id}, not from a request
    that raised this directly. Still an AppError (status_code/error_code,
    422 by default: the image was readable but no usable face was found in
    it) so it's ready to cross an HTTP boundary directly the moment any
    route calls into this module synchronously — e.g. a future "validate
    this selfie before upload" endpoint."""

    def __init__(self, message: str, *, status_code: int = 422, error_code: str = "face_match_error") -> None:
        super().__init__(message, status_code=status_code, error_code=error_code)


class SelfieFaceNotDetectedError(FaceMatchError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422, error_code="selfie_face_not_detected")


class _Photo(NamedTuple):
    id: int
    object_key: str
    embedding: Any | None  # a pgvector value — see db/orm_models.py


def _cosine_distance(source: list[float], target: list[float]) -> float:
    dot_product = sum(left * right for left, right in zip(source, target))
    source_norm = math.sqrt(sum(v * v for v in source))
    target_norm = math.sqrt(sum(v * v for v in target))

    if source_norm == 0 or target_norm == 0:
        raise FaceMatchError("Unable to compare face embeddings")

    similarity = dot_product / (source_norm * target_norm)
    return 1 - max(min(similarity, 1.0), -1.0)


def _validate_image_key(object_key: str) -> None:
    """Extension check only — existence is proven by the fetch itself."""
    if Path(object_key).suffix.lower() not in settings.allowed_image_extensions:
        raise FaceMatchError(
            "Image must have one of these extensions: "
            + ", ".join(sorted(settings.allowed_image_extensions))
        )


def _load_resized_image(object_key: str, max_dimension: int) -> np.ndarray:
    """Fetch an image from object storage and shrink it to max_dimension.

    Detection/embedding cost scales with pixel count, and phone photos are
    far larger than a face needs — this is the biggest CPU win available.
    """
    try:
        data = storage.get_object(object_key)
    except storage.StorageError as exc:
        raise FaceMatchError(f"Could not read image '{object_key}': {exc}") from exc

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FaceMatchError(f"Could not decode image '{object_key}'")

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_dimension:
        scale = max_dimension / longest_side
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return image


def _event_photo_embedding(image_array: np.ndarray) -> list[float]:
    """Detect + align with a lightweight ONNX detector (insightface SCRFD),
    then embed with DeepFace's ArcFace via detector_backend="skip" since the
    face is already cropped and aligned to the standard ArcFace convention.
    """
    from insightface.utils import face_align

    app = _get_insightface_app()
    faces = app.get(image_array)
    if not faces:
        raise FaceMatchError("Could not detect a face in one of the event photos")

    # Largest face = most prominent person in the photo. Matches how a
    # single event photo is judged: is the target person visibly in it.
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aligned = face_align.norm_crop(image_array, best.kps, image_size=112, mode="arcface")

    result = _get_deepface().represent(
        img_path=aligned, model_name=DEFAULT_MODEL, detector_backend="skip", enforce_detection=False
    )
    if not result:
        raise FaceMatchError("Could not generate a face embedding for an event photo")
    return result[0]["embedding"]


def _selfie_embedding(image_array: np.ndarray) -> list[float]:
    """Selfie path: DeepFace + mtcnn, unchanged since the MVP."""
    try:
        result = _get_deepface().represent(
            img_path=image_array,
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


def _embedding_for_image(object_key: str, *, is_selfie: bool) -> list[float]:
    _validate_image_key(object_key)

    max_dimension = (
        settings.selfie_max_dimension if is_selfie else settings.event_photo_max_dimension
    )
    image_array = _load_resized_image(object_key, max_dimension)

    if is_selfie:
        return _selfie_embedding(image_array)

    try:
        return _event_photo_embedding(image_array)
    except FaceMatchError:
        raise
    except Exception as exc:
        raise FaceMatchError("Could not detect a face in one of the event photos") from exc


def _embed_and_score_event_photo(
    selfie_embedding: list[float], photo: _Photo
) -> tuple[int, float | None, Any | None]:
    try:
        if photo.embedding is not None:
            image_embedding = list(photo.embedding)  # cached
            new_embedding_to_save = None
        else:
            image_embedding = _embedding_for_image(photo.object_key, is_selfie=False)
            new_embedding_to_save = image_embedding

        distance = _cosine_distance(selfie_embedding, image_embedding)
        return photo.id, distance, new_embedding_to_save
    except FaceMatchError:
        return photo.id, None, None


def build_matches_for_job(
    db: Session, job: UploadJob, event_photos: list[EventPhoto], threshold: float
) -> list[MatchedPhoto]:
    if threshold <= 0:
        raise FaceMatchError("Threshold must be greater than 0")

    # Compute selfie embedding first — this also warms the DeepFace model in
    # memory so all worker threads find it already loaded.
    selfie_embedding = _embedding_for_image(job.selfie_object_key, is_selfie=True)

    # Same warm-up idea for the insightface detector used for event photos:
    # build it once here (single-threaded) so worker threads only ever read
    # from the already-built singleton, never race to build it concurrently.
    _get_insightface_app()

    photos = [_Photo(p.id, p.object_key, p.embedding) for p in event_photos]
    orm_by_id = {p.id: p for p in event_photos}

    matches: list[MatchedPhoto] = []
    embeddings_to_save: dict[int, Any] = {}
    worker_count = min(settings.face_match_workers, max(1, len(photos)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_embed_and_score_event_photo, selfie_embedding, photo): photo
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
