from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        default_deepface_home = base_dir / "storage" / "deepface"
        default_insightface_home = base_dir / "storage" / "insightface"

        self.app_name = os.getenv("APP_NAME", "Event Photo Finder API")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        # Postgres only — the embedding column is a real pgvector type, so
        # there's no second backend to fall back to. docker-compose points
        # this at its own postgres service.
        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://snapfind:snapfind@localhost:5432/snapfind",
        )
        self.deepface_home = Path(
            os.getenv("DEEPFACE_HOME", str(default_deepface_home))
        ).resolve()
        self.insightface_home = Path(
            os.getenv("INSIGHTFACE_HOME", str(default_insightface_home))
        ).resolve()
        self.max_event_photos = int(os.getenv("MAX_EVENT_PHOTOS", "500"))
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.selfie_detector = os.getenv("SELFIE_DETECTOR", "mtcnn")
        # Event photos are always detected with insightface/SCRFD now — see
        # services/face_matcher.py and README "Event-photo detector history".
        # Two earlier detectors (opencv, retinaface) were tried and replaced;
        # they're preserved as a runnable comparison in
        # benchmarks/detector_comparison.py at the repo root, not here — this
        # settings module only configures what production actually uses.
        self.face_match_workers = int(os.getenv("FACE_MATCH_WORKERS", "4"))
        # Phone photos are commonly 3000-4000px on the long side. Detection and
        # embedding cost scale with pixel count, so we downscale before handing
        # the image to DeepFace. Selfie keeps a higher ceiling since it is a
        # single image and accuracy matters most there; event photos are
        # downscaled harder since there are many of them and speed matters more.
        self.selfie_max_dimension = int(os.getenv("SELFIE_MAX_DIMENSION", "1024"))
        self.event_photo_max_dimension = int(os.getenv("EVENT_PHOTO_MAX_DIMENSION", "800"))
        # SCRFD's input size. Deliberately NOT tied to the downscale settings
        # above: measured, det_size=3200 makes the detector lose large faces
        # entirely (a 588,187px2 face becomes 730px2), because its anchor
        # scales stop matching. Detection is stable at 640-1600 regardless of
        # how big the source image is, so this stays pinned.
        self.face_detector_size = int(os.getenv("FACE_DETECTOR_SIZE", "800"))
        # Faces are cropped from the ORIGINAL image, not the downscaled one.
        # Measured: two different people 0.1596 apart on 17px crops (a false
        # match at threshold 0.68) vs 0.7144 on 73px crops. Detection is cheap
        # at low resolution; embedding needs real pixels.
        self.crop_from_original = os.getenv("CROP_FROM_ORIGINAL", "true").lower() != "false"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.rq_queue_name = os.getenv("RQ_QUEUE_NAME", "face-matching")
        self.rq_job_timeout_seconds = int(os.getenv("RQ_JOB_TIMEOUT_SECONDS", "1800"))
        self.rq_worker_class = os.getenv(
            "RQ_WORKER_CLASS", "simple" if os.name == "nt" else "default"
        )
        self.job_stale_after_seconds = int(os.getenv("JOB_STALE_AFTER_SECONDS", "3600"))
        # Object storage. MinIO locally (docker-compose), any S3-compatible
        # service in production — boto3 talks to both identically, so moving
        # to real AWS S3 is an endpoint + credentials change, not a code
        # change. Photos live here as objects; Postgres stores only their
        # keys (never presigned URLs — those are minted on demand and expire).
        self.s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        # Where *clients* reach storage. Differs from the above whenever the
        # server talks to storage on an internal address the client can't
        # resolve (in Docker: minio:9000 internally, localhost:9000 outside).
        # Presigned URLs are built from this; defaults to the internal one.
        self.s3_public_endpoint_url = os.getenv(
            "S3_PUBLIC_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        )
        self.s3_access_key = os.getenv("S3_ACCESS_KEY", "snapfind")
        self.s3_secret_key = os.getenv("S3_SECRET_KEY", "snapfind123")
        self.s3_bucket = os.getenv("S3_BUCKET", "snapfind")
        # How long a presigned download URL stays valid. Short on purpose —
        # the client re-fetches /jobs/{id}/matches for fresh URLs if an image
        # fails to load, so there's no need for long-lived signed links.
        self.presigned_url_ttl_seconds = int(os.getenv("PRESIGNED_URL_TTL_SECONDS", "3600"))


settings = Settings()
