from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        default_upload_dir = base_dir / "storage" / "uploads"
        default_database_path = base_dir / "app.db"
        default_deepface_home = base_dir / "storage" / "deepface"

        self.app_name = os.getenv("APP_NAME", "Event Photo Finder API")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        self.database_url = os.getenv(
            "DATABASE_URL", f"sqlite:///{default_database_path.as_posix()}"
        )
        self.upload_root = Path(os.getenv("UPLOAD_ROOT", str(default_upload_dir))).resolve()
        self.deepface_home = Path(
            os.getenv("DEEPFACE_HOME", str(default_deepface_home))
        ).resolve()
        self.max_event_photos = int(os.getenv("MAX_EVENT_PHOTOS", "500"))
        self.allowed_image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.selfie_detector = os.getenv("SELFIE_DETECTOR", "mtcnn")
        self.event_photo_detector = os.getenv("EVENT_PHOTO_DETECTOR", "opencv")
        self.face_match_workers = int(os.getenv("FACE_MATCH_WORKERS", "4"))


settings = Settings()
