from __future__ import annotations

import asyncio
import shutil
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from core.settings import settings


class StorageError(Exception):
    pass


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
