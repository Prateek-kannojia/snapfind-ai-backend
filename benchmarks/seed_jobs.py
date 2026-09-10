"""Load every test job into a clean database, once.

Uploads each job in sample_test_data/ through the real API exactly as a phone
would: init, presigned PUTs, multipart zip, complete. After this the jobs
exist as real rows in Postgres with their photos in object storage, so
evaluate_corpus.py can re-process them without pushing 165 photos through
MinIO on every run.

Run this once, or again after wiping the database. It is NOT part of a normal
measurement run.

Local folders keep their readable names. The job ids the API hands back are
minted per upload — re-seeding produces new ones — so naming folders after
them would go stale immediately. The link is `selfie_filename`, set to the job
name here and read back by _common.seeded_job_ids().

    venv\\Scripts\\python.exe benchmarks\\seed_jobs.py          # wipe, then seed
    venv\\Scripts\\python.exe benchmarks\\seed_jobs.py --keep   # seed alongside
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from _common import discover_jobs, p, seeded_job_ids  # noqa: E402

API = "http://localhost:8000"
PART_SIZE = 8 * 1024 * 1024   # S3 multipart floor is 5MB per non-final part


def wipe() -> None:
    """Empty the database and the bucket so seeding starts from nothing.

    Only test data lives here, and seeding rebuilds all of it from the photos
    on disk — but this is still a destructive call, so it is opt-out
    (--keep) rather than silent.
    """
    from sqlalchemy import create_engine, text

    from core.settings import settings
    from services import storage_service as storage

    engine = create_engine(settings.database_url)
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "TRUNCATE matched_photos, event_photos, upload_jobs "
                "RESTART IDENTITY CASCADE"
            ))
    finally:
        engine.dispose()
    p("  database: truncated")

    storage.delete_prefix("jobs/")
    p(f"  object storage: cleared jobs/ in bucket '{settings.s3_bucket}'")


def zip_bytes(photos: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for ph in photos:
            zf.write(ph, arcname=ph.name)
    return buf.getvalue()


def put(url: str, body: bytes) -> None:
    """No Content-Type header — it is part of the signed string, and sending
    one the server did not sign gets a 403, or on a large body, a hang."""
    r = requests.put(url, data=body, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"PUT failed {r.status_code}: {r.text[:300]}")


def upload(job: dict) -> tuple[str, int]:
    """Init, PUT the selfie and every zip part, complete. Returns (id, parts)."""
    payload = zip_bytes(job["photos"])
    parts = [payload[i:i + PART_SIZE] for i in range(0, len(payload), PART_SIZE)] or [b""]

    r = requests.post(f"{API}/jobs/upload/init", json={
        "selfie_filename": f"{job['name']}.jpg",   # carries the job identity through
        "zip_filename": f"{job['name']}.zip",
        "part_count": len(parts),
    }, timeout=60)
    r.raise_for_status()
    init = r.json()

    put(init["selfie_upload_url"], job["selfie"].read_bytes())
    for entry in sorted(init["zip_upload_urls"], key=lambda e: e["part_number"]):
        put(entry["url"], parts[entry["part_number"] - 1])

    r = requests.post(f"{API}/jobs/{init['job_id']}/upload/complete", timeout=120)
    r.raise_for_status()
    return init["job_id"], len(parts)


def main() -> None:
    jobs = discover_jobs()
    try:
        requests.get(f"{API}/docs", timeout=10).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"API not reachable at {API} ({exc}) — is docker compose up?")

    if "--keep" in sys.argv:
        p("keeping existing jobs (--keep)\n")
    else:
        p("wiping existing data...")
        wipe()
        p("")

    p(f"seeding {len(jobs)} jobs through {API}\n")
    started = time.time()
    for job in jobs:
        job_id, parts = upload(job)
        p(f"  {job['name']:<24} -> {job_id}  ({len(job['photos'])} photos, "
          f"{parts} zip part{'s' if parts != 1 else ''})")

    p(f"\nseeded in {time.time() - started:.0f}s")
    resolved = seeded_job_ids()
    missing = [j["name"] for j in jobs if j["name"] not in resolved]
    if missing:
        raise SystemExit(f"seeded but not resolvable by name: {missing}")
    p(f"all {len(jobs)} jobs resolve by name — evaluate_corpus.py can run now")
    p("Photos are not extracted yet; the worker does that on the first /process.")


if __name__ == "__main__":
    main()
