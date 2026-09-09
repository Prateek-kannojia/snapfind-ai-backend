"""Run the REAL production matching path against the local sample jobs.

Why this exists: `embedder_comparison.py` reimplements the production pipeline
step by step so it can swap one piece at a time. That's the right shape for a
comparison, but it means its numbers are only as trustworthy as the
reimplementation. This script removes that doubt — it imports
`services/face_matcher.py` and calls the actual production functions:

    fm._selfie_embedding()        <- mtcnn, exactly as production does it
    fm._event_photo_embedding()   <- SCRFD detect + align + DeepFace ArcFace
    fm._cosine_distance()         <- production's own distance function

Nothing is reimplemented here except reading a file off disk and downscaling
it, which production does in `_load_resized_image()` — that function is
skipped only because it fetches from S3/MinIO, which would need a running
container. Same resize maths, same `settings.*_max_dimension` values.

The only thing not exercised is `build_matches_for_job()`, which adds the
thread pool, the embedding cache and the DB writes around these calls. Those
affect speed and persistence, not which photos come back.

For contrast it also runs the candidate on-device stack over the same jobs:
det_500m.onnx + w600k_mbf.onnx, all faces scored rather than just the
largest. Both pipelines see identical input files.

Usage (from the Face_recognition/ repo root):
    venv\\Scripts\\python.exe benchmarks\\verify_production.py

Reads every job directory under storage/uploads/ that has both a selfie/ and
an event_photos/ folder. Needs no Postgres, no Redis and no MinIO.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.settings import settings  # noqa: E402

settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)
settings.insightface_home.mkdir(parents=True, exist_ok=True)
os.environ["INSIGHTFACE_HOME"] = str(settings.insightface_home)

import cv2  # noqa: E402

UPLOADS = BACKEND_ROOT / "storage" / "uploads"
RESULTS_PATH = Path(__file__).resolve().parent / "verify_production_results.json"

# Production's threshold, from api/routes.py:74.
PRODUCTION_THRESHOLD = 0.68
# Candidate threshold for w600k_mbf, derived in embedder_comparison.py stage 8.
# From ONE identity's worth of labelled pairs — provisional, not calibrated.
MBF_THRESHOLD = 0.74


def p(msg: str = "") -> None:
    print(msg, flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_resized(path: Path, max_dim: int):
    """Same maths as face_matcher._load_resized_image, minus the S3 fetch."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not decode {path}")
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    return image


def find_jobs() -> list[Path]:
    if not UPLOADS.exists():
        raise SystemExit(f"No uploads directory at {UPLOADS}")
    jobs = [
        d for d in sorted(UPLOADS.iterdir())
        if d.is_dir() and (d / "selfie").is_dir() and (d / "event_photos").is_dir()
        and any((d / "event_photos").iterdir())
    ]
    if not jobs:
        raise SystemExit(f"No job directories with photos under {UPLOADS}")
    return jobs


def run_production(fm, selfie: Path, photos: list[Path]) -> dict:
    """The actual production path. Failures are reported, not swallowed —
    production skips undetectable photos by design, so a skip is a 'no match'
    for our purposes, but we want to see it rather than infer it."""
    selfie_emb = fm._selfie_embedding(load_resized(selfie, settings.selfie_max_dimension))
    rows = []
    for photo in photos:
        try:
            emb = fm._event_photo_embedding(
                load_resized(photo, settings.event_photo_max_dimension))
            dist = fm._cosine_distance(selfie_emb, emb)
            rows.append({"photo": photo.name, "distance": dist,
                         "matched": dist <= PRODUCTION_THRESHOLD})
        except Exception as exc:  # noqa: BLE001 - mirrors production's broad catch
            rows.append({"photo": photo.name, "distance": None,
                         "matched": False, "error": type(exc).__name__})
    return {"threshold": PRODUCTION_THRESHOLD, "rows": rows,
            "matched": sum(r["matched"] for r in rows)}


def run_on_device(app, sess, selfie: Path, photos: list[Path]) -> dict:
    """Candidate on-device stack: same ONNX detector, ONNX embedder instead of
    TensorFlow, and every detected face scored rather than only the largest."""
    from embedder_comparison import all_faces, cosine_distance, embed_mbf

    selfie_faces = all_faces(app, load_resized(selfie, settings.selfie_max_dimension))
    if not selfie_faces:
        return {"threshold": MBF_THRESHOLD, "rows": [], "matched": 0,
                "error": "no face in selfie"}
    selfie_emb = embed_mbf(sess, selfie_faces[0][0])

    rows = []
    for photo in photos:
        faces = all_faces(app, load_resized(photo, settings.event_photo_max_dimension))
        if not faces:
            rows.append({"photo": photo.name, "faces": 0, "distance": None,
                         "matched": False})
            continue
        best = min(cosine_distance(selfie_emb, embed_mbf(sess, c)) for c, _, _ in faces)
        rows.append({"photo": photo.name, "faces": len(faces), "distance": best,
                     "matched": best <= MBF_THRESHOLD})
    return {"threshold": MBF_THRESHOLD, "rows": rows,
            "matched": sum(r["matched"] for r in rows)}


def main() -> None:
    from embedder_comparison import build_detector, make_mbf_session
    from services import face_matcher as fm

    jobs = find_jobs()
    p(f"Production module : {fm.__file__}")
    p(f"Selfie detector   : {settings.selfie_detector}")
    p(f"Embedding model   : {fm.DEFAULT_MODEL}")
    p(f"Selfie max dim    : {settings.selfie_max_dimension}")
    p(f"Event max dim     : {settings.event_photo_max_dimension}")
    p(f"Threshold         : {PRODUCTION_THRESHOLD}")
    p(f"Jobs found        : {len(jobs)}")

    app = build_detector()
    sess = make_mbf_session()

    out = []
    for job in jobs:
        selfie = sorted(f for f in (job / "selfie").iterdir() if f.is_file())[0]
        photos = sorted(f for f in (job / "event_photos").iterdir() if f.is_file())

        p("\n" + "=" * 72)
        p(f"JOB {job.name}")
        p(f"  selfie: {selfie.name}  sha256:{digest(selfie)}")
        p(f"  photos: {len(photos)}")

        prod = run_production(fm, selfie, photos)
        p(f"\n  PRODUCTION (mtcnn selfie + DeepFace ArcFace, threshold {PRODUCTION_THRESHOLD})")
        p(f"    {'photo':<38}{'distance':>10}{'match':>8}")
        for r in prod["rows"]:
            d = f"{r['distance']:.4f}" if r["distance"] is not None else r.get("error", "ERROR")
            p(f"    {r['photo'][:36]:<38}{d:>10}{('YES' if r['matched'] else 'no'):>8}")
        p(f"    -> {prod['matched']} of {len(photos)} matched")

        dev = run_on_device(app, sess, selfie, photos)
        p(f"\n  ON-DEVICE STACK (det_500m + w600k_mbf, all faces, threshold {MBF_THRESHOLD})")
        p(f"    {'photo':<38}{'faces':>6}{'distance':>10}{'match':>8}")
        for r in dev["rows"]:
            d = f"{r['distance']:.4f}" if r["distance"] is not None else "-"
            p(f"    {r['photo'][:36]:<38}{r['faces']:>6}{d:>10}"
              f"{('YES' if r['matched'] else 'no'):>8}")
        p(f"    -> {dev['matched']} of {len(photos)} matched")

        out.append({"job": job.name, "selfie": selfie.name,
                    "selfie_sha256_12": digest(selfie), "photo_count": len(photos),
                    "production": prod, "on_device": dev})

    p("\n" + "=" * 72)
    p("SUMMARY")
    p(f"  {'job':<40}{'production':>12}{'on-device':>12}")
    for r in out:
        p(f"  {r['job']:<40}{str(r['production']['matched']) + '/' + str(r['photo_count']):>12}"
          f"{str(r['on_device']['matched']) + '/' + str(r['photo_count']):>12}")

    RESULTS_PATH.write_text(json.dumps({
        "production_threshold": PRODUCTION_THRESHOLD,
        "mbf_threshold": MBF_THRESHOLD,
        "selfie_detector": settings.selfie_detector,
        "event_photo_max_dimension": settings.event_photo_max_dimension,
        "selfie_max_dimension": settings.selfie_max_dimension,
        "jobs": out,
    }, indent=2))
    p(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
