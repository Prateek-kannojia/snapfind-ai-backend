"""Runnable comparison of every face detector this project tried, in order.

Why this file exists: services/face_matcher.py only contains the current
production path (mtcnn for the selfie, insightface/SCRFD for event photos).
The two detectors it replaced along the way (opencv Haar cascade, then
retinaface) are not dead — they're preserved here, fully runnable, so the
evolution of the project (and the actual measured reasons for each swap) can
be demonstrated and reproduced instead of just described in a README.

What this measures, against the real sample photos already sitting in the
backend's storage/uploads/ from manual testing:
  1. Detection accuracy  - did each detector find a face at all?
  2. Detection speed     - how long does each detector take per photo?
  3. Embedding agreement - for the two detectors that actually work
                            (retinaface, insightface), do they produce the
                            same ArcFace embedding for the same face? This is
                            the check that answers "is the fast one actually
                            as accurate, or just fast" - see README.md in
                            this folder for why that distinction matters.

Requires this backend's virtual environment (this script only uses
dependencies already listed in requirements.txt).

Usage (from the Face_recognition/ repo root):
    venv\\Scripts\\python.exe benchmarks\\detector_comparison.py [job_id]

If job_id is omitted, the first job directory with event photos is used.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import os  # noqa: E402

from core.settings import settings  # noqa: E402

# Must be set before importing deepface/insightface, same as
# services/face_matcher.py does — otherwise both libraries fall back to
# their own default cache dirs (e.g. ~/.deepface) instead of this project's
# already-downloaded weights, and re-download everything from scratch.
settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)
settings.insightface_home.mkdir(parents=True, exist_ok=True)
os.environ["INSIGHTFACE_HOME"] = str(settings.insightface_home)

import cv2  # noqa: E402

MODEL = "ArcFace"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def p(msg: str) -> None:
    print(msg, flush=True)


def find_job_dir(job_id: str | None) -> Path:
    uploads = settings.upload_root
    if job_id:
        job_dir = uploads / job_id
        if not job_dir.exists():
            raise SystemExit(f"No such job directory: {job_dir}")
        return job_dir
    for candidate in sorted(uploads.iterdir()):
        photos = list((candidate / "event_photos").glob("*")) if (candidate / "event_photos").exists() else []
        if photos:
            return candidate
    raise SystemExit(f"No job with event photos found under {uploads}")


def resize(image_path: Path, max_dim: int):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 1 - max(min(dot / (na * nb), 1.0), -1.0)


# --- Detector 1: opencv (Haar cascade) - the ORIGINAL default, replaced ---
def run_opencv(photos: list[Path]) -> dict:
    from deepface import DeepFace

    p("\n=== Detector 1/3: opencv (Haar cascade) — the original default ===")
    p("Weak/no-deep-learning detector, kept only for historical comparison.")
    DeepFace.represent(img_path=resize(photos[0], settings.event_photo_max_dimension),
                        model_name=MODEL, detector_backend="opencv", enforce_detection=False)

    detected, total_time, embeddings = 0, 0.0, {}
    for photo in photos:
        img = resize(photo, settings.event_photo_max_dimension)
        start = time.perf_counter()
        try:
            result = DeepFace.represent(img_path=img, model_name=MODEL,
                                         detector_backend="opencv", enforce_detection=True)
            elapsed = time.perf_counter() - start
            detected += 1
            embeddings[photo.name] = result[0]["embedding"]
            p(f"  {photo.name}: face detected, {elapsed*1000:.0f}ms")
        except Exception:
            elapsed = time.perf_counter() - start
            p(f"  {photo.name}: NO FACE DETECTED, {elapsed*1000:.0f}ms")
        total_time += elapsed
    return {"name": "opencv", "detected": detected, "total": len(photos),
            "total_time_s": total_time, "embeddings": embeddings}


# --- Detector 2: retinaface (ResNet50 backbone, via DeepFace) - 2nd choice ---
def run_retinaface(photos: list[Path]) -> dict:
    from deepface import DeepFace

    p("\n=== Detector 2/3: retinaface (ResNet50 backbone, via DeepFace) ===")
    p("Fixed opencv's accuracy problem, but slow on CPU.")
    DeepFace.represent(img_path=resize(photos[0], settings.event_photo_max_dimension),
                        model_name=MODEL, detector_backend="retinaface", enforce_detection=False)

    detected, total_time, embeddings = 0, 0.0, {}
    for photo in photos:
        img = resize(photo, settings.event_photo_max_dimension)
        start = time.perf_counter()
        try:
            result = DeepFace.represent(img_path=img, model_name=MODEL,
                                         detector_backend="retinaface", enforce_detection=True)
            elapsed = time.perf_counter() - start
            detected += 1
            embeddings[photo.name] = result[0]["embedding"]
            p(f"  {photo.name}: face detected, {elapsed*1000:.0f}ms")
        except Exception:
            elapsed = time.perf_counter() - start
            p(f"  {photo.name}: NO FACE DETECTED, {elapsed*1000:.0f}ms")
        total_time += elapsed
    return {"name": "retinaface", "detected": detected, "total": len(photos),
            "total_time_s": total_time, "embeddings": embeddings}


# --- Detector 3: insightface/SCRFD-500MF - current production default ---
def run_insightface(photos: list[Path]) -> dict:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from deepface import DeepFace

    p("\n=== Detector 3/3: insightface / SCRFD-500MF (current production default) ===")
    p("Same weight class as 'RetinaFace-MobileNet-0.25'. Lightweight ONNX model.")
    app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"], root=str(settings.insightface_home))
    app.prepare(ctx_id=-1, det_size=(settings.event_photo_max_dimension,) * 2)

    detected, total_time, embeddings = 0, 0.0, {}
    for photo in photos:
        img = resize(photo, settings.event_photo_max_dimension)
        start = time.perf_counter()
        faces = app.get(img)
        if not faces:
            elapsed = time.perf_counter() - start
            p(f"  {photo.name}: NO FACE DETECTED, {elapsed*1000:.0f}ms")
            total_time += elapsed
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        aligned = face_align.norm_crop(img, best.kps, image_size=112, mode="arcface")
        result = DeepFace.represent(img_path=aligned, model_name=MODEL,
                                     detector_backend="skip", enforce_detection=False)
        elapsed = time.perf_counter() - start
        detected += 1
        embeddings[photo.name] = result[0]["embedding"]
        p(f"  {photo.name}: face detected ({len(faces)} total), {elapsed*1000:.0f}ms")
        total_time += elapsed
    return {"name": "insightface", "detected": detected, "total": len(photos),
            "total_time_s": total_time, "embeddings": embeddings}


def main() -> None:
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    job_dir = find_job_dir(job_id)
    photos = sorted((job_dir / "event_photos").glob("*"))
    p(f"Using job: {job_dir.name} ({len(photos)} event photos)")

    results = [run_opencv(photos), run_retinaface(photos), run_insightface(photos)]

    p("\n=== Embedding agreement: retinaface vs insightface (same photos) ===")
    p("Near 0 = same face embedded consistently regardless of which detector found it.")
    retinaface_embs = results[1]["embeddings"]
    insightface_embs = results[2]["embeddings"]
    agreements = {}
    for name in retinaface_embs:
        if name in insightface_embs:
            dist = cosine_distance(retinaface_embs[name], insightface_embs[name])
            agreements[name] = dist
            p(f"  {name}: distance={dist:.4f}")

    p("\n=== SUMMARY ===")
    p(f"{'Detector':<14}{'Detected':<12}{'Avg ms/photo':<16}")
    for r in results:
        n = r["total"]
        avg_ms = (r["total_time_s"] / n * 1000) if n else 0
        p(f"{r['name']:<14}{r['detected']}/{n:<10}{avg_ms:<16.0f}")

    RESULTS_PATH.write_text(json.dumps({
        "job_id": job_dir.name,
        "photo_count": len(photos),
        "detectors": [{"name": r["name"], "detected": r["detected"], "total": r["total"],
                        "avg_ms_per_photo": (r["total_time_s"] / r["total"] * 1000) if r["total"] else 0}
                       for r in results],
        "embedding_agreement_retinaface_vs_insightface": agreements,
    }, indent=2))
    p(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
