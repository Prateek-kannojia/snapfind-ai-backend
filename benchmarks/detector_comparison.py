"""Every face detector this project tried, in order, on the same photos.

services/face_matcher.py only holds the current path (insightface/SCRFD).
The two detectors it replaced — opencv's Haar cascade, then retinaface — are
kept here fully runnable, so each swap can be reproduced rather than just
claimed in a README.

Measures three things:
  1. detection rate  — did it find a face at all?
  2. speed           — ms per photo on CPU
  3. agreement       — for the two that work, does the same face produce the
                       same embedding? Answers "is the fast one as accurate,
                       or only fast".

All three see the identical input image, so the only variable is the detector.

Photos come from the corpus (build_corpus.py first), or from a folder passed
as an argument. retinaface is slow on CPU, so only a sample is used.

    venv\\Scripts\\python.exe benchmarks\\detector_comparison.py [photo_folder]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CORPUS, cosine, md_table, p, write_results_section  # noqa: E402

import os  # noqa: E402

from core.settings import settings  # noqa: E402

# Must be set before importing deepface/insightface, same as
# services/face_matcher.py does — otherwise both libraries ignore this
# project's already-downloaded weights and re-download everything.
settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)
settings.insightface_home.mkdir(parents=True, exist_ok=True)
os.environ["INSIGHTFACE_HOME"] = str(settings.insightface_home)

import cv2  # noqa: E402

MODEL = "ArcFace"
SAMPLE_SIZE = 30


def corpus_photos() -> tuple[list[Path], set[str]]:
    """A spread across jobs, not the first N — one job's photos share a
    background and would flatter or punish every detector equally.

    Also returns which of them are real photos: the synthetic ones are
    composites holding two faces, which matters when reading agreement.
    """
    gt_path = CORPUS / "ground_truth.json"
    if not gt_path.exists():
        raise SystemExit(f"No corpus at {CORPUS} — run build_corpus.py first")
    photos, real = [], set()
    for job in json.loads(gt_path.read_text()):
        root = Path(job["root"]) / "event_photos"
        for ph in job["event_photos"]:
            photos.append(root / ph["file"])
            if job.get("source") == "real":
                real.add(ph["file"])
    random.Random(0).shuffle(photos)
    return photos[:SAMPLE_SIZE], real


def folder_photos(folder: Path) -> list[Path]:
    if not folder.exists():
        raise SystemExit(f"No photo folder at {folder}")
    photos = sorted(
        path for path in folder.iterdir()
        if path.suffix.lower() in settings.allowed_image_extensions
    )
    if not photos:
        raise SystemExit(f"No supported images found in {folder}")
    return photos


def resize(image_path: Path, max_dim: int):
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        image = cv2.resize(image, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image


def _run_deepface_detector(photos: list[Path], backend: str) -> dict:
    from deepface import DeepFace

    DeepFace.represent(img_path=resize(photos[0], settings.event_photo_max_dimension),
                       model_name=MODEL, detector_backend=backend, enforce_detection=False)

    detected, total_time, embeddings = 0, 0.0, {}
    for photo in photos:
        img = resize(photo, settings.event_photo_max_dimension)
        start = time.perf_counter()
        try:
            result = DeepFace.represent(img_path=img, model_name=MODEL,
                                        detector_backend=backend, enforce_detection=True)
            elapsed = time.perf_counter() - start
            detected += 1
            embeddings[photo.name] = result[0]["embedding"]
            p(f"  {photo.name}: face detected, {elapsed*1000:.0f}ms")
        except Exception:
            elapsed = time.perf_counter() - start
            p(f"  {photo.name}: NO FACE DETECTED, {elapsed*1000:.0f}ms")
        total_time += elapsed
    return {"name": backend, "detected": detected, "total": len(photos),
            "total_time_s": total_time, "embeddings": embeddings}


# --- Detector 1: opencv (Haar cascade) — the original default, replaced ---
def run_opencv(photos: list[Path]) -> dict:
    p("\n=== Detector 1/3: opencv (Haar cascade) — the original default ===")
    p("Weak, no deep learning. Kept only for historical comparison.")
    return _run_deepface_detector(photos, "opencv")


# --- Detector 2: retinaface (ResNet50 backbone, via DeepFace) — 2nd choice ---
def run_retinaface(photos: list[Path]) -> dict:
    p("\n=== Detector 2/3: retinaface (ResNet50 backbone, via DeepFace) ===")
    p("Fixed opencv's accuracy problem, but slow on CPU.")
    return _run_deepface_detector(photos, "retinaface")


# --- Detector 3: insightface/SCRFD-500MF — current production default ---
def run_insightface(photos: list[Path]) -> dict:
    from deepface import DeepFace
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    p("\n=== Detector 3/3: insightface / SCRFD-500MF (current production) ===")
    p("Same weight class as RetinaFace-MobileNet-0.25. Lightweight ONNX.")
    app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"],
                       root=str(settings.insightface_home))
    app.prepare(ctx_id=-1, det_size=(settings.face_detector_size,) * 2)

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
    if len(sys.argv) > 1:
        photos, real = folder_photos(Path(sys.argv[1]).resolve()), set()
        source = sys.argv[1]
    else:
        photos, real = corpus_photos()
        source = f"corpus sample ({SAMPLE_SIZE} photos)"
    p(f"Using {len(photos)} photos from {source}")

    results = [run_opencv(photos), run_retinaface(photos), run_insightface(photos)]

    p("\n=== Embedding agreement: retinaface vs insightface (same photos) ===")
    p("Near 0 = the same face embeds the same regardless of which detector found it.")
    retina, insight = results[1]["embeddings"], results[2]["embeddings"]
    agreements = {n: cosine(retina[n], insight[n]) for n in retina if n in insight}
    for name, dist in agreements.items():
        p(f"  {name}: distance={dist:.4f}")

    rows = []
    for r in results:
        n = r["total"]
        rows.append([r["name"], f"{r['detected']}/{n}",
                     f"{r['detected'] / n:.0%}" if n else "-",
                     f"{r['total_time_s'] / n * 1000:.0f}ms" if n else "-"])
    p("\n=== SUMMARY ===")
    for row in rows:
        p("  " + "  ".join(str(c) for c in row))

    body = [
        f"{len(photos)} photos, identical input to all three — the detector is "
        f"the only variable. CPU only.\n",
        md_table(["detector", "found a face", "rate", "avg per photo"], rows),
        "\n**Embedding agreement** — retinaface vs insightface on the photos "
        "both found. Near 0 means the cheap detector hands the embedder the "
        "same face as the expensive one, so the speed win costs no accuracy:\n",
    ]
    groups = [("real photos (one subject)", [d for n, d in agreements.items() if n in real]),
              ("synthetic composites (two faces)",
               [d for n, d in agreements.items() if n not in real])]
    agree_rows = [[label, len(v), f"{sorted(v)[0]:.4f}",
                   f"{sorted(v)[len(v) // 2]:.4f}", f"{sorted(v)[-1]:.4f}"]
                  for label, v in groups if v]
    if agree_rows:
        body.append(md_table(["photos", "count", "min", "median", "max"], agree_rows))
        body.append(
            "\nThe two rows measure different things. On a single-subject photo "
            "there is only one face to find, so this is genuine embedding "
            "agreement. The composites hold two faces and the detectors rank "
            "them differently — DeepFace returns its own first face, this "
            "pipeline takes the largest — so a high distance there often means "
            "they embedded *different people*, not that either was wrong.")
    else:
        body.append("_No photo was found by both detectors, so no comparison._")
    write_results_section("detectors", "Detector comparison", "\n".join(body))


if __name__ == "__main__":
    main()
