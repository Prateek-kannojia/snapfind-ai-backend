"""Audit of the EVENT-PHOTO path — independent check of the reported bugs.

Claims under test:
  3. Largest-face-only picks the wrong person, and flips with resolution
  6. The 800px downscale destroys small faces
  7. Threshold 0.68 was never calibrated
  8. Degenerate embeddings — do the event embeddings collapse together?

Uses the real production functions. The key test for claim 8 is
event-photo-vs-event-photo distance: if four different photos embed to nearly
the same vector, they have collapsed and any "match" is meaningless.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

from core.settings import settings  # noqa: E402
from services import face_matcher as fm  # noqa: E402

UPLOADS = BACKEND / "storage" / "uploads"
RESOLUTIONS = [800, 1600, 3200]  # SCRFD needs det_size divisible by 32


def p(msg=""):
    print(msg, flush=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def resized(path, max_dim):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = max_dim / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


p("=" * 78)
p("CLAIMS 3 + 6 — face sizes and which face wins, by resolution")
p("=" * 78)

app = fm._get_insightface_app()
job6 = next(d for d in UPLOADS.iterdir() if d.name.startswith("6fe1d707"))

for photo in sorted((job6 / "event_photos").glob("*.jpg")):
    p(f"\n{photo.name}")
    for max_dim in RESOLUTIONS:
        img = resized(photo, max_dim)
        # insightface det_size is fixed at build time; re-prepare per resolution
        app.prepare(ctx_id=-1, det_size=(max_dim, max_dim))
        try:
            faces = app.get(img)
        except Exception as exc:
            p(f"  at {max_dim:>4}px: detector failed ({type(exc).__name__})")
            continue
        if not faces:
            p(f"  at {max_dim:>4}px ({img.shape[1]}x{img.shape[0]}): no faces")
            continue
        areas = [((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), f) for f in faces]
        areas.sort(key=lambda t: -t[0])
        winner_idx = 0
        desc = ", ".join(f"{int(a):,}px2" for a, _ in areas)
        px = int(math.sqrt(areas[0][0]))
        p(f"  at {max_dim:>4}px ({img.shape[1]}x{img.shape[0]}): {len(faces)} face(s) [{desc}]"
          f"  winner~{px}x{px}")

# restore production det_size
app.prepare(ctx_id=-1, det_size=(settings.event_photo_max_dimension,) * 2)

p("\n" + "=" * 78)
p("CLAIM 8 — do the event-photo embeddings collapse together?")
p("=" * 78)
p("Four DIFFERENT photos, production pipeline. If these are all ~0 apart,")
p("they've collapsed to one vector and every 'match' is meaningless.\n")

for job_dir in sorted(UPLOADS.iterdir()):
    job = job_dir.name[:8]
    selfie = next((job_dir / "selfie").glob("*.jpg"), None)
    photos = sorted((job_dir / "event_photos").glob("*.jpg"))
    p(f"--- job {job}... ---")

    try:
        sel_emb = fm._selfie_embedding(resized(selfie, settings.selfie_max_dimension))
    except Exception as exc:
        p(f"  selfie failed: {exc}")
        continue

    embs = {}
    for photo in photos:
        try:
            orig = cv2.imread(str(photo))
            det = resized(photo, settings.event_photo_max_dimension)
            embs[photo.name] = fm._event_photo_embeddings(det, orig)[0]
        except Exception as exc:
            p(f"  {photo.name}: {type(exc).__name__}")

    p("  selfie vs each photo (production distance, threshold 0.68):")
    for name, e in embs.items():
        d = cosine(sel_emb, e)
        verdict = "MATCH" if d <= 0.68 else "no"
        near = "  <-- near-zero, suspicious" if d < 0.05 else ""
        p(f"    {name[:34]:<34} {d:.4f}  {verdict}{near}")

    names = list(embs)
    p("  photo vs photo (these are DIFFERENT photos — should NOT be ~0):")
    collapsed = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = cosine(embs[names[i]], embs[names[j]])
            if d >= 0.05:
                collapsed = False
            p(f"    {names[i][:22]:<22} vs {names[j][:22]:<22} {d:.4f}")
    p(f"  => embeddings collapsed to one vector: {collapsed}")
    p("")
