"""Which variable actually loses the face — det_size, or image resolution?

The earlier audit varied BOTH together, so it can't tell them apart. This
holds one fixed at a time.

  A. image FIXED at full res, det_size varied
  B. det_size FIXED at 800 (production), image resolution varied
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

from services import face_matcher as fm  # noqa: E402

PHOTO = (
    BACKEND / "storage/uploads/6fe1d707-565a-4f7f-a5b8-4faf75fca594"
    / "event_photos/IMG_20250302_155231085_HDR.jpg"
)
DET_SIZES = [640, 800, 1600, 3200]
IMG_SIZES = [800, 1600, 3200]


def p(m=""):
    print(m, flush=True)


def scaled(raw, max_dim):
    h, w = raw.shape[:2]
    s = max_dim / max(h, w)
    return cv2.resize(raw, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else raw


def detect(app, img, det):
    app.prepare(ctx_id=-1, det_size=(det, det))
    try:
        faces = app.get(img)
    except Exception as exc:
        return None, f"FAILED ({type(exc).__name__})"
    if not faces:
        return [], "no faces"
    areas = sorted((int((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])) for f in faces), reverse=True)
    return areas, ", ".join(f"{a:,}" for a in areas)


app = fm._get_insightface_app()
raw = cv2.imread(str(PHOTO))
p(f"photo: {PHOTO.name}  {raw.shape[1]}x{raw.shape[0]}")

p("\n" + "=" * 70)
p("A. IMAGE FIXED at 3200px  —  only det_size varies")
p("=" * 70)
img_fixed = scaled(raw, 3200)
p(f"   (image {img_fixed.shape[1]}x{img_fixed.shape[0]} for every row)")
for det in DET_SIZES:
    areas, desc = detect(app, img_fixed, det)
    big = f"  largest ~{int(math.sqrt(areas[0]))}px" if areas else ""
    p(f"   det_size {det:>4}: {desc}{big}")

p("\n" + "=" * 70)
p("B. det_size FIXED at 800 (production)  —  only image resolution varies")
p("=" * 70)
for size in IMG_SIZES:
    img = scaled(raw, size)
    areas, desc = detect(app, img, 800)
    big = f"  largest ~{int(math.sqrt(areas[0]))}px" if areas else ""
    p(f"   image {img.shape[1]:>4}x{img.shape[0]:<4} det_size 800: {desc}{big}")

p("\n" + "=" * 70)
p("VERDICT")
p("=" * 70)
p("If A varies wildly and B is stable  -> det_size is the culprit (coupling matters)")
p("If B varies and A is stable         -> image resolution is the culprit")
p("If both vary                        -> they interact; pin det_size AND test")

app.prepare(ctx_id=-1, det_size=(800, 800))
