"""Can the production embedder tell two DIFFERENT people apart?

IMG_20250301_101253092 contains two people (target + bystander), same photo,
same camera, same instant. That removes every confound: lighting, pose,
outfit, time, compression are all identical.

  - If the two embed FAR apart  -> embedder discriminates; low burst-shot
    distances elsewhere are correct same-person behaviour (Prateek's reading).
  - If they embed CLOSE together -> the embedder cannot separate identities
    on this input, and low distances mean nothing (my degeneracy reading).

Run at production's 800px and at full resolution to test whether face size
is what drives it.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from services import face_matcher as fm  # noqa: E402

PHOTO = (
    BACKEND / "storage/uploads/6fe1d707-565a-4f7f-a5b8-4faf75fca594"
    / "event_photos/IMG_20250301_101253092_HDR.jpg"
)


def p(m=""):
    print(m, flush=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def embed_crop(aligned):
    res = fm._get_deepface().represent(
        img_path=aligned, model_name="ArcFace",
        detector_backend="skip", enforce_detection=False,
    )
    return res[0]["embedding"]


from insightface.utils import face_align  # noqa: E402

app = fm._get_insightface_app()
raw = cv2.imread(str(PHOTO))
p(f"photo: {raw.shape[1]}x{raw.shape[0]}\n")

for label, max_dim, det in [("production (800px)", 800, 800), ("full res", 3200, 3200)]:
    h, w = raw.shape[:2]
    s = max_dim / max(h, w)
    img = cv2.resize(raw, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else raw
    app.prepare(ctx_id=-1, det_size=(det, det))
    try:
        faces = app.get(img)
    except Exception as exc:
        p(f"{label}: detector failed ({type(exc).__name__})")
        continue

    if len(faces) < 2:
        p(f"{label}: only {len(faces)} face(s) found, need 2")
        continue

    faces = sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))[:2]
    areas = [int((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) for f in faces]
    embs = [embed_crop(face_align.norm_crop(img, f.kps, image_size=112, mode="arcface")) for f in faces]
    d = cosine(embs[0], embs[1])

    px = [int(math.sqrt(a)) for a in areas]
    p(f"{label}  image {img.shape[1]}x{img.shape[0]}")
    p(f"  face A: {areas[0]:,}px2 (~{px[0]}x{px[0]})   face B: {areas[1]:,}px2 (~{px[1]}x{px[1]})")
    p(f"  size difference: {abs(areas[0]-areas[1]) / max(areas) * 100:.1f}%  <- decides 'largest'")
    p(f"  DISTANCE BETWEEN TWO DIFFERENT PEOPLE: {d:.4f}")
    if d < 0.15:
        p(f"  => CANNOT separate two different people. Degeneracy confirmed.")
    elif d < 0.4:
        p(f"  => weak separation")
    else:
        p(f"  => separates them fine")
    p("")
