"""Audit of the SELFIE path — independent check of the reported bugs.

Claims under test:
  1. No quality gate: mtcnn returns non-faces and they're accepted
  2. No face selection: result[0] is taken arbitrarily from multi-face selfies
  8. Degenerate embeddings pass silently

Calls the real production functions from services/face_matcher.py — no
reimplementation — plus raw mtcnn to see what production is throwing away.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from core.settings import settings  # noqa: E402
from services import face_matcher as fm  # noqa: E402

UPLOADS = BACKEND / "storage" / "uploads"


def p(msg=""):
    print(msg, flush=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 1 - dot / (na * nb)


p("=" * 78)
p("SELFIE PATH AUDIT")
p("=" * 78)

selfie_embeddings = {}

for job_dir in sorted(UPLOADS.iterdir()):
    selfie = next((job_dir / "selfie").glob("*.jpg"), None)
    if selfie is None:
        continue
    job = job_dir.name[:8]
    p(f"\n--- job {job}… ---")

    # Exactly what production does: downscale to selfie_max_dimension first.
    img = fm._load_resized_image_from_path(selfie) if hasattr(fm, "_load_resized_image_from_path") else None
    import cv2  # noqa: E402

    raw = cv2.imread(str(selfie))
    h, w = raw.shape[:2]
    scale = settings.selfie_max_dimension / max(h, w)
    resized = cv2.resize(raw, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else raw
    p(f"  original {w}x{h}  ->  production sees {resized.shape[1]}x{resized.shape[0]}")

    # What mtcnn ACTUALLY finds (production only ever looks at [0])
    faces = fm._get_deepface().extract_faces(
        img_path=resized,
        detector_backend=settings.selfie_detector,
        enforce_detection=False,
        align=True,
    )
    p(f"  mtcnn returned {len(faces)} face(s):")
    for i, f in enumerate(faces):
        a = f["facial_area"]
        conf = f.get("confidence", 0)
        marker = "  <-- production uses THIS one ([0])" if i == 0 else ""
        p(f"    [{i}] {a['w']}x{a['h']} px ({a['w']*a['h']:,} px²)  confidence={conf:.4f}{marker}")

    if faces:
        a0 = faces[0]["facial_area"]
        if a0["w"] < 80 or a0["h"] < 80:
            p(f"    ^^ FLAG: {a0['w']}x{a0['h']} is far too small to identify a person")
        if len(faces) > 1:
            biggest = max(range(len(faces)), key=lambda i: faces[i]["facial_area"]["w"] * faces[i]["facial_area"]["h"])
            if biggest != 0:
                p(f"    ^^ FLAG: [0] is not the largest face — [{biggest}] is. Production picks arbitrarily.")

    # The real production embedding
    try:
        emb = fm._selfie_embedding(resized)
        selfie_embeddings[job] = emb
        norm = math.sqrt(sum(x * x for x in emb))
        spread = max(emb) - min(emb)
        p(f"  production embedding: dim={len(emb)} L2norm={norm:.3f} range={spread:.3f}")
    except Exception as exc:
        p(f"  production embedding FAILED: {type(exc).__name__}: {exc}")

p("\n" + "=" * 78)
p("CLAIM 8 — are these embeddings degenerate?")
p("=" * 78)
p("Three DIFFERENT selfies. If the embedder works, they should be far apart.")
jobs = list(selfie_embeddings)
for i in range(len(jobs)):
    for j in range(i + 1, len(jobs)):
        d = cosine(selfie_embeddings[jobs[i]], selfie_embeddings[jobs[j]])
        flag = "  <-- SUSPICIOUS: near-identical" if d < 0.05 else ""
        p(f"  {jobs[i]}… vs {jobs[j]}… : {d:.4f}{flag}")
