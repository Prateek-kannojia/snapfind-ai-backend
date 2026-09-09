"""Measure the pipeline against labelled ground truth, before vs after the fix.

Runs the real production functions in two configurations:
  BEFORE - crop from the 800px downscaled image, score largest face only
  AFTER  - crop from the original, score every face

Reports precision/recall/accuracy, a breakdown by face size, and a threshold
sweep (0.68 has never been calibrated against labelled data).
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

from core.settings import settings  # noqa: E402
from services import face_matcher as fm  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"
RESULTS = Path(__file__).resolve().parent / "corpus_results.json"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85]


def p(m=""):
    print(m, flush=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def min_distance(selfie_emb, photo_path: Path, crop_from_original: bool,
                 largest_only: bool) -> float | None:
    original = cv2.imread(str(photo_path))
    if original is None:
        return None
    detect_image = fm._downscale(original, settings.event_photo_max_dimension)
    crop_src = original if crop_from_original else detect_image
    try:
        if largest_only:
            from insightface.utils import face_align
            app = fm._get_insightface_app()
            faces = app.get(detect_image)
            if not faces:
                return None
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            scale = crop_src.shape[1] / detect_image.shape[1]
            kps = best.kps * scale if scale != 1 else best.kps
            embs = [fm._embed_aligned(face_align.norm_crop(crop_src, kps, image_size=112, mode="arcface"))]
        else:
            embs = fm._event_photo_embeddings(detect_image, crop_src)
    except Exception:
        return None
    return min(cosine(selfie_emb, e) for e in embs)


def score(records, threshold):
    tp = sum(1 for r in records if r["truth"] and r["dist"] is not None and r["dist"] <= threshold)
    fp = sum(1 for r in records if not r["truth"] and r["dist"] is not None and r["dist"] <= threshold)
    fn = sum(1 for r in records if r["truth"] and (r["dist"] is None or r["dist"] > threshold))
    tn = sum(1 for r in records if not r["truth"] and (r["dist"] is None or r["dist"] > threshold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f1": f1, "accuracy": (tp + tn) / max(1, len(records))}


truth = json.loads((CORPUS / "ground_truth.json").read_text())
configs = [
    ("BEFORE (crop@800, largest face only)", False, True),
    ("AFTER  (crop from original, all faces)", True, False),
]

all_results = {}
for label, crop_orig, largest in configs:
    p("\n" + "=" * 74)
    p(label)
    p("=" * 74)
    records = []
    started = time.time()
    for job in truth:
        job_dir = CORPUS / job["job_id"]
        selfie_img = cv2.imread(str(job_dir / job["selfie"]))
        try:
            sel = fm._selfie_embedding(fm._downscale(selfie_img, settings.selfie_max_dimension))
        except Exception as exc:
            p(f"  {job['job_id']}: selfie failed ({type(exc).__name__}) - job skipped")
            continue
        for photo in job["event_photos"]:
            d = min_distance(sel, job_dir / "event_photos" / photo["file"],
                             crop_orig, largest)
            records.append({"job": job["job_id"], "truth": photo["contains_target"],
                            "dist": d, "face_px": photo["face_px"]})
        p(f"  {job['job_id']}: {len(job['event_photos'])} photos done")

    elapsed = time.time() - started
    s = score(records, 0.68)
    p(f"\n  at threshold 0.68  ({elapsed:.0f}s)")
    p(f"    TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}")
    p(f"    precision={s['precision']:.3f}  recall={s['recall']:.3f}  "
      f"F1={s['f1']:.3f}  accuracy={s['accuracy']:.3f}")

    p("\n  by face size (accuracy at 0.68):")
    by_size = defaultdict(list)
    for r in records:
        by_size[r["face_px"]].append(r)
    for px in sorted(by_size):
        ss = score(by_size[px], 0.68)
        p(f"    {px:>4}px : acc={ss['accuracy']:.3f}  "
          f"(TP={ss['tp']} FP={ss['fp']} FN={ss['fn']} TN={ss['tn']})")

    p("\n  threshold sweep:")
    p(f"    {'thresh':>7} {'prec':>7} {'recall':>7} {'F1':>7} {'acc':>7}")
    best = None
    for t in THRESHOLDS:
        ss = score(records, t)
        star = ""
        if best is None or ss["f1"] > best[1]["f1"]:
            best = (t, ss)
        p(f"    {t:>7.2f} {ss['precision']:>7.3f} {ss['recall']:>7.3f} "
          f"{ss['f1']:>7.3f} {ss['accuracy']:>7.3f}{star}")
    p(f"    best F1 at threshold {best[0]:.2f} (F1={best[1]['f1']:.3f})")

    all_results[label] = {
        "at_0.68": s,
        "best_threshold": best[0],
        "best": best[1],
        "by_face_px": {str(px): score(by_size[px], 0.68) for px in sorted(by_size)},
        "elapsed_s": elapsed,
        "n_records": len(records),
    }

RESULTS.write_text(json.dumps(all_results, indent=2))
p(f"\nwrote {RESULTS}")
