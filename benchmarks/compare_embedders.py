"""Head-to-head: DeepFace ArcFace vs w600k_mbf, on the fixed pipeline.

Everything is held constant except the embedder: same photos, same detector,
same crops (from the original image), same ground truth. The two produce
different vector spaces, so each gets its own threshold sweep — comparing
them at a shared threshold would be meaningless.

Needs a corpus with ground truth; build one with build_corpus.py.

    venv\\Scripts\\python.exe benchmarks\\compare_embedders.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from _common import (  # noqa: E402
    CORPUS, aligned_faces, cosine, md_table, p, score, write_results_section,
)

from core.settings import settings  # noqa: E402
from services import face_matcher as fm  # noqa: E402

MBF_PATH = settings.insightface_home / "models" / "buffalo_sc" / "w600k_mbf.onnx"
MBF_INPUT_MEAN, MBF_INPUT_STD, MBF_INPUT_SIZE = 127.5, 127.5, (112, 112)
THRESHOLDS = [round(x * 0.05, 2) for x in range(6, 21)]  # 0.30 .. 1.00


mbf_session = ort.InferenceSession(str(MBF_PATH), providers=["CPUExecutionProvider"])


def embed_mbf(aligned: np.ndarray) -> list[float]:
    """(x - 127.5) / 127.5, BGR->RGB, NCHW — read from insightface's
    arcface_onnx.py, not recalled. That file branches on whether
    normalization is baked into the graph; guessing picks wrong 50% of the
    time and fails silently."""
    blob = cv2.dnn.blobFromImage(
        aligned, 1.0 / MBF_INPUT_STD, MBF_INPUT_SIZE, (MBF_INPUT_MEAN,) * 3, swapRB=True
    )
    out = mbf_session.run(None, {mbf_session.get_inputs()[0].name: blob})[0]
    return out[0].tolist()


def validate_mbf_port(sample: np.ndarray) -> float:
    """Prove our ONNX path runs the model correctly before trusting any result.

    ML preprocessing fails silently: wrong pixel scaling or swapped channels
    still returns a confident, correctly-shaped 512-D vector. Comparing
    against insightface's own reference on identical input is a hard
    pass/fail needing no labels. Must be ~0.
    """
    from insightface.model_zoo import get_model

    ref = get_model(str(MBF_PATH))
    ref.prepare(ctx_id=-1)
    return cosine(embed_mbf(sample), ref.get_feat(sample).flatten().tolist())


def main() -> None:
    if not (CORPUS / "ground_truth.json").exists():
        raise SystemExit(f"No corpus at {CORPUS} — run build_corpus.py first")
    truth = [j for j in json.loads((CORPUS / "ground_truth.json").read_text())
             if j.get("source") == "synthetic"]  # needs the face-size axis

    # Crop once, embed with both — guarantees identical input to each model.
    p("cropping faces (shared by both embedders)...")
    per_photo, selfie_crops = [], {}
    t0 = time.time()
    for job in truth:
        job_dir = Path(job["root"])
        sc = aligned_faces(job_dir / job["selfie"])
        if not sc:
            p(f"  {job['job_id']}: no face in selfie, skipped")
            continue
        selfie_crops[job["job_id"]] = max(sc, key=lambda c: c.size)
        for photo in job["event_photos"]:
            per_photo.append({
                "job": job["job_id"],
                "truth": photo["contains_target"],
                "face_px": photo["face_px"],
                "crops": aligned_faces(job_dir / "event_photos" / photo["file"]),
            })
        p(f"  {job['job_id']}: {len(job['event_photos'])} photos")
    p(f"cropping took {time.time() - t0:.0f}s; {len(per_photo)} photos\n")

    port_error = validate_mbf_port(next(iter(selfie_crops.values())))
    p(f"w600k_mbf port validation vs insightface reference: {port_error:.8f}")
    if port_error > 1e-5:
        raise SystemExit("port validation FAILED — preprocessing wrong, results meaningless")
    p("port OK — comparison below is trustworthy\n")

    results = {"_mbf_port_validation": port_error}
    for name, embed_fn in [("DeepFace ArcFace", fm._embed_aligned), ("w600k_mbf", embed_mbf)]:
        p("=" * 70)
        p(name)
        p("=" * 70)
        started = time.time()
        sel_emb = {j: embed_fn(c) for j, c in selfie_crops.items()}
        records = []
        for item in per_photo:
            s = sel_emb.get(item["job"])
            d = None
            if s is not None and item["crops"]:
                d = min(cosine(s, embed_fn(c)) for c in item["crops"])
            records.append({"truth": item["truth"], "d": d, "face_px": item["face_px"]})
        elapsed = time.time() - started

        best = max(THRESHOLDS, key=lambda t, rs=records: score(rs, t)["f1"])
        bs = score(records, best)
        p(f"  best threshold {best:.2f}: precision={bs['precision']:.3f} "
          f"recall={bs['recall']:.3f} F1={bs['f1']:.3f} acc={bs['accuracy']:.3f}")
        p(f"  (TP={bs['tp']} FP={bs['fp']} FN={bs['fn']} TN={bs['tn']}, {elapsed:.0f}s)")

        p("  accuracy by face size at its best threshold:")
        by = defaultdict(list)
        for r in records:
            by[r["face_px"]].append(r)
        for px in sorted(by):
            ss = score(by[px], best)
            p(f"    {px:>4}px : {ss['accuracy']:.3f}  "
              f"(TP={ss['tp']} FP={ss['fp']} FN={ss['fn']} TN={ss['tn']})")

        # Does ANY threshold cleanly split same from different?
        same = [r["d"] for r in records if r["truth"] and r["d"] is not None]
        diff = [r["d"] for r in records if not r["truth"] and r["d"] is not None]
        gap = min(diff) - max(same)
        p(f"  worst SAME={max(same):.4f}  best DIFFERENT={min(diff):.4f}  gap={gap:+.4f}")
        p(f"  -> {'a threshold separates cleanly' if gap > 0 else 'ranges overlap; no perfect threshold'}\n")

        results[name] = {"best_threshold": best, **bs, "elapsed_s": elapsed,
                         "worst_same": max(same), "best_diff": min(diff), "gap": gap,
                         "by_face_px": {str(px): score(by[px], best) for px in sorted(by)}}

    models = [(n, r) for n, r in results.items() if n != "_mbf_port_validation"]
    body = [
        "Same photos, same detector, same crops — only the embedder differs. "
        "Each model gets its own threshold sweep, because the two produce "
        "different vector spaces and a shared threshold would mean nothing.\n",
        f"ONNX port validated against insightface's own reference: "
        f"`{port_error:.8f}`. If that is not ~0 the preprocessing is wrong and "
        f"every number below is meaningless.\n",
        md_table(["model", "best threshold", "precision", "recall", "F1",
                  "accuracy", "FP", "embed time"],
                 [[n, f"{r['best_threshold']:.2f}", f"{r['precision']:.3f}",
                   f"{r['recall']:.3f}", f"**{r['f1']:.3f}**",
                   f"{r['accuracy']:.3f}", r["fp"], f"{r['elapsed_s']:.0f}s"]
                  for n, r in models]),
        "\n**Separability** — a threshold can only work if the worst same-person "
        "pair scores closer than the best different-person pair:\n",
        md_table(["model", "worst SAME", "best DIFFERENT", "gap"],
                 [[n, f"{r['worst_same']:.4f}", f"{r['best_diff']:.4f}",
                   f"{r['gap']:+.4f}"] for n, r in models]),
        "\nA negative gap means the two ranges overlap, so no threshold is "
        "perfect for that model.\n",
        f"\n**Accuracy by face size** (each model at its own best threshold) — "
        f"this is what decides whether swapping the model fixes small faces:\n",
        md_table(["face px"] + [n for n, _ in models],
                 [[px] + [f"{r['by_face_px'][px]['accuracy']:.3f}"
                          if px in r["by_face_px"] else "-" for _, r in models]
                  for px in sorted(models[0][1]["by_face_px"], key=int)]),
    ]
    write_results_section("embedders", "Embedder comparison", "\n".join(body))


if __name__ == "__main__":
    main()
