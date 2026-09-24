"""Head-to-head: DeepFace ArcFace vs w600k_mbf, on the fixed pipeline.

Everything is held constant except the embedder: same photos, same detector,
same crops (from the original image), same ground truth. The two produce
different vector spaces, so each gets its own threshold sweep — comparing
them at a shared threshold would be meaningless.

Needs a corpus with ground truth; build one with build_corpus.py.

    venv\\Scripts\\python.exe benchmarks\\compare_embedders.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from _common import (  # noqa: E402
    aligned_faces, cosine, discover_jobs, label_from, md_table, p, score,
    write_results_section,
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


# Is the incumbent losing because of HOW it is called? Two suspects, both read
# out of the installed deepface source rather than recalled:
#
#   SCALE   Confirmed wrong. preprocessing.py:34 — normalization="base" returns
#           the image untouched in [0,1]. ArcFace's own paper specifies
#           (x-127.5)/128 on [0,255], which deepface ships as
#           normalization="ArcFace" (:66-71). Production never passes the
#           argument, so the model gets a pixel range it was not trained on.
#   COLOUR  Suspicious, NOT confirmed. representation.py:144 flips the input
#           BGR->RGB and :173 flips it back, so the model receives whatever
#           order it was handed — and :43 documents the expected input as BGR.
#           So deepface deliberately feeds its ArcFace weights BGR. Either that
#           is a library-wide bug or those weights want BGR; this repo cannot
#           tell which, and the sweep does not settle it (RGB is better on the
#           synthetic corpus, BGR is better on the real photos, both by little).
#
# Run with --diagnose-incumbent. Neither closes the gap — the whole
# configuration question is worth ~0.007 F1 against a 0.027 model gap, so
# DeepFace loses on merit in every configuration. See README.
INCUMBENT_CONFIGS = [
    ("BGR + base  (PRODUCTION)", False, "base"),
    ("RGB + base", True, "base"),
    ("BGR + ArcFace", False, "ArcFace"),
    ("RGB + ArcFace", True, "ArcFace"),
    ("RGB + raw", True, "raw"),
]


def diagnose_incumbent(per_photo: list[dict], selfie_crops: dict) -> None:
    """Sweep how DeepFace is called, on the same crops as the main comparison."""
    p("=" * 70)
    p("Incumbent configuration sweep — is DeepFace losing on merit?")
    p("=" * 70)
    p(f"{'config':<28}{'thr':>5}{'prec':>7}{'recall':>8}{'F1':>7}{'FP':>5}"
      f"{'real':>7}{'all at':>8}{'FP there':>9}")
    p("-" * 84)
    for label, flip, norm in INCUMBENT_CONFIGS:
        def emb(c, flip=flip, norm=norm):
            return fm._get_deepface().represent(
                img_path=c[:, :, ::-1] if flip else c, model_name=fm.DEFAULT_MODEL,
                detector_backend="skip", enforce_detection=False, normalization=norm,
            )[0]["embedding"]

        sel = {j: emb(c) for j, c in selfie_crops.items()}
        recs = []
        for item in per_photo:
            s = sel.get(item["job"])
            d = min((cosine(s, emb(c)) for c in item["crops"]), default=None) \
                if s is not None and item["crops"] else None
            recs.append({**item, "d": d})

        syn = [r for r in recs if r["source"] == "synthetic"]
        real = [r for r in recs if r["source"] == "real" and not r["ambiguous"]]
        best = max(THRESHOLDS, key=lambda t, rs=syn: score(rs, t)["f1"])
        bs, rs_ = score(syn, best), score(real, best)
        full = next((t for t in THRESHOLDS if score(real, t)["recall"] >= 1.0), None)
        cost = score(syn, full)["fp"] if full is not None else None
        real_col = f"{rs_['tp']}/{len(real)}"
        full_col = f"{full:.2f}" if full is not None else "never"
        cost_col = cost if cost is not None else "-"
        p(f"{label:<28}{best:>5.2f}{bs['precision']:>7.3f}{bs['recall']:>8.3f}"
          f"{bs['f1']:>7.3f}{bs['fp']:>5}{real_col:>7}{full_col:>8}{cost_col:>9}")
    p("")


def main() -> None:
    # Both sources. The synthetic corpus carries per-photo labels and face
    # sizes, so it drives thresholds, precision and the face-size breakdown.
    # The real photos carry neither — but they are the actual product input,
    # and scoring the two models on them is the whole point of the exercise.
    # They are reported separately rather than pooled: they are all positives
    # (see REAL_JOBS in _common.py), so precision is undefined on them and
    # mixing them into one number would quietly inflate recall.
    jobs = discover_jobs()
    if not any(j["source"] == "synthetic" for j in jobs):
        raise SystemExit("No synthetic jobs — run build_corpus.py first")

    # Crop once, embed with both — guarantees identical input to each model.
    p("cropping faces (shared by both embedders)...")
    per_photo, selfie_crops = [], {}
    t0 = time.time()
    for job in jobs:
        sc = aligned_faces(job["selfie"])
        if not sc:
            p(f"  {job['name']}: no face in selfie, skipped")
            continue
        selfie_crops[job["name"]] = max(sc, key=lambda c: c.size)
        for photo in job["photos"]:
            contains, face_px = label_from(photo.name)
            if contains is None:  # real photos carry no filename marker
                contains = job["all_positive"]
            per_photo.append({
                "job": job["name"],
                "source": job["source"],
                "ambiguous": job["ambiguous"],
                "truth": contains,
                "face_px": face_px,
                "crops": aligned_faces(photo),
            })
        p(f"  {job['name']}: {len(job['photos'])} photos ({job['source']})")
    n_syn = sum(1 for i in per_photo if i["source"] == "synthetic")
    n_real = len(per_photo) - n_syn
    p(f"cropping took {time.time() - t0:.0f}s; {n_syn} synthetic + {n_real} real\n")

    if "--diagnose-incumbent" in sys.argv:
        diagnose_incumbent(per_photo, selfie_crops)

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
            records.append({"truth": item["truth"], "d": d, "face_px": item["face_px"],
                            "source": item["source"], "ambiguous": item["ambiguous"]})
        elapsed = time.time() - started

        syn = [r for r in records if r["source"] == "synthetic"]
        real_ok = [r for r in records if r["source"] == "real" and not r["ambiguous"]]
        real_amb = [r for r in records if r["source"] == "real" and r["ambiguous"]]

        # Threshold is chosen on the synthetic corpus only — it is the set with
        # both positives and negatives, so F1 is meaningful there. Tuning it on
        # the real photos would be fitting the threshold to the test set.
        best = max(THRESHOLDS, key=lambda t, rs=syn: score(rs, t)["f1"])
        bs = score(syn, best)
        p(f"  [synthetic] best threshold {best:.2f}: precision={bs['precision']:.3f} "
          f"recall={bs['recall']:.3f} F1={bs['f1']:.3f} acc={bs['accuracy']:.3f}")
        p(f"  (TP={bs['tp']} FP={bs['fp']} FN={bs['fn']} TN={bs['tn']}, {elapsed:.0f}s both sets)")

        p("  accuracy by face size at its best threshold:")
        by = defaultdict(list)
        for r in syn:
            by[r["face_px"]].append(r)
        for px in sorted(by):
            ss = score(by[px], best)
            p(f"    {px:>4}px : {ss['accuracy']:.3f}  "
              f"(TP={ss['tp']} FP={ss['fp']} FN={ss['fn']} TN={ss['tn']})")

        # Does ANY threshold cleanly split same from different? Synthetic only:
        # the real set has no negatives, so it cannot contribute a "different".
        same = [r["d"] for r in syn if r["truth"] and r["d"] is not None]
        diff = [r["d"] for r in syn if not r["truth"] and r["d"] is not None]
        gap = min(diff) - max(same)
        p(f"  worst SAME={max(same):.4f}  best DIFFERENT={min(diff):.4f}  gap={gap:+.4f}")
        p(f"  -> {'a threshold separates cleanly' if gap > 0 else 'ranges overlap; no perfect threshold'}")

        # Real photos: recall only. Every one contains the target, so there is
        # nothing to get wrong — precision would be 1.000 by construction and
        # would mean nothing. Reported across the sweep because the headline
        # question is how far the threshold must move to recover them all.
        real_recall = {}
        if real_ok:
            p(f"  [real] recall at each threshold ({len(real_ok)} photos, all positives):")
            for t in THRESHOLDS:
                rs = score(real_ok, t)
                real_recall[f"{t:.2f}"] = {"matched": rs["tp"], "total": len(real_ok),
                                           "recall": rs["recall"]}
                mark = "  <- its synthetic-best" if abs(t - best) < 1e-9 else ""
                p(f"    {t:.2f}: {rs['tp']}/{len(real_ok)}  recall={rs['recall']:.3f}{mark}")
        # The decisive number: the loosest a threshold has to get before this
        # model recovers ALL the real photos — and what that costs in false
        # positives back on the labelled corpus. A model that needs to be run
        # wide open to find real faces has not really won anything.
        full = next((t for t in THRESHOLDS if score(real_ok, t)["recall"] >= 1.0), None)
        full_cost = score(syn, full) if full is not None else None
        if full is not None:
            p(f"  [real] 100% recall first reached at {full:.2f}; "
              f"synthetic there: precision={full_cost['precision']:.3f} "
              f"FP={full_cost['fp']} F1={full_cost['f1']:.3f}")
        else:
            p("  [real] never reaches 100% recall within the sweep")

        if real_amb:
            amb = score(real_amb, best)
            p(f"  [real, ambiguous] {amb['tp']}/{len(real_amb)} at {best:.2f} "
              f"— excluded from the numbers above")
        p("")

        results[name] = {"best_threshold": best, **bs, "elapsed_s": elapsed,
                         "worst_same": max(same), "best_diff": min(diff), "gap": gap,
                         "by_face_px": {str(px): score(by[px], best) for px in sorted(by)},
                         "real_recall": real_recall, "real_full_threshold": full,
                         "real_full_cost": full_cost,
                         "real_n": len(real_ok), "real_amb_n": len(real_amb)}

    models = [(n, r) for n, r in results.items() if n != "_mbf_port_validation"]
    write_results_section("embedders", "Embedder comparison",
                          build_body(models, port_error))


def build_body(models: list[tuple[str, dict]], port_error: float) -> str:
    """The RESULTS.md section. Two populations, reported separately and
    labelled as such — see the note in main() for why they are not pooled."""
    real_n = models[0][1]["real_n"]
    amb_n = models[0][1]["real_amb_n"]

    body = [
        "Same photos, same detector, same crops — only the embedder differs. "
        "Each model gets its own threshold, chosen on the synthetic corpus, "
        "because the two produce different vector spaces and a shared "
        "threshold would mean nothing.\n",
        f"ONNX port validated against insightface's own reference: "
        f"`{port_error:.8f}`. If that is not ~0 the preprocessing is wrong and "
        f"every number below is meaningless.\n",
        "### Synthetic corpus\n",
        "Labelled positives and negatives, so every metric is defined here. "
        "This is what picks each model's threshold.\n",
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
        "\n**Accuracy by face size** (each model at its own best threshold) — "
        "this is what decides whether swapping the model fixes small faces:\n",
        md_table(["face px"] + [n for n, _ in models],
                 [[px] + [f"{r['by_face_px'][px]['accuracy']:.3f}"
                          if px in r["by_face_px"] else "-" for _, r in models]
                  for px in sorted(models[0][1]["by_face_px"], key=int)]),
    ]

    if real_n:
        thresholds = sorted(models[0][1]["real_recall"], key=float)
        bests = {n: f"{r['best_threshold']:.2f}" for n, r in models}
        body += [
            f"\n### Real photos — {real_n} photos\n",
            "The actual product input, and the harder case: phone photos of one "
            "person, not LFW composites. **Every photo contains the target**, so "
            "this measures *recall only* — precision is undefined here (with no "
            "negatives it would read 1.000 by construction and mean nothing). "
            "Each model's synthetic-derived threshold is marked **bold**.\n",
            md_table(["threshold"] + [n for n, _ in models],
                     [[t] + [(lambda c: f"**{c}**" if bests[n] == t else c)(
                         f"{r['real_recall'][t]['matched']}/{r['real_recall'][t]['total']}")
                         for n, r in models]
                      for t in thresholds]),
            "\nThe question this answers: how far does the threshold have to move "
            "before each model recovers the real photos — and what that costs "
            "back on the labelled corpus, where false positives are measurable:\n",
            md_table(["model", "threshold for 8/8 real", "synthetic precision there",
                      "synthetic FP there"],
                     [[n,
                       f"{r['real_full_threshold']:.2f}"
                       if r["real_full_threshold"] is not None else "never",
                       f"{r['real_full_cost']['precision']:.3f}"
                       if r["real_full_cost"] else "—",
                       r["real_full_cost"]["fp"] if r["real_full_cost"] else "—"]
                      for n, r in models]),
        ]
    if amb_n:
        body.append(
            f"\n**Excluded — {amb_n} photos.** Job `4092130d` uses a four-person "
            "group photo as its selfie and production picks a face arbitrarily, "
            "so \"the target\" is not well defined. Kept out of the numbers above.\n"
        )
    return "\n".join(body)


if __name__ == "__main__":
    main()
