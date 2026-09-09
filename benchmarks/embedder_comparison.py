"""Runnable comparison of the two candidate *embedding* models for SnapFind.

Companion to detector_comparison.py, which answered a different question.
That script varied the DETECTOR and held the embedder fixed (ArcFace via
DeepFace) — so its embeddings were always directly comparable, and
"agreement distance near 0" was a valid pass/fail. This script does the
opposite: it holds the detector fixed (insightface/SCRFD, what production
already uses) and varies the EMBEDDER.

That flip changes what can be measured. Two different embedders produce
vectors in different coordinate systems, so the cosine distance BETWEEN
them is meaningless — a near-1.0 distance would say nothing about quality.
What is comparable is the DECISION each one makes: for the same selfie and
the same event photos, does each embedder pick out the same set of matches?

  detector_comparison.py:  vary detector, fixed embedder -> compare vectors
  embedder_comparison.py:  vary embedder, fixed detector -> compare decisions

Candidates:
  A. DeepFace ArcFace  - current production. TensorFlow, arcface_weights.h5,
                         137 MB. Cannot run on Android (no Python/TF).
  B. w600k_mbf.onnx    - MobileFaceNet, ArcFace-trained, 13.6 MB. Already
                         sitting in storage/insightface/models/buffalo_sc/
                         (it ships inside the buffalo_sc pack downloaded for
                         detection) but never loaded — face_matcher.py passes
                         allowed_modules=["detection"]. Runs on ONNX Runtime,
                         which has an Android build.

This script CHANGES NOTHING in the production pipeline. It imports settings
read-only, reads the same weights production already downloaded, and writes
its results next to itself. services/face_matcher.py is untouched — if B
loses, nothing needs reverting.

Two deliberate deviations from production, both because this measures the
ON-DEVICE path, not the current server path:

  1. The selfie is detected with SCRFD, not mtcnn. Production uses mtcnn for
     selfies (face_matcher.py _selfie_embedding), but mtcnn is TensorFlow and
     cannot go on a phone — so on device, det_500m.onnx has to handle the
     selfie too. Measuring the mtcnn path here would measure a pipeline that
     can never ship on Android.
  2. Both embedders receive the byte-identical aligned 112x112 crop. Any
     difference in the results is therefore attributable to the embedder
     alone, not to detection or alignment.

Preprocessing for B is copied from this repo's own venv, not from memory:
venv/Lib/site-packages/insightface/model_zoo/arcface_onnx.py lines 37-83 —
input_mean=127.5, input_std=127.5, swapRB=True. Getting any of those wrong
produces confident-looking 512-D vectors that are quietly meaningless, with
no exception raised, which is the single most common way this port fails.

Requires this backend's virtual environment (uses only dependencies already
in requirements.txt — onnxruntime arrives as an insightface dependency).

Usage (from the Face_recognition/ repo root):
    venv\\Scripts\\python.exe benchmarks\\embedder_comparison.py
    venv\\Scripts\\python.exe benchmarks\\embedder_comparison.py C:\\some\\folder

Reads event photos from benchmarks/sample_photos/ (same gitignored corpus
detector_comparison.py uses) and the selfie from benchmarks/sample_photos/selfie/.
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
# services/face_matcher.py and detector_comparison.py do — otherwise both
# libraries fall back to their own default cache dirs and re-download
# weights this repo already has.
settings.deepface_home.mkdir(parents=True, exist_ok=True)
os.environ["DEEPFACE_HOME"] = str(settings.deepface_home)
settings.insightface_home.mkdir(parents=True, exist_ok=True)
os.environ["INSIGHTFACE_HOME"] = str(settings.insightface_home)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEEPFACE_MODEL = "ArcFace"
MBF_PATH = settings.insightface_home / "models" / "buffalo_sc" / "w600k_mbf.onnx"
RESULTS_PATH = Path(__file__).resolve().parent / "embedder_results.json"

# insightface/model_zoo/arcface_onnx.py:37-41 and :82-83.
MBF_INPUT_MEAN = 127.5
MBF_INPUT_STD = 127.5
MBF_INPUT_SIZE = (112, 112)

# Production's threshold (0.68, api/routes.py) is calibrated for DeepFace's
# ArcFace specifically. It is NOT transferable to a different embedder, so
# rather than pick a number for w600k_mbf out of the air, sweep a range and
# print the match set each threshold would produce. Separation between
# same-person and different-person distances is what matters here; a single
# calibrated number needs labelled pairs this repo does not have yet.
THRESHOLD_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 1.0, 1.1, 1.2]


def p(msg: str = "") -> None:
    print(msg, flush=True)


# Same corpus and same convention as detector_comparison.py: event photos sit
# flat in benchmarks/sample_photos/ (gitignored — they're photos of real
# people and this repo is public). This benchmark additionally needs a selfie
# to match against, which lives in a selfie/ subdirectory so it stays out of
# the event-photo list.
#
# Deliberately NOT settings.upload_root: that attribute no longer exists
# (core/settings.py dropped it when object storage replaced local uploads),
# and a benchmark wants a fixed offline corpus anyway — no MinIO container,
# no Postgres, no network.
SAMPLES_DIR = Path(__file__).resolve().parent / "sample_photos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_photos(folder: Path) -> list[Path]:
    if not folder.exists():
        raise SystemExit(
            f"No such folder: {folder}\n"
            f"Drop a few event photos into {SAMPLES_DIR} (gitignored), or pass "
            "a folder path as the first argument."
        )
    photos = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
    )
    if not photos:
        raise SystemExit(f"No images found in {folder}")
    return photos


def find_selfie(folder: Path) -> Path:
    selfie_dir = folder / "selfie"
    candidates = (
        sorted(f for f in selfie_dir.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        if selfie_dir.is_dir() else []
    )
    if not candidates:
        raise SystemExit(
            f"No selfie found in {selfie_dir}\n"
            "This benchmark matches event photos against one selfie — put the "
            f"selfie in {selfie_dir} (it is excluded from the event-photo list)."
        )
    return candidates[0]


def load_resized(image_path: Path, max_dim: int) -> np.ndarray:
    """Same downscale production does in face_matcher._load_resized_image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not decode {image_path}")
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        image = cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return float("nan")
    return 1 - max(min(dot / (na * nb), 1.0), -1.0)


# --- Shared front-end: detect + align. Held CONSTANT across both embedders --

def build_detector():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_sc",
        allowed_modules=["detection"],
        root=str(settings.insightface_home),
    )
    app.prepare(ctx_id=-1, det_size=(settings.event_photo_max_dimension,) * 2)
    return app


def detect_and_align(app, image: np.ndarray):
    """Returns (aligned_112x112_bgr, face_count, detect_ms) or (None, 0, ms).

    Largest-face-only, mirroring production (face_matcher.py:174) so this
    compares embedders rather than embedder + face-selection policy. Note
    that policy is itself a known recall gap for group photos — out of scope
    here, deliberately, so the one variable under test stays the one variable.
    """
    from insightface.utils import face_align

    start = time.perf_counter()
    faces = app.get(image)
    detect_ms = (time.perf_counter() - start) * 1000
    if not faces:
        return None, 0, detect_ms
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    aligned = face_align.norm_crop(image, best.kps, image_size=112, mode="arcface")
    return aligned, len(faces), detect_ms


# --- Embedder A: DeepFace ArcFace (current production, TensorFlow) ----------

def embed_deepface(aligned: np.ndarray) -> list[float]:
    from deepface import DeepFace

    result = DeepFace.represent(
        img_path=aligned,
        model_name=DEEPFACE_MODEL,
        detector_backend="skip",  # crop is already aligned, same as production
        enforce_detection=False,
    )
    return result[0]["embedding"]


# --- Embedder B: w600k_mbf.onnx (MobileFaceNet, ONNX Runtime) --------------

def make_mbf_session(intra_threads: int | None = None):
    import onnxruntime as ort

    if not MBF_PATH.exists():
        raise SystemExit(
            f"Missing {MBF_PATH}\n"
            "It ships inside the buffalo_sc pack — run detector_comparison.py "
            "once to trigger the download."
        )
    opts = ort.SessionOptions()
    if intra_threads is not None:
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(MBF_PATH), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def embed_mbf(session, aligned: np.ndarray) -> list[float]:
    # (x - 127.5) / 127.5, BGR->RGB, NCHW. See module docstring for source.
    blob = cv2.dnn.blobFromImage(
        aligned, 1.0 / MBF_INPUT_STD, MBF_INPUT_SIZE,
        (MBF_INPUT_MEAN, MBF_INPUT_MEAN, MBF_INPUT_MEAN), swapRB=True,
    )
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return out[0].tolist()


def time_embed(fn, aligned: np.ndarray, repeats: int = 3) -> float:
    """Best-of-N ms. Best rather than mean: we want the model's cost, not the
    machine's background noise."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(aligned)
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


# --- Ground truth ----------------------------------------------------------
# Established by visually inspecting the aligned 112x112 crops — the exact
# pixels the models receive — not by reading filenames or trusting distances.
# Keyed by folder name, so pointing the script at a different corpus means
# adding an entry rather than editing logic.
#
# The sample photos all show the same person ("A") except P1, which
# contains TWO people. The second person ("B") has the slightly LARGER face
# (8250 px2 vs 7770 px2 at full resolution), so production's largest-face rule
# (face_matcher.py:174) scores the wrong person for that photo.
#
# This is the labelled data the repo previously did not have. Without at least
# one different-person pair, no threshold can be evaluated at all: you can
# always reach 100% recall by accepting everything.
GROUND_TRUTH: dict[str, dict[str, str]] = {
    "sample_photos": {
        "SELFIE.f0": "A",
        "P1.f0": "B",  # larger face, different person — what production picks
        "P1.f1": "A",  # the actual target, smaller by 6%
        "P2.f0": "A",
        "P3.f0": "A",
        "P4.f0": "A",
    },
}

RESOLUTION_SWEEP = [800, 1200, 1600, 2000, 4000]


def all_faces(app, image: np.ndarray):
    """Every detected face, largest first, aligned. Production takes [0] only."""
    from insightface.utils import face_align

    faces = sorted(
        app.get(image),
        key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    out = []
    for f in faces:
        area = int((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        out.append((face_align.norm_crop(image, f.kps, image_size=112, mode="arcface"),
                    area, float(f.det_score)))
    return out


def stage_resolution_sweep(app, mbf, sel_df, sel_mbf, event_photos) -> list[dict]:
    """Does production's 800px downscale cost recall?

    Only the crop SOURCE resolution varies — insightface still runs detection
    at det_size=(800,800) and scales landmarks back, so this isolates 'how many
    real pixels does the 112x112 crop come from' from 'how well can the
    detector find the face'. Those are two separate levers.
    """
    p("\n=== Stage 7: crop resolution vs match distance ===")
    p("Production downscales event photos to 800px (settings.event_photo_max_dimension).")
    p("A 4000px photo at 800px turns a 90px face into an 18px face.")
    p()
    p(f"  {'photo':<7}{'max_dim':<9}{'face_px2':>10}{'deepface':>11}{'mbf':>9}")
    rows = []
    for i, photo in enumerate(event_photos, start=1):
        for dim in RESOLUTION_SWEEP:
            faces = all_faces(app, load_resized(photo, dim))
            if not faces:
                p(f"  {'P%d' % i:<7}{dim:<9}{'NO FACE':>10}{'-':>11}{'-':>9}")
                continue
            crop, area, _ = faces[0]  # largest, mirroring production
            d_df = cosine_distance(sel_df, embed_deepface(crop))
            d_mbf = cosine_distance(sel_mbf, embed_mbf(mbf, crop))
            p(f"  {'P%d' % i:<7}{dim:<9}{area:>10}{d_df:>11.4f}{d_mbf:>9.4f}")
            rows.append({"photo": f"P{i}", "max_dim": dim, "face_px2": area,
                         "deepface": d_df, "mbf": d_mbf})
        p()
    return rows


def stage_labelled_pairs(app, mbf, sel_df, sel_mbf, job_dir, event_photos) -> dict:
    """Score EVERY face (not just the largest) against the labelled identities.

    This is the only stage that can say anything about thresholds, because it
    is the only one with different-person pairs in it.
    """
    p("=== Stage 8: labelled pairs, every face at full resolution ===")
    truth = GROUND_TRUTH.get(job_dir.name)
    if not truth:
        p(f"  No ground-truth labels for job {job_dir.name} — skipping.")
        p("  Add an entry to GROUND_TRUTH after inspecting the crops visually.")
        return {}

    embeddings = {"SELFIE.f0": (sel_df, sel_mbf)}
    areas: dict[str, int] = {}
    for i, photo in enumerate(event_photos, start=1):
        for j, (crop, area, _) in enumerate(all_faces(app, load_resized(photo, 4000))):
            key = f"P{i}.f{j}"
            embeddings[key] = (embed_deepface(crop), embed_mbf(mbf, crop))
            areas[key] = area

    unlabelled = [k for k in embeddings if k not in truth]
    if unlabelled:
        p(f"  NOTE: {len(unlabelled)} face(s) with no label, excluded: {unlabelled}")

    keys = [k for k in embeddings if k in truth]
    same_df, same_mbf, diff_df, diff_mbf = [], [], [], []
    p(f"\n  {'pair':<26}{'people':<10}{'deepface':>11}{'mbf':>9}")
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1:]:
            d_df = cosine_distance(embeddings[a][0], embeddings[b][0])
            d_mbf = cosine_distance(embeddings[a][1], embeddings[b][1])
            is_same = truth[a] == truth[b]
            (same_df if is_same else diff_df).append(d_df)
            (same_mbf if is_same else diff_mbf).append(d_mbf)
            tag = "SAME" if is_same else "DIFFERENT"
            p(f"  {a + ' vs ' + b:<26}{tag:<10}{d_df:>11.4f}{d_mbf:>9.4f}")

    p("\n  Separation — a usable threshold exists only if the worst same-person")
    p("  pair is closer than the best different-person pair:")
    p()
    p(f"  {'embedder':<12}{'worst SAME':>12}{'best DIFF':>12}{'gap':>9}   verdict")
    summary = {}
    for name, s, d in (("deepface", same_df, diff_df), ("mbf", same_mbf, diff_mbf)):
        if not s or not d:
            continue
        worst_same, best_diff = max(s), min(d)
        gap = best_diff - worst_same
        ok = gap > 0
        verdict = (f"separable, e.g. threshold {(worst_same + best_diff) / 2:.2f}"
                   if ok else "NO threshold separates these")
        p(f"  {name:<12}{worst_same:>12.4f}{best_diff:>12.4f}{gap:>9.4f}   {verdict}")
        summary[name] = {"worst_same": worst_same, "best_diff": best_diff,
                         "gap": gap, "separable": ok,
                         "suggested_threshold": (worst_same + best_diff) / 2 if ok else None}
    return {"areas": areas, "separation": summary}


def stage_deepface_config(app, job_dir, event_photos, sel_crop) -> list[dict]:
    """Is DeepFace losing because of HOW production calls it?

    Production hands DeepFace a BGR crop straight from cv2/norm_crop
    (face_matcher.py:175-179) and leaves `normalization` at its "base"
    default. Both are plausible bugs — OpenCV is BGR while most models expect
    RGB, and DeepFace ships an ArcFace-specific normalization it isn't being
    asked for. Before concluding anything about the model, rule those out:
    a fair comparison has to give the incumbent its best configuration.
    """
    import itertools

    from deepface import DeepFace

    p("\n=== Stage 9: is production calling DeepFace correctly? ===")
    truth = GROUND_TRUTH.get(job_dir.name)
    if not truth:
        p("  No ground-truth labels for this job — skipping.")
        return []

    crops = {"SELFIE.f0": sel_crop}
    for i, photo in enumerate(event_photos, start=1):
        for j, (crop, _, _) in enumerate(all_faces(app, load_resized(photo, 4000))):
            crops[f"P{i}.f{j}"] = crop

    p(f"\n  {'config':<28}{'worst SAME':>12}{'best DIFF':>12}{'gap':>9}   verdict")
    rows = []
    for order in ("BGR (production)", "RGB"):
        for norm in ("base", "ArcFace"):
            embs = {}
            for key, crop in crops.items():
                img = crop if order.startswith("BGR") else crop[:, :, ::-1]
                embs[key] = DeepFace.represent(
                    img_path=img, model_name=DEEPFACE_MODEL, detector_backend="skip",
                    enforce_detection=False, normalization=norm,
                )[0]["embedding"]
            keys = [k for k in embs if k in truth]
            same, diff = [], []
            for a, b in itertools.combinations(keys, 2):
                d = cosine_distance(embs[a], embs[b])
                (same if truth[a] == truth[b] else diff).append(d)
            worst_same, best_diff = max(same), min(diff)
            gap = best_diff - worst_same
            verdict = (f"separable @ {(worst_same + best_diff) / 2:.2f}"
                       if gap > 0 else "NOT separable")
            label = f"{order} / {norm}"
            p(f"  {label:<28}{worst_same:>12.4f}{best_diff:>12.4f}{gap:>9.4f}   {verdict}")
            rows.append({"color_order": order, "normalization": norm,
                         "worst_same": worst_same, "best_diff": best_diff,
                         "gap": gap, "separable": gap > 0})
    p("\n  If production's row is the worst of the four, that is a live accuracy")
    p("  bug in services/face_matcher.py independent of anything on-device.")
    return rows


def main() -> None:
    job_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SAMPLES_DIR
    event_photos = find_photos(job_dir)
    selfie_path = find_selfie(job_dir)

    p(f"Folder: {job_dir}")
    p(f"Selfie: {selfie_path.name}")
    p(f"Photos: {len(event_photos)}")
    p()
    p("Detector held CONSTANT (insightface/SCRFD det_500m.onnx) for both.")
    p("Selfie detected with SCRFD, not mtcnn — mtcnn is TensorFlow and cannot")
    p("run on Android, so this measures the path that could actually ship.")

    app = build_detector()

    # ---- Detect + align everything once; both embedders get identical crops.
    p("\n=== Stage 1: detect + align (shared front-end) ===")
    crops: dict[str, np.ndarray] = {}
    detect_times: dict[str, float] = {}
    labels: dict[str, str] = {}

    aligned, count, ms = detect_and_align(app, load_resized(selfie_path, settings.selfie_max_dimension))
    if aligned is None:
        raise SystemExit(
            f"SCRFD found no face in the selfie ({selfie_path.name}). "
            "On-device would fail here too — worth knowing before anything else."
        )
    crops["SELFIE"] = aligned
    detect_times["SELFIE"] = ms
    labels["SELFIE"] = selfie_path.name
    p(f"  SELFIE  {selfie_path.name}: {count} face(s), {ms:.0f}ms")

    for i, photo in enumerate(event_photos, start=1):
        key = f"P{i}"
        aligned, count, ms = detect_and_align(
            app, load_resized(photo, settings.event_photo_max_dimension)
        )
        labels[key] = photo.name
        detect_times[key] = ms
        if aligned is None:
            p(f"  {key}      {photo.name}: NO FACE DETECTED, {ms:.0f}ms")
            continue
        crops[key] = aligned
        p(f"  {key}      {photo.name}: {count} face(s), {ms:.0f}ms")

    photo_keys = [k for k in crops if k != "SELFIE"]
    if not photo_keys:
        raise SystemExit("No event photo produced a face — nothing to compare.")

    # ---- Embed the same crops with both models.
    p("\n=== Stage 2: embed (the one variable under test) ===")
    mbf = make_mbf_session()
    mbf_1t = make_mbf_session(intra_threads=1)

    # Warm both up before timing — same reason production warms DeepFace on
    # the selfie first (face_matcher.py:250). First call pays graph
    # optimization and allocation that steady-state never pays again.
    warm = crops["SELFIE"]
    embed_deepface(warm)
    embed_mbf(mbf, warm)
    embed_mbf(mbf_1t, warm)

    emb_deepface: dict[str, list[float]] = {}
    emb_mbf: dict[str, list[float]] = {}
    t_deepface: dict[str, float] = {}
    t_mbf: dict[str, float] = {}
    t_mbf_1t: dict[str, float] = {}

    for key, aligned in crops.items():
        emb_deepface[key] = embed_deepface(aligned)
        emb_mbf[key] = embed_mbf(mbf, aligned)
        t_deepface[key] = time_embed(embed_deepface, aligned)
        t_mbf[key] = time_embed(lambda a: embed_mbf(mbf, a), aligned)
        t_mbf_1t[key] = time_embed(lambda a: embed_mbf(mbf_1t, a), aligned)
        p(f"  {key}: deepface {t_deepface[key]:7.1f}ms | "
          f"mbf {t_mbf[key]:6.1f}ms | mbf(1 thread) {t_mbf_1t[key]:6.1f}ms")

    p(f"\n  Embedding dims: deepface={len(emb_deepface['SELFIE'])}, "
      f"mbf={len(emb_mbf['SELFIE'])}")

    # ---- Selfie -> photo distances, per embedder.
    p("\n=== Stage 3: selfie-to-photo distance (NOT comparable across columns) ===")
    p("Each embedder has its own coordinate system. Read DOWN a column for")
    p("separation between photos; comparing ACROSS columns is meaningless.")
    p()
    p(f"  {'':<5}{'photo':<40}{'deepface':>10}{'mbf':>10}")
    dist_deepface: dict[str, float] = {}
    dist_mbf: dict[str, float] = {}
    for key in photo_keys:
        dist_deepface[key] = cosine_distance(emb_deepface["SELFIE"], emb_deepface[key])
        dist_mbf[key] = cosine_distance(emb_mbf["SELFIE"], emb_mbf[key])
        p(f"  {key:<5}{labels[key][:38]:<40}{dist_deepface[key]:>10.4f}{dist_mbf[key]:>10.4f}")

    # ---- Threshold sweep: which photos each embedder would return.
    p("\n=== Stage 4: match set by threshold ===")
    p("This is the server-side rehearsal of the on-device acceptance test:")
    p("same job in, which photos come out. A good embedder shows a WIDE band")
    p("of thresholds where the set is stable and correct.")
    p()
    p(f"  {'threshold':<12}{'deepface':<28}{'mbf':<28}")
    sweep = []
    for t in THRESHOLD_SWEEP:
        set_df = [k for k in photo_keys if dist_deepface[k] <= t]
        set_mbf = [k for k in photo_keys if dist_mbf[k] <= t]
        marker = "  <- production" if t == 0.68 else ""
        p(f"  {t:<12.2f}{' '.join(set_df) or '(none)':<28}"
          f"{' '.join(set_mbf) or '(none)':<28}{marker}")
        sweep.append({
            "threshold": t,
            "deepface_matches": [labels[k] for k in set_df],
            "mbf_matches": [labels[k] for k in set_mbf],
            "agree": set_df == set_mbf,
        })

    # ---- Pairwise matrix: does each embedder separate these faces the same way?
    p("\n=== Stage 5: pairwise distance matrix ===")
    p("All faces against each other. Same-person pairs should be low, ")
    p("different-person pairs high. Look for the same STRUCTURE in both, ")
    p("not the same numbers.")
    all_keys = ["SELFIE"] + photo_keys
    for name, embs in (("deepface", emb_deepface), ("mbf", emb_mbf)):
        p(f"\n  {name}:")
        p("        " + "".join(f"{k:>9}" for k in all_keys))
        for a in all_keys:
            row = "".join(f"{cosine_distance(embs[a], embs[b]):>9.4f}" for b in all_keys)
            p(f"  {a:<6}{row}")

    # ---- Timing summary.
    n = len(crops)
    avg_det = sum(detect_times[k] for k in crops) / n
    avg_df = sum(t_deepface.values()) / n
    avg_mbf = sum(t_mbf.values()) / n
    avg_mbf_1t = sum(t_mbf_1t.values()) / n

    p("\n=== Stage 6: timing ===")
    p(f"  detect + align (shared):        {avg_det:8.1f} ms/photo")
    p(f"  embed, deepface ArcFace:        {avg_df:8.1f} ms/photo")
    p(f"  embed, w600k_mbf (all threads): {avg_mbf:8.1f} ms/photo")
    p(f"  embed, w600k_mbf (1 thread):    {avg_mbf_1t:8.1f} ms/photo   <- phone-core proxy")
    if avg_mbf > 0:
        p(f"\n  embedder speedup: {avg_df / avg_mbf:.1f}x (all threads), "
          f"{avg_df / avg_mbf_1t:.1f}x (1 thread)")
    p(f"  model size: arcface_weights.h5 137 MB  vs  w600k_mbf.onnx 13.6 MB")

    est_500 = (avg_det + avg_mbf_1t) * 500 / 1000
    p(f"\n  Very rough 500-photo estimate, 1 core, THIS machine: {est_500:.0f}s")
    p("  Not a phone number. No thermal throttling, no JPEG decode of full-size")
    p("  originals, different CPU. Real figure needs the on-device harness.")

    sweep_rows = stage_resolution_sweep(app, mbf, emb_deepface["SELFIE"],
                                        emb_mbf["SELFIE"], event_photos)
    labelled = stage_labelled_pairs(app, mbf, emb_deepface["SELFIE"],
                                    emb_mbf["SELFIE"], job_dir, event_photos)
    df_configs = stage_deepface_config(app, job_dir, event_photos, crops["SELFIE"])

    # ---- Caveats. Printed, not buried in a README.
    p("\n=== What this does NOT establish ===")
    p("  - ONE identity, and only two different-person pairs. Enough to show a")
    p("    direction, nowhere near enough to trust a specific threshold.")
    p("  - All photos from a single person on a single trip: one outfit, one")
    p("    hairstyle, similar lighting. No coverage of age, ethnicity or")
    p("    lighting diversity, which is where face models usually fail.")
    p("  - Timings are this dev machine's CPU. Not a phone: no thermal")
    p("    throttling, no full-size JPEG decode, different instruction set.")
    p("  - Stages 1-6 use largest-face-only to mirror production; stage 8 shows")
    p("    what that rule costs, but the production path is unchanged.")

    RESULTS_PATH.write_text(json.dumps({
        "folder": str(job_dir),
        "selfie": selfie_path.name,
        "photo_count": len(event_photos),
        "faces_found": len(photo_keys),
        "labels": labels,
        "detector": "insightface SCRFD det_500m.onnx (constant)",
        "embedders": {
            "deepface": {"model": "ArcFace (TensorFlow)", "size_mb": 137,
                         "dims": len(emb_deepface["SELFIE"]),
                         "avg_ms": avg_df, "android_capable": False},
            "mbf": {"model": "w600k_mbf.onnx (MobileFaceNet)", "size_mb": 13.6,
                    "dims": len(emb_mbf["SELFIE"]),
                    "avg_ms": avg_mbf, "avg_ms_1_thread": avg_mbf_1t,
                    "android_capable": True},
        },
        "avg_detect_ms": avg_det,
        "selfie_distances": {
            labels[k]: {"deepface": dist_deepface[k], "mbf": dist_mbf[k]}
            for k in photo_keys
        },
        "threshold_sweep": sweep,
        "resolution_sweep": sweep_rows,
        "labelled_pairs": labelled,
        "deepface_config_variants": df_configs,
        "caveats": [
            "one identity, two different-person pairs - direction only",
            "single trip: one outfit, one hairstyle, similar lighting",
            "timings are this dev machine's CPU, not a phone",
            "stages 1-6 mirror production's largest-face rule; stage 8 shows its cost",
        ],
    }, indent=2))
    p(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
