"""Shared helpers, so the four benchmark scripts don't each carry a copy.

Also owns RESULTS.md: every script writes its own named section and leaves
the others untouched, so they can be run independently and in any order.
"""
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent
BACKEND = BENCHMARKS.parent
SAMPLE_DATA = BENCHMARKS / "sample_test_data"
RESULTS_MD = BENCHMARKS / "RESULTS.md"

sys.path.insert(0, str(BACKEND))

# The real jobs, keyed by the job id they were uploaded under. Synthetic jobs
# carry their labels in the filename; these photos don't, so the human
# judgement lives here — the one thing no script can derive.
REAL_JOBS = {
    "24c8ea7c-093a-4694-b33d-9a4585978a98": {
        "target_in_all": True,   # correct person, clearly usable
        "ambiguous": False,
    },
    "6fe1d707-565a-4f7f-a5b8-4faf75fca594": {
        "target_in_all": True,
        "ambiguous": False,
    },
    "4092130d-f8ff-4218-ab6b-c2fef1d65bab": {
        "target_in_all": True,
        # Selfie is a four-person group shot and production picks a face
        # arbitrarily, so "the target" is not well defined. Reported
        # separately, kept out of headline metrics.
        "ambiguous": True,
    },
}


def label_from(filename: str) -> tuple[bool | None, int | None]:
    """(contains_target, face_px), read from the name build_corpus.py gave it.

    This is why there is no ground-truth file: `p03_pos_240px.jpg` already
    says everything, and the pipeline preserves it as original_filename.
    Returns (None, None) for real photos, which carry no marker.
    """
    m = re.search(r"_(pos|neg)_(\d+)px", filename)
    if not m:
        return None, None
    return m.group(1) == "pos", int(m.group(2))


def discover_jobs() -> list[dict]:
    """Every job in sample_test_data — synthetic first, then real.

    Synthetic folders are named `job01_...`; real ones are named by their
    job id and must appear in REAL_JOBS to be labelled.
    """
    if not SAMPLE_DATA.exists():
        raise SystemExit(f"No test data at {SAMPLE_DATA} — run build_corpus.py first")

    jobs = []
    for job_dir in sorted(SAMPLE_DATA.glob("job*")):
        selfie = job_dir / "selfie" / "selfie.jpg"
        photos = sorted((job_dir / "event_photos").glob("*.jpg"))
        if selfie.exists() and photos:
            jobs.append({"name": job_dir.name, "source": "synthetic", "ambiguous": False,
                         "selfie": selfie, "photos": photos, "all_positive": True})

    for job_id, meta in REAL_JOBS.items():
        job_dir = SAMPLE_DATA / job_id
        selfie = next((job_dir / "selfie").glob("*.jpg"), None) if job_dir.exists() else None
        photos = sorted((job_dir / "event_photos").glob("*.jpg")) if job_dir.exists() else []
        if selfie and photos:
            jobs.append({"name": f"real_{job_id[:8]}", "source": "real",
                         "ambiguous": meta["ambiguous"], "selfie": selfie,
                         "photos": photos, "all_positive": meta["target_in_all"]})
    return jobs


def seeded_job_ids() -> dict[str, str]:
    """corpus job name -> the job id the API assigned when it was seeded.

    Read from Postgres rather than a local file: the id is minted by the
    server and exists nowhere else, and the database is already the record.
    A file would just be a copy that goes stale the next time you re-seed.
    The link is `selfie_filename`, which seed_jobs.py sets to the job name.
    """
    from sqlalchemy import create_engine, text

    from core.settings import settings

    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT selfie_filename, id FROM upload_jobs ORDER BY created_at"
            )).all()
    finally:
        engine.dispose()
    # Later seeds win, so re-seeding without wiping still resolves correctly.
    return {Path(name).stem: job_id for name, job_id in rows}


def wipe_cached_embeddings() -> None:
    """Clear EventPhoto.embedding for every seeded photo.

    build_matches_for_job() caches each photo's winning embedding and reuses
    it on the next /process call (services/face_matcher.py:251-253) — fine
    for production, but it means re-processing the same job under a
    different SELFIE_DETECTOR_MODE/EMBEDDER would silently score the
    PREVIOUS config's embeddings. Call this between sweep configs.
    """
    from sqlalchemy import create_engine, text

    from core.settings import settings

    engine = create_engine(settings.database_url)
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE event_photos SET embedding = NULL"))
    finally:
        engine.dispose()


def _record(job: dict, filename: str, d: float | None) -> dict:
    """One scored photo, in the shape every scorer and both real pipelines
    (backend, on-device) agree on. The only thing that differs between
    sources is how `d` was obtained — this is what stays common."""
    truth, face_px = label_from(filename)
    if truth is None:
        truth = job["all_positive"]
    return {"job": job["name"], "source": job["source"],
            "ambiguous": job["ambiguous"], "truth": truth,
            "face_px": face_px, "file": filename, "d": d}


def records_for(job: dict, distance_of) -> list[dict]:
    """Score one job's photos by calling in-process. `distance_of(job, photo_path)`
    returns the cosine distance, or None when no face was found. Use this when
    the pipeline can be called directly, e.g. a local ONNX session."""
    return [_record(job, ph.name, distance_of(job, ph)) for ph in job["photos"]]


def records_from_distances(job: dict, distances: dict[str, float | None]) -> list[dict]:
    """Score one job's photos from precomputed distances, keyed by filename.

    For sources that cannot be called in-process from this script — the live
    API (`{m["filename"]: m["match_distance"]}` from /jobs/{id}/matches) and,
    once it exists, an on-device run (a CSV pulled off the phone: job,file,d).
    A filename missing from `distances` scores as "no match" (d=None), same
    as a face that was never found.
    """
    return [_record(job, ph.name, distances.get(ph.name)) for ph in job["photos"]]


def load_device_distances(csv_path: Path) -> dict[str, dict[str, float | None]]:
    """Read an on-device export: columns job,file,d (d blank = no match).

    This is the contract for step 3/4's Android run: compute distances on
    the phone, export this CSV, `adb pull` it, and score it with the exact
    same records_from_distances()/score() as the backend and desktop
    numbers — one scoring path for all three sources, nothing to reconcile.
    Grouped by job name so a caller can do:
        by_job = load_device_distances(path)
        for job in jobs:
            records += records_from_distances(job, by_job.get(job["name"], {}))
    """
    out: dict[str, dict[str, float | None]] = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = float(row["d"]) if row.get("d") not in (None, "") else None
            out[row["job"]][row["file"]] = d
    return dict(out)


def p(msg: str = "") -> None:
    print(msg, flush=True)


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1 - dot / (na * nb)


def score(records: list[dict], threshold: float) -> dict:
    """records: [{"truth": bool, "d": float|None}, ...] — d is None when no
    face was found, which counts as "no match"."""
    tp = sum(1 for r in records if r["truth"] and r["d"] is not None and r["d"] <= threshold)
    fp = sum(1 for r in records if not r["truth"] and r["d"] is not None and r["d"] <= threshold)
    fn = sum(1 for r in records if r["truth"] and (r["d"] is None or r["d"] > threshold))
    tn = sum(1 for r in records if not r["truth"] and (r["d"] is None or r["d"] > threshold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f1": f1, "accuracy": (tp + tn) / max(1, len(records))}


def aligned_faces(photo_path: Path) -> list:
    """Production's crops: detect on the downscaled image, crop from the original."""
    import cv2
    from insightface.utils import face_align

    from core.settings import settings
    from services import face_matcher as fm

    original = cv2.imread(str(photo_path))
    if original is None:
        return []
    detect_image = fm._downscale(original, settings.event_photo_max_dimension)
    faces = fm._get_insightface_app().get(detect_image)
    if not faces:
        return []
    scale = original.shape[1] / detect_image.shape[1]
    return [
        face_align.norm_crop(
            original, f.kps * scale if scale != 1 else f.kps, image_size=112, mode="arcface"
        )
        for f in faces
    ]


def write_results_section(key: str, title: str, body: str) -> None:
    """Replace (or append) one section of RESULTS.md by name.

    Each script owns one section, so running one doesn't wipe another's
    numbers. Sections are delimited by HTML comments.
    """
    start, end = f"<!-- {key}:start -->", f"<!-- {key}:end -->"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"{start}\n## {title}\n\n_generated {stamp}_\n\n{body.rstrip()}\n{end}"

    header = (
        "# Benchmark results\n\n"
        "Generated by the scripts in this folder — do not edit by hand.\n"
        "Each section is rewritten by its own script; see README.md for what\n"
        "each one measures and the caveats that apply.\n"
    )
    existing = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else header

    if start in existing and end in existing:
        updated = re.sub(
            re.escape(start) + r".*?" + re.escape(end), lambda _: block, existing, flags=re.S
        )
    else:
        updated = existing.rstrip() + "\n\n---\n\n" + block + "\n"

    RESULTS_MD.write_text(updated, encoding="utf-8")
    p(f"\nwrote '{key}' section -> {RESULTS_MD}")


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)
