"""How accurate is the pipeline? Measured through the real API.

Every corpus job is uploaded the way a phone would: init, presigned PUTs,
multipart zip, complete, process. Nothing reads photos off disk behind the
API's back, so this exercises upload, extraction, detection, embedding and
matching together.

There is no ground-truth file. The answers are already in the filenames
build_corpus.py writes, which the pipeline preserves as original_filename:

    p03_pos_240px.jpg  ->  target present, face 240px wide
    p11_neg_110px.jpg  ->  target absent

Real photos carry no marker; their labels come from REAL_JOBS in
build_corpus.py, where that human judgement already lives.

Each job is processed once at a threshold just under 1.0, so every detected
face comes back with its distance. The sweep is then arithmetic on those
numbers instead of nine more trips through the models.

    venv\\Scripts\\python.exe benchmarks\\build_corpus.py <lfw-path>   # first
    venv\\Scripts\\python.exe benchmarks\\evaluate_corpus.py
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from _common import BACKEND, CORPUS, md_table, p, score, write_results_section  # noqa: E402
from build_corpus import REAL_JOBS  # noqa: E402

API = "http://localhost:8000"
PART_SIZE = 8 * 1024 * 1024        # S3 multipart floor is 5MB per non-final part
COLLECT_THRESHOLD = 0.99           # route requires < 1.0; returns every detected face
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85]
DEFAULT_THRESHOLD = 0.68
POLL_TIMEOUT = 900


def label_from(filename: str) -> tuple[bool | None, int | None]:
    """(contains_target, face_px) read from the name build_corpus.py gave it.
    (None, None) for real photos, which carry no marker."""
    m = re.search(r"_(pos|neg)_(\d+)px", filename)
    if not m:
        return None, None
    return m.group(1) == "pos", int(m.group(2))


def discover_jobs() -> list[dict]:
    jobs = []
    for job_dir in sorted(CORPUS.glob("job*")):
        selfie = job_dir / "selfie" / "selfie.jpg"
        photos = sorted((job_dir / "event_photos").glob("*.jpg"))
        if selfie.exists() and photos:
            jobs.append({"name": job_dir.name, "source": "synthetic", "ambiguous": False,
                         "selfie": selfie, "photos": photos, "all_positive": True})

    uploads = BACKEND / "storage" / "uploads"
    for job_id, meta in REAL_JOBS.items():
        job_dir = uploads / job_id
        selfie = next((job_dir / "selfie").glob("*.jpg"), None)
        photos = sorted((job_dir / "event_photos").glob("*.jpg"))
        if selfie and photos:
            jobs.append({"name": f"real_{job_id[:8]}", "source": "real",
                         "ambiguous": meta["ambiguous"], "selfie": selfie,
                         "photos": photos, "all_positive": meta["target_in_all"]})
    return jobs


def zip_bytes(photos: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for ph in photos:
            zf.write(ph, arcname=ph.name)
    return buf.getvalue()


def put(url: str, body: bytes) -> None:
    """No Content-Type header — it is part of the signed string, and sending
    one the server did not sign gets a 403, or on a large body, a hang."""
    r = requests.put(url, data=body, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"PUT failed {r.status_code}: {r.text[:300]}")


def upload(job: dict) -> str:
    """Init, PUT selfie and zip parts, complete. Returns the job UUID."""
    payload = zip_bytes(job["photos"])
    parts = [payload[i:i + PART_SIZE] for i in range(0, len(payload), PART_SIZE)] or [b""]

    r = requests.post(f"{API}/jobs/upload/init", json={
        "selfie_filename": f"{job['name']}.jpg",   # carries job identity through
        "zip_filename": f"{job['name']}.zip",
        "part_count": len(parts),
    }, timeout=60)
    r.raise_for_status()
    init = r.json()

    put(init["selfie_upload_url"], job["selfie"].read_bytes())
    for entry in sorted(init["zip_upload_urls"], key=lambda e: e["part_number"]):
        put(entry["url"], parts[entry["part_number"] - 1])

    r = requests.post(f"{API}/jobs/{init['job_id']}/upload/complete", timeout=120)
    r.raise_for_status()
    return init["job_id"]


def process_and_wait(job_id: str) -> str:
    r = requests.post(f"{API}/jobs/{job_id}/process",
                      params={"threshold": COLLECT_THRESHOLD}, timeout=60)
    r.raise_for_status()

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        status = requests.get(f"{API}/jobs/{job_id}", timeout=30).json()["status"]
        if status in ("completed", "failed"):
            return status
        time.sleep(2)
    return "timeout"


def distances(job_id: str) -> dict[str, float]:
    data = requests.get(f"{API}/jobs/{job_id}/matches", timeout=60).json()
    return {m["filename"]: m["match_distance"] for m in data["matches"]}


def run_job(job: dict) -> list[dict]:
    job_id = upload(job)
    status = process_and_wait(job_id)
    if status != "completed":
        p(f"  {job['name']}: {status.upper()} (job {job_id[:8]})")
        return []
    found = distances(job_id)

    records = []
    for ph in job["photos"]:
        truth, face_px = label_from(ph.name)
        if truth is None:
            truth = job["all_positive"]
        records.append({"job": job["name"], "source": job["source"],
                        "ambiguous": job["ambiguous"], "truth": truth,
                        "face_px": face_px, "d": found.get(ph.name)})
    p(f"  {job['name']}: {len(records)} photos, {len(found)} faces found "
      f"(job {job_id[:8]})")
    return records


def main() -> None:
    jobs = discover_jobs()
    if not jobs:
        raise SystemExit(f"No corpus at {CORPUS} — run build_corpus.py first")

    # Optional name filters, for a quick partial run. A partial run does not
    # touch RESULTS.md — a subset written into that section would read as the
    # whole corpus.
    wanted = sys.argv[1:]
    if wanted:
        jobs = [j for j in jobs if any(w in j["name"] for w in wanted)]
        if not jobs:
            raise SystemExit(f"no jobs matched {wanted}")
    try:
        requests.get(f"{API}/docs", timeout=10).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"API not reachable at {API} ({exc}) — is docker compose up?")

    p(f"pushing {len(jobs)} jobs through {API}\n")
    started = time.time()
    records: list[dict] = []
    for job in jobs:
        records += run_job(job)
    elapsed = time.time() - started

    synthetic = [r for r in records if r["source"] == "synthetic"]
    real_ok = [r for r in records if r["source"] == "real" and not r["ambiguous"]]
    real_amb = [r for r in records if r["source"] == "real" and r["ambiguous"]]

    body: list[str] = [
        "Every job was uploaded and processed through the running API — "
        "presigned PUTs, multipart zip, worker extraction, the real matcher. "
        "Labels come from the filenames, so there is no ground-truth file to "
        "drift out of sync.\n"
    ]

    s = score(synthetic, DEFAULT_THRESHOLD)
    p(f"\nSYNTHETIC ({len(synthetic)} photos) at {DEFAULT_THRESHOLD}: "
      f"prec={s['precision']:.3f} rec={s['recall']:.3f} F1={s['f1']:.3f}")
    body.append(f"### Synthetic corpus — {len(synthetic)} photos, "
                f"{len({r['job'] for r in synthetic})} identities\n")
    body.append(md_table(
        ["threshold", "precision", "recall", "F1", "accuracy", "TP", "FP", "FN", "TN"],
        [[f"**{t}**" if t == DEFAULT_THRESHOLD else t,
          f"{x['precision']:.3f}", f"{x['recall']:.3f}", f"{x['f1']:.3f}",
          f"{x['accuracy']:.3f}", x["tp"], x["fp"], x["fn"], x["tn"]]
         for t in THRESHOLDS for x in [score(synthetic, t)]]))

    by_size = defaultdict(list)
    for r in synthetic:
        by_size[r["face_px"]].append(r)
    body.append(f"\n**Accuracy by face size** (at {DEFAULT_THRESHOLD}) — face size is the "
                "variable that matters most:\n")
    body.append(md_table(
        ["face px", "accuracy", "TP", "FP", "FN", "TN"],
        [[px, f"{x['accuracy']:.3f}", x["tp"], x["fp"], x["fn"], x["tn"]]
         for px in sorted(k for k in by_size if k is not None)
         for x in [score(by_size[px], DEFAULT_THRESHOLD)]]))

    if real_ok:
        rs = score(real_ok, DEFAULT_THRESHOLD)
        p(f"REAL ({len(real_ok)} photos) at {DEFAULT_THRESHOLD}: recall={rs['recall']:.3f}")
        body.append(f"\n### Real photos — {len(real_ok)} photos\n")
        body.append("All positives, so this measures **recall only** — there are no "
                    "negatives to get wrong.\n")
        body.append(md_table(
            ["threshold", "recall", "matched", "missed"],
            [[f"**{t}**" if t == DEFAULT_THRESHOLD else t,
              f"{x['recall']:.3f}", x["tp"], x["fn"]]
             for t in THRESHOLDS for x in [score(real_ok, t)]]))

    if real_amb:
        body.append(f"\n### Excluded — {len(real_amb)} photos\n")
        body.append("Job `4092130d` uses a four-person group photo as its selfie and "
                    "production picks a face arbitrarily, so \"the target\" is not "
                    "well defined. Reported here, kept out of the numbers above.\n")
        ra = score(real_amb, DEFAULT_THRESHOLD)
        body.append(f"At {DEFAULT_THRESHOLD}: matched {ra['tp']}/{len(real_amb)}.")

    body.append(f"\n_{len(jobs)} jobs through the API in {elapsed:.0f}s._")
    if wanted:
        p(f"\npartial run ({', '.join(j['name'] for j in jobs)}) — RESULTS.md not touched")
    else:
        write_results_section("evaluate", "Matching accuracy", "\n".join(body))


if __name__ == "__main__":
    main()
