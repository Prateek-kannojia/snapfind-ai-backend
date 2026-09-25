"""How accurate is the pipeline? Measured through the real API.

Re-processes the jobs already seeded by seed_jobs.py and scores what comes
back. Nothing is uploaded here and nothing is read off disk behind the API's
back — the numbers come from /process and /matches on real jobs whose photos
live in object storage.

Run seed_jobs.py first (once, or after a database wipe).

There is no ground-truth file. The answers are already in the filenames
build_corpus.py writes, which the pipeline preserves as original_filename:

    p03_pos_240px.jpg  ->  target present, face 240px wide
    p11_neg_110px.jpg  ->  target absent

Real photos carry no marker; their labels come from REAL_JOBS in _common.py,
where that human judgement already lives.

Each job is processed once at a threshold just under 1.0, so every detected
face comes back with its distance. The sweep is then arithmetic on those
numbers instead of nine more trips through the models.

    venv\\Scripts\\python.exe benchmarks\\seed_jobs.py        # once
    venv\\Scripts\\python.exe benchmarks\\evaluate_corpus.py

--sweep runs the whole corpus through the real API once per
(SELFIE_DETECTOR_MODE, EMBEDDER) combo in SWEEP_CONFIGS below — force-
recreating the worker container between configs so it picks up the new
env, and wiping cached embeddings so each config scores its own freshly
computed distances rather than the previous config's cache. Needs `docker
compose` on PATH and the stack already up.

    venv\\Scripts\\python.exe benchmarks\\evaluate_corpus.py --sweep
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from _common import (  # noqa: E402
    BACKEND, discover_jobs, md_table, p, records_from_distances, score,
    seeded_job_ids, wipe_cached_embeddings, write_results_section,
)

API = "http://localhost:8000"
COLLECT_THRESHOLD = 0.99           # route requires < 1.0; returns every detected face
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85]
DEFAULT_THRESHOLD = 0.68
POLL_TIMEOUT = 1800                # per job, wall clock
POLL_INTERVAL = 3
HTTP_TIMEOUT = 120                 # generous: the API competes with the worker for CPU


def _request(method: str, url: str, **kwargs) -> dict | None:
    """One HTTP call, returning None instead of raising on a transport error.

    Callers are polling loops with their own deadline, so a timed-out or
    refused request means "ask again", not "give up on 13 jobs of work".
    """
    try:
        r = requests.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        p(f"    (transient: {type(exc).__name__}; retrying)")
        return None


def process_and_wait(job_id: str) -> str:
    """Queue the job, then poll until it settles.

    The worker saturates the CPU while embedding, which starves the API
    container enough that a status request can occasionally take longer than
    its timeout. That is not a failure — the job is still running — so a slow
    or dropped poll is retried rather than aborting the whole run. Only the
    overall deadline gives up.
    """
    # A failed enqueue means the job never starts, so polling would just burn
    # the deadline. Retry it, then give up loudly rather than quietly.
    for _ in range(3):
        if _request("post", f"{API}/jobs/{job_id}/process",
                    params={"threshold": COLLECT_THRESHOLD}) is not None:
            break
        time.sleep(POLL_INTERVAL)
    else:
        return "could not enqueue"

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        payload = _request("get", f"{API}/jobs/{job_id}")
        if payload is not None and payload["status"] in ("completed", "failed"):
            return payload["status"]
        time.sleep(POLL_INTERVAL)
    return "timeout"


def distances(job_id: str) -> dict[str, float] | None:
    for _ in range(3):
        data = _request("get", f"{API}/jobs/{job_id}/matches")
        if data is not None:
            return {m["filename"]: m["match_distance"] for m in data["matches"]}
        time.sleep(POLL_INTERVAL)
    return None


def run_job(job: dict, job_id: str) -> list[dict]:
    status = process_and_wait(job_id)
    if status != "completed":
        p(f"  {job['name']}: {status.upper()} (job {job_id[:8]})")
        return []
    found = distances(job_id)
    if found is None:
        p(f"  {job['name']}: could not read matches (job {job_id[:8]})")
        return []

    records = records_from_distances(job, found)
    p(f"  {job['name']}: {len(records)} photos, {len(found)} faces found "
      f"(job {job_id[:8]})")
    return records


# label, SELFIE_DETECTOR_MODE, EMBEDDER. "baseline" reproduces exactly
# today's production defaults (see core/settings.py) — everything else is
# opt-in via these env vars, never the default.
SWEEP_CONFIGS = [
    ("baseline (current production)", "legacy", "deepface"),
    ("scrfd selfie + DeepFace", "scrfd", "deepface"),
    ("scrfd selfie + w600k_mbf", "scrfd", "mbf"),
    ("scrfd selfie + w600k_r50", "scrfd", "r50"),
]


def _restart_worker(selfie_mode: str, embedder: str) -> None:
    """Force-recreate the worker container with new env — Settings is read
    once at process start, so there's no way to pick up a new
    SELFIE_DETECTOR_MODE/EMBEDDER without restarting the process."""
    env = {**os.environ, "SELFIE_DETECTOR_MODE": selfie_mode, "EMBEDDER": embedder}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "worker"],
        cwd=BACKEND, env=env, check=True, capture_output=True, text=True,
    )
    time.sleep(5)  # let the process finish starting before jobs land on it


def _score_config(records: list[dict]) -> dict:
    synthetic = [r for r in records if r["source"] == "synthetic"]
    real_ok = [r for r in records if r["source"] == "real" and not r["ambiguous"]]
    best = (max(THRESHOLDS, key=lambda t, rs=synthetic: score(rs, t)["f1"])
            if synthetic else DEFAULT_THRESHOLD)
    return {"threshold": best, **score(synthetic, best),
            "real": score(real_ok, best) if real_ok else None}


def build_sweep_body(all_results: dict[str, dict]) -> str:
    rows = [
        [label, f"{r['threshold']:.2f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
         f"**{r['f1']:.3f}**", r["fp"], f"{r['real']['recall']:.3f}" if r["real"] else "-",
         f"{r['elapsed_s']:.0f}s"]
        for label, r in all_results.items()
    ]
    return "\n".join([
        "Every config below runs through the real API — real jobs, real "
        "Postgres, real worker — not a reimplementation. Cached embeddings "
        "are wiped between configs so each one scores distances it "
        "actually computed, not a stale cache left by the previous config. "
        "Threshold is chosen per-config on the synthetic corpus, same rule "
        "as the desktop comparison in the embedder-comparison section.\n",
        md_table(["config", "threshold", "precision", "recall", "F1", "FP",
                  "real recall", "time"], rows),
    ])


def main_sweep() -> None:
    jobs = discover_jobs()
    try:
        requests.get(f"{API}/docs", timeout=10).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"API not reachable at {API} ({exc}) — is docker compose up?")

    seeded = seeded_job_ids()
    unseeded = [j["name"] for j in jobs if j["name"] not in seeded]
    if unseeded:
        raise SystemExit(
            f"{len(unseeded)} job(s) not in the database — run "
            "benchmarks/seed_jobs.py first"
        )

    all_results: dict[str, dict] = {}
    for label, selfie_mode, embedder in SWEEP_CONFIGS:
        p("=" * 70)
        p(f"{label}  (SELFIE_DETECTOR_MODE={selfie_mode} EMBEDDER={embedder})")
        p("=" * 70)
        _restart_worker(selfie_mode, embedder)
        wipe_cached_embeddings()

        started = time.time()
        records: list[dict] = []
        for job in jobs:
            records += run_job(job, seeded[job["name"]])
        elapsed = time.time() - started

        result = _score_config(records)
        result["elapsed_s"] = elapsed
        real_txt = f"real recall={result['real']['recall']:.3f}" if result["real"] else "no real jobs"
        p(f"  threshold {result['threshold']:.2f}: precision={result['precision']:.3f} "
          f"recall={result['recall']:.3f} F1={result['f1']:.3f} FP={result['fp']}  "
          f"{real_txt}  ({elapsed:.0f}s)\n")
        all_results[label] = result

    write_results_section("sweep", "Detector/embedder sweep (real API)",
                          build_sweep_body(all_results))


def main() -> None:
    jobs = discover_jobs()

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

    seeded = seeded_job_ids()
    unseeded = [j["name"] for j in jobs if j["name"] not in seeded]
    if unseeded:
        raise SystemExit(
            f"{len(unseeded)} job(s) not in the database: {', '.join(unseeded[:4])}"
            f"{' ...' if len(unseeded) > 4 else ''}\n"
            "Run benchmarks/seed_jobs.py first — this script measures jobs that "
            "already exist, it does not upload."
        )

    p(f"re-processing {len(jobs)} seeded jobs through {API}\n")
    started = time.time()
    records: list[dict] = []
    for job in jobs:
        records += run_job(job, seeded[job["name"]])
    elapsed = time.time() - started

    synthetic = [r for r in records if r["source"] == "synthetic"]
    real_ok = [r for r in records if r["source"] == "real" and not r["ambiguous"]]
    real_amb = [r for r in records if r["source"] == "real" and r["ambiguous"]]

    body: list[str] = [
        "Every job was processed through the running API — real jobs, photos "
        "fetched from object storage, the real matcher. Labels come from the "
        "filenames, so there is no ground-truth file to drift out of sync.\n"
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
    if "--sweep" in sys.argv:
        main_sweep()
    else:
        main()
