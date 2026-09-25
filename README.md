# SnapFind AI — Face Recognition Backend

A FastAPI backend that finds every photo in an event archive where a specific person appears — upload one selfie and a ZIP of event photos, get back only the matches.

**Full technical write-up, dated work log, and interview notes:** [DEEP_DIVE.md](DEEP_DIVE.md)

---

## How it works (short version)

1. **Detect** a face — mtcnn for the selfie, insightface/SCRFD for event photos, by default. That's a real mismatch (two different detectors, and the selfie skips the crop-from-original step event photos use) — measured and confirmed costly; see [Measured performance](#measured-performance). `SELFIE_DETECTOR_MODE=scrfd` routes the selfie through the same SCRFD path as event photos; not the default yet, since flipping it is a migration (every cached embedding changes), not a config tweak.
2. **Embed** the face into a 512-number vector — DeepFace's ArcFace by default, or one of two insightface ONNX models (`w600k_mbf`, `w600k_r50`) via `EMBEDDER`, for both selfie and event photos identically
3. **Compare** via cosine distance — under the threshold (default `0.68`) counts as a match
4. **Cache** each embedding in the database, so re-running with a different threshold is nearly instant
5. **Parallelize** event-photo processing across threads

Jobs move through `pending → queued → processing → completed/failed`. Matching runs on a separate Redis/RQ worker process, not inside the HTTP request — the API returns immediately and the client polls for status.

## Tech stack

FastAPI · SQLAlchemy · PostgreSQL + pgvector · MinIO / S3 · Redis + RQ · DeepFace (ArcFace) · insightface (SCRFD) · Docker

## API

Photo bytes never pass through this API. Clients upload straight to object
storage using presigned URLs, and the zip goes up in parts so an interrupted
upload can resume instead of restarting.

| Endpoint | What it does |
|---|---|
| `POST /jobs/upload/init` | Create a job, get presigned URLs for the selfie and each zip part |
| `POST /jobs/{id}/upload/urls` | Fresh URLs for specific parts — to resume, or when the originals expired |
| `POST /jobs/{id}/upload/complete` | Finish the multipart upload |
| `GET /jobs/{id}` | Poll job status; also lists which parts have landed |
| `POST /jobs/{id}/process` | Queue face matching |
| `GET /jobs/{id}/matches` | List matched photos with presigned download URLs |
| `GET /jobs/{id}/matches/{match_id}/download` | Redirects (307) to a presigned URL |

## Run it

```powershell
docker compose up -d --build
```
Starts everything — API, worker, Redis, Postgres, MinIO. API at
`http://localhost:8000/docs`, object browser at `http://localhost:9001`.
Manual (no-Docker) setup: see [DEEP_DIVE.md](DEEP_DIVE.md#how-to-run-locally).

## Measured performance

Real numbers, not estimates. Generated output in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md), methodology and caveats in
[`benchmarks/`](benchmarks/).

Detector choice, 30 photos, identical input to all three:

| Detector | Time/photo | Found a face |
|---|---|---|
| opencv (original default) | 0.23s | 53% — silently broken |
| retinaface | 14.4s | 97% — correct, too slow |
| **insightface/SCRFD (current)** | **0.28s** | **100%** |

Matching accuracy against a labelled corpus (153 composited LFW photos, 10
identities, known answers) at the default 0.68 threshold: **precision 0.955,
recall 0.829, F1 0.887**. Accuracy tracks face size closely — 0.968 at 160px,
0.800 at 70px — which is the honest limit of the current pipeline.

Four configurations, run through the real API on the real database (not a
reimplementation — see [`benchmarks/README.md`](benchmarks/README.md)):

| Config | F1 | False positives | Time (13 jobs) |
|---|---|---|---|
| Current production (mtcnn selfie, DeepFace) | 0.903 | 3 | 929s |
| + SCRFD for the selfie (embedder unchanged) | 0.938 | 1 | 377s |
| + `w600k_mbf` embedder (13.6 MB, ONNX) | **0.959** | **0** | **74s** |
| + `w600k_r50` embedder (174 MB, ONNX) | 0.959 | 0 | 150s |

Two separate, additive findings: the selfie's detector mismatch was a real,
measured bug — fixing just that improved every number *and* ran 2.5x faster,
since `mtcnn` carries real TensorFlow overhead SCRFD doesn't have. And once
that's fixed, `w600k_mbf` beats DeepFace outright (better F1, zero false
positives, 12x faster) and ties the larger `w600k_r50` on accuracy while
being twice as fast and an order of magnitude smaller — there's no accuracy
being left on the table by picking the small model.

**Measured, not adopted** — switching invalidates every cached embedding and
needs the threshold re-derived, so it is migration work rather than a config
change. Both `SELFIE_DETECTOR_MODE` and `EMBEDDER` exist as settings today
specifically to make that migration a config flip when it happens, not a
code change.

## Project structure

```
Face_recognition/
├── main.py, worker.py            # FastAPI app, RQ worker
├── api/, core/, db/, services/   # routes, config/errors, ORM, business logic
├── benchmarks/                   # corpus builder + seeder, 4 reproducible measurements
├── Dockerfile, docker-compose.yml
└── DEEP_DIVE.md                  # full pipeline explanation + dated work log
```

Full architecture reasoning, database schema, every config variable, the dated history behind each decision (why insightface over retinaface, why Redis+RQ, why pgvector), and an interview cheat-sheet: **[DEEP_DIVE.md](DEEP_DIVE.md)**.
