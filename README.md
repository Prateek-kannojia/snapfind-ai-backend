# SnapFind AI — Face Recognition Backend

A FastAPI backend that finds every photo in an event archive where a specific person appears — upload one selfie and a ZIP of event photos, get back only the matches.

**Full technical write-up, dated work log, and interview notes:** [DEEP_DIVE.md](DEEP_DIVE.md)

---

## How it works (short version)

1. **Detect** a face — mtcnn for the selfie (accuracy matters most, one shot to get it right), insightface/SCRFD for event photos (speed matters, many photos per job)
2. **Embed** the face into a 512-number vector via DeepFace's ArcFace model
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

A 13.6 MB ONNX model (`w600k_mbf`, already on disk inside the detection pack)
beats the 137 MB TensorFlow ArcFace currently in production. On the eight real
phone photos it recovers **all eight at threshold 0.70 while still producing
zero false positives** on the labelled corpus; ArcFace only reaches all eight at
0.90, where precision collapses to 0.606 and it produces 43. **Measured, not
adopted** — switching invalidates every cached embedding and needs the threshold
re-derived, so it is migration work rather than a config change.

## Project structure

```
Face_recognition/
├── main.py, worker.py            # FastAPI app, RQ worker
├── api/, core/, db/, services/   # routes, config/errors, ORM, business logic
├── benchmarks/                   # corpus builder + seeder, 3 reproducible measurements
├── Dockerfile, docker-compose.yml
└── DEEP_DIVE.md                  # full pipeline explanation + dated work log
```

Full architecture reasoning, database schema, every config variable, the dated history behind each decision (why insightface over retinaface, why Redis+RQ, why pgvector), and an interview cheat-sheet: **[DEEP_DIVE.md](DEEP_DIVE.md)**.
