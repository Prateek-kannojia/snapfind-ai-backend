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

FastAPI · SQLAlchemy · PostgreSQL + pgvector (SQLite for local dev) · Redis + RQ · DeepFace (ArcFace) · insightface (SCRFD) · Docker

## API

| Endpoint | What it does |
|---|---|
| `POST /jobs/upload` | Upload selfie + event photos zip |
| `GET /jobs/{id}` | Poll job status |
| `POST /jobs/{id}/process` | Queue face matching |
| `GET /jobs/{id}/matches` | List matched photos |
| `GET /jobs/{id}/matches/{match_id}/download` | Download a matched photo |

## Run it

```powershell
docker compose up -d --build
```
Starts everything — API, worker, Redis, Postgres. API at `http://localhost:8000/docs`. Manual (no-Docker) setup: see [DEEP_DIVE.md](DEEP_DIVE.md#how-to-run-locally).

## Measured performance

Real numbers, not estimates — full comparison and methodology in [`benchmarks/`](benchmarks/).

| Detector | Time/photo | Accuracy |
|---|---|---|
| opencv (original default) | 0.26s | 0/4 faces — silently broken |
| retinaface | ~11s | 4/4 — correct, too slow |
| **insightface/SCRFD (current)** | **~0.4s** | **4/4 — correct and fast** |

## Project structure

```
Face_recognition/
├── main.py, worker.py            # FastAPI app, RQ worker
├── api/, core/, db/, services/   # routes, config/errors, ORM, business logic
├── benchmarks/                   # detector comparison, reproducible
├── Dockerfile, docker-compose.yml
└── DEEP_DIVE.md                  # full pipeline explanation + dated work log
```

Full architecture reasoning, database schema, every config variable, the dated history behind each decision (why insightface over retinaface, why Redis+RQ, why pgvector), and an interview cheat-sheet: **[DEEP_DIVE.md](DEEP_DIVE.md)**.
