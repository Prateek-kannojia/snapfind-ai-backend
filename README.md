# SnapFind AI — Face Recognition Backend

A FastAPI backend that accepts a selfie and a ZIP archive of event photos, then finds every photo in the archive where that person appears. Built with Python, SQLAlchemy, and DeepFace.

---

## What problem does this solve?

At events like weddings, conferences, or college fests, a photographer takes hundreds of photos. Finding the ones you are in is tedious. This backend automates that: you upload one selfie and the full photo archive, and the system returns only the photos containing your face.

## How the face matching works

1. **Detection** — mtcnn for the selfie (accurate, slower, and the selfie is the one image that must not be missed), insightface/SCRFD for event photos (lightweight ONNX detector, many photos per job so speed matters more).
2. **Embedding** — DeepFace's ArcFace model turns a detected/aligned face into a 512-number vector.
3. **Cosine distance** — compares the selfie's embedding against each event photo's embedding; a distance below the threshold (default `0.68`) counts as a match.
4. **Caching** — each event photo's embedding is stored in the database after the first computation, so re-running with a different threshold doesn't recompute anything.
5. **Parallelism** — event photos are embedded concurrently via a `ThreadPoolExecutor`.

## Job lifecycle

```
pending -> queued -> processing -> completed
                              \-> failed
```

Uploading creates a `pending` job. `/process` pushes it onto a Redis queue and returns immediately (`queued`); a separate worker process claims it, runs face matching, and marks it `completed` or `failed`. The client polls `GET /jobs/{id}` for status.

## API endpoints

- `POST /jobs/upload` — selfie + event photos zip, returns a job id
- `GET /jobs/{job_id}` — job status
- `POST /jobs/{job_id}/process?threshold=0.68` — queue the job for matching
- `GET /jobs/{job_id}/matches` — matched photos with distances
- `GET /jobs/{job_id}/matches/{match_id}/download` — download a matched photo

## Project structure

```
Face_recognition/
├── main.py                    # FastAPI app, startup hook
├── worker.py                  # RQ worker entry point
├── requirements.txt
├── api/
│   ├── routes.py               # HTTP endpoints
│   └── schemas.py              # Pydantic response models
├── core/
│   └── settings.py             # Config via environment variables
├── db/
│   ├── database.py             # SQLAlchemy engine, session factory
│   └── orm_models.py           # UploadJob, EventPhoto, MatchedPhoto
└── services/
    ├── upload_service.py       # Upload validation, file saving, job creation
    ├── job_service.py          # Job lifecycle: queue, claim, process, status
    ├── queue_service.py        # Redis/RQ enqueue helper
    ├── face_matcher.py         # ML pipeline: detection, embedding, matching
    └── storage_service.py      # File I/O: save, extract zip, path validation
```

## How to run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

docker run --name snapfind-redis -p 6379:6379 redis:7

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
python worker.py   # second terminal
```

On the first request, DeepFace/insightface download their model weights and cache them under `storage/`.
