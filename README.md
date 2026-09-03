# SnapFind AI — Face Recognition Backend

A FastAPI backend that accepts a selfie and a ZIP archive of event photos, then finds every photo in the archive where that person appears. Built with Python, SQLAlchemy, and DeepFace.

---

## What problem does this solve?

At events like weddings, conferences, or college fests, a photographer takes hundreds of photos. Finding the ones you are in is tedious. This backend automates that: you upload one selfie and the full photo archive, and the system returns only the photos containing your face.

**Jump to:** [ML pipeline](#how-the-face-matching-actually-works-the-ml-pipeline) · [Job lifecycle](#job-lifecycle) · [Database schema](#database-schema) · [API endpoints](#api-endpoints) · [Project structure](#project-structure) · [Key design decisions](#key-design-decisions-explained) · [How to run](#how-to-run-locally) · [Config reference](#configuration-environment-variables) · [Performance numbers](#performance-reality) · [Work log — the full dated history, why each decision was made](#work-log) · [Cross-project status](#cross-project-status) · [Interview cheat-sheet](#interview-cheat-sheet)

New here and want the short version first? Read "What problem does this solve," the ML pipeline, and the Interview cheat-sheet at the bottom — that's the 5-minute version. The Work log is the detailed, dated "why we did X" history — worth reading before an interview, not needed to just understand or run the project.

---

## How the face matching actually works (the ML pipeline)

Understanding this is the most important part of the project.

### Step 1 — Face Detection

Before comparing faces, we need to locate the face inside each image. This is called **face detection**. We use two different detectors:

- **mtcnn** (Multi-task Cascaded Convolutional Networks) for the selfie. It is a neural network itself, slower but very accurate. We use it for the selfie because it is the most important image — if we miss the selfie face, the whole job fails. mtcnn handles slight angles, shadows, and imperfect lighting well.
- **insightface / SCRFD-500MF** for event photos, via `insightface`'s `FaceAnalysis` (ONNX Runtime, CPU) — a lightweight modern detector, same weight class as the "RetinaFace-MobileNet-0.25" variant benchmarked in the original RetinaFace paper. The detected face is aligned to the standard 112×112 ArcFace convention (`insightface.utils.face_align.norm_crop`, using the detector's 5-point landmarks) and only then handed to DeepFace's ArcFace model for embedding (`detector_backend="skip"`, since detection+alignment is already done) — so the embedding step is still the same ArcFace model either way, only the detection+alignment front-end changed. This went through two earlier choices first (`opencv`, then `retinaface`) — the full reasoning, measurements, and the retired detectors themselves (still runnable, just not part of production anymore) live in `../benchmarks/` at the repo root, not in this backend's own code.

This split (accurate for selfie, fast-and-still-accurate for bulk) is a deliberate tradeoff between speed and reliability — see the Work log for how "fast" and "accurate" were actually measured against real photos rather than assumed.

### Step 2 — Face Embedding (the core of face recognition)

Once a face is detected and cropped, we pass it through a deep neural network called **ArcFace**. ArcFace does not classify who the person is — instead, it converts the face image into a list of 512 numbers called an **embedding** (or feature vector).

```
face image  →  ArcFace model  →  [0.023, -0.891, 0.441, ... 512 numbers]
```

The key property: faces of the same person produce embeddings that are numerically close to each other. Faces of different people produce embeddings that are far apart — regardless of lighting, angle, or expression. This is what makes the comparison possible.

ArcFace is provided via the **DeepFace** library, which handles model downloading, caching, and inference.

### Step 3 — Cosine Distance

To compare two embeddings (selfie vs event photo), we compute the **cosine distance** between them.

Cosine distance measures the angle between two vectors in 512-dimensional space. The formula used:

```
distance = 1 - (A · B) / (|A| × |B|)
```

Where `A · B` is the dot product and `|A|`, `|B|` are the magnitudes. The result is:

- `0.0` — identical faces (same image)
- `~0.2–0.4` — same person, different photo
- `~0.6–1.0` — different people

We compute this ourselves in `_cosine_distance()` in `face_matcher.py` rather than relying on DeepFace's built-in comparison, giving us full control over the threshold logic.

### Step 4 — Threshold

The **threshold** is the cutoff distance. Any event photo with a distance ≤ threshold is considered a match.

- Default: `0.68` (balanced between missing matches and false positives)
- Lower = stricter (fewer matches, higher confidence)
- Upper bound: `1.0` (validated in the API to prevent nonsense values)

The client can pass a custom threshold via the `?threshold=` query parameter.

### Step 5 — Embedding Cache

Computing an ArcFace embedding takes 1–4 seconds per photo on CPU. If the same job is processed twice (e.g., with a different threshold), we should not recompute embeddings for photos that have not changed.

Solution: after computing an embedding for an event photo, we store it as a JSON string in the `embedding` column of the `event_photos` database table. On the next processing run, we load the cached embedding directly from the database instead of running the model again.

This makes repeat runs nearly instant regardless of how many photos the archive contains.

### Step 6 — Parallel Processing

Event photos are processed in parallel using a `ThreadPoolExecutor` with up to 4 worker threads. DeepFace releases Python's GIL during TensorFlow inference, so threads genuinely run in parallel for the detection and embedding steps.

To avoid SQLAlchemy session threading issues, each thread works only with plain Python data (a `_Photo` namedtuple with the photo's id, path, and cached embedding). All database writes (saving new embeddings, creating match records) happen in the main thread after all futures complete.

---

## Job lifecycle

Every request is tracked as a **job** that moves through five states:

```
pending  ->  queued  ->  processing  ->  completed
                                  \->  failed
```

- `pending` - job created, files uploaded, waiting to be submitted for processing
- `queued` - job was pushed to Redis/RQ and is waiting for a worker
- `processing` - an RQ worker claimed the job and face matching is running
- `completed` - matching done, results available
- `failed` - something went wrong (selfie face not detected, corrupted image, etc.)

The `/process` endpoint returns immediately with `status: queued`. It does not run face matching inside the FastAPI process anymore. Instead, it pushes a small message to Redis through RQ. A separate worker process reads that message, marks the job as `processing`, computes matches, and finally marks the job as `completed` or `failed`.

The status transitions are guarded by atomic SQL updates. The API only queues jobs that are not already `queued` or `processing`. The worker only claims jobs currently in `queued`. This prevents duplicate processing if the same endpoint is called twice.

### Stale job recovery

A `409 Conflict` is still correct while a job is genuinely `queued` or `processing`. However, a production system also needs a way to recover if a process dies at the wrong time.

This backend treats old queued/processing jobs as stale after `JOB_STALE_AFTER_SECONDS` seconds. When `/process` is called again:

- active `queued` or `processing` jobs still return `409`
- stale `queued` jobs can be queued again
- stale `processing` jobs can be moved back to `queued` and picked up by a worker again

This is why the job table stores `queued_at`, `processing_started_at`, `rq_job_id`, and `last_error`.

---

## Database schema

Three tables, all in SQLite via SQLAlchemy:

```
upload_jobs
  id               TEXT PRIMARY KEY  (UUID)
  status           ENUM              (pending / queued / processing / completed / failed)
  selfie_filename  TEXT
  selfie_storage_path TEXT
  event_photo_count INTEGER
  created_at       DATETIME
  queued_at        DATETIME NULL
  processing_started_at DATETIME NULL
  rq_job_id        TEXT NULL
  last_error       TEXT NULL

event_photos
  id               INTEGER PRIMARY KEY
  job_id           TEXT → upload_jobs.id
  original_filename TEXT
  storage_path     TEXT
  embedding        TEXT   ← JSON string of 512 floats, NULL until first processing run
  created_at       DATETIME

matched_photos
  id               INTEGER PRIMARY KEY
  job_id           TEXT → upload_jobs.id
  event_photo_id   INTEGER → event_photos.id
  match_distance   FLOAT
  created_at       DATETIME
```

The `matched_photos` table is fully replaced on every processing run. This means you can re-run processing with a different threshold and always get a fresh, correct result set.

---

## File storage layout

Each job gets its own directory under `storage/uploads/`:

```
storage/uploads/{job_id}/
  selfie/
    {uuid}.jpg           ← selfie saved with a UUID filename to avoid collisions
  archive/
    {uuid}.zip           ← original ZIP archive
  event_photos/
    photo1.jpg           ← extracted from ZIP
    subdir/photo2.jpg    ← preserves ZIP directory structure
```

ZIP extraction is protected against **path traversal attacks**: we verify every extracted path is actually inside the destination directory using `Path.is_relative_to()` before writing any file. Malicious ZIPs containing paths like `../../etc/passwd` are rejected.

If any error occurs during upload (ZIP extraction fails, no valid images found, etc.), both the database transaction and the entire job directory on disk are rolled back and deleted together, leaving no orphaned files.

---

## API endpoints

### `GET /`
Health check. Returns `{ "message": "Event photo finder API is working" }`.

### `POST /jobs/upload`
Upload the selfie and event photo archive.

**Form fields (multipart):**
- `selfie` — image file (`.jpg`, `.jpeg`, `.png`, `.webp`)
- `event_photos_zip` — ZIP archive containing event photos

**What happens internally:**
1. Validates file types
2. Creates a job record in the database (flush to get the job ID)
3. Saves selfie and ZIP to disk under `storage/uploads/{job_id}/`
4. Extracts the ZIP and records each valid image as an `EventPhoto` row
5. Commits everything and returns the job summary

**Returns:** `UploadJobResponse` with `job_id`, `status: "pending"`, and photo count.

### `GET /jobs/{job_id}`
Get the current status and summary of a job. Used for polling.

**Returns:** `JobSummaryResponse` with `status`, `matched_photo_count`, and other job metadata.

### `POST /jobs/{job_id}/process?threshold=0.68`
Trigger face matching for a job.

**What happens internally:**
1. Atomically sets `status = queued` (rejects with 409 if already queued or processing)
2. Pushes a small RQ message into Redis containing the `job_id` and `threshold`
3. Returns immediately with `status: "queued"`
4. A separate worker process claims the queued job, sets `status = processing`, computes/loads embeddings, finds matches, saves results, and sets `status = completed`
5. If anything fails during processing: marks job as `failed`

**Query param:** `threshold` (float, `0 < threshold < 1.0`, default `0.68`)

### `GET /jobs/{job_id}/matches`
Fetch the list of matched photos with distances and download URLs.

**Returns:** `MatchListResponse` with a list of matches sorted by `match_distance` ascending (closest match first).

### `GET /jobs/{job_id}/matches/{match_id}/download`
Download a specific matched photo file.

**Returns:** The raw image file as `application/octet-stream`.

---

## Project structure

```
Face_recognition/
├── main.py                    # FastAPI app, lifespan startup hook
├── worker.py                  # RQ worker entry point; WORKER_COUNT>1 launches multiple processes
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Builds one image, shared by the api and worker services
├── docker-compose.yml         # Orchestrates api, worker, redis, postgres (pgvector-enabled) together
├── .dockerignore
├── api/
│   ├── routes.py              # All HTTP endpoints
│   └── schemas.py             # Pydantic response models (what the API returns)
├── core/
│   └── settings.py            # Config via environment variables with sane defaults
├── db/
│   ├── database.py            # SQLAlchemy engine, session factory, get_db dependency
│   └── orm_models.py          # UploadJob, EventPhoto, MatchedPhoto ORM classes
├── services/
│   ├── upload_service.py      # Upload validation, file saving, job creation
│   ├── job_service.py         # Job lifecycle: queue, claim, process, status, matches
│   ├── queue_service.py       # Redis/RQ enqueue helper
│   ├── face_matcher.py        # ML pipeline: detection, embedding, distance, parallel matching
│   └── storage_service.py     # Low-level file I/O: save, extract ZIP, path validation
└── storage/
    ├── uploads/               # Per-job uploaded files (created at runtime)
    ├── deepface/              # DeepFace model weights cache (mtcnn selfie detector, ArcFace embedder)
    └── insightface/            # insightface model weights cache (event-photo detector)
```

**Why this structure?** Each layer has one responsibility. `routes.py` only handles HTTP. `job_service.py` only knows about jobs and their lifecycle. `face_matcher.py` only knows about faces. `storage_service.py` only knows about files. This makes each piece testable and replaceable independently.

---

## Key design decisions explained

### Why FastAPI?
FastAPI generates interactive API documentation automatically at `/docs`. It validates request and response shapes using Pydantic. It supports async endpoints natively. For a Python ML backend, it is the standard choice.

### Why SQLite?
Zero infrastructure. No separate database server to run. The file `app.db` is created automatically on startup. For quick local iteration outside Docker, SQLite is completely sufficient. Switching to PostgreSQL requires only changing the `DATABASE_URL` environment variable — this was true by design from the start and has since been verified for real: `docker-compose.yml` runs Postgres by default (see the Work log below), and the exact same SQLAlchemy models generate a correct schema — including a real pgvector `Vector(512)` column — on both backends without any other code change.

### Why store embeddings in the database?
The first time 100 event photos are processed, 100 ArcFace inferences run. That takes time on CPU. If the user re-runs with a different threshold, none of the embeddings have changed — only the cutoff distance has. Without caching, you would run 100 inferences again for no reason. With the `embedding` column in `event_photos`, the second run reads from the database and skips inference entirely.

### Why Redis + RQ instead of FastAPI BackgroundTasks?
Face matching for 100 photos can take 1-5 minutes on a CPU. If the HTTP connection stays open that long, mobile clients will time out (Android's default OkHttp timeout is 30 seconds). The API therefore returns immediately and the client polls `/jobs/{id}` until `status` becomes `completed`.

The important production detail is where the long-running work runs. FastAPI `BackgroundTasks` still run inside the API server process. If that process restarts or crashes after marking a job as `processing`, the job can stay stuck forever.

Redis + RQ moves that work to a separate worker process:

```
FastAPI API  ->  Redis queue  ->  RQ worker  ->  database results
```

This gives us clearer process separation:

- the API accepts uploads, queues jobs, and serves status/results
- Redis stores the pending work message
- the worker performs CPU-heavy face matching and updates the database

This is still free and does not require a cloud account when Redis is self-hosted locally or in your own deployment.

### Why parallel threads for event photos?
Each event photo's embedding computation is independent of the others. Running them sequentially is wasteful on a multi-core CPU. `ThreadPoolExecutor` lets up to 4 photos be processed simultaneously. The selfie embedding is computed first (single-threaded) to warm the DeepFace model before the parallel section starts.

### Why downscale, why multiple workers, and a dependency bug found along the way

Three changes made together during the initial perf pass — full numbers, code specifics, and status for each are in the Work log's "Backend performance pass" entry rather than repeated here: (1) resize every image before inference (**10.48x** measured speedup), (2) `WORKER_COUNT` for horizontal worker scaling, (3) a broken `opencv-python` install found while benchmarking that had silently disabled event-photo detection entirely.

---

## How to run locally

Two ways to run this: Docker Compose (one command, everything containerized, matches how it'd actually be deployed) or a manual Python venv + a standalone Redis container (closer to how the project started, useful for fast iteration since code changes don't need a rebuild).

### Option A: Docker Compose (recommended)

```powershell
docker compose up -d --build
```

This builds one image (shared by `api` and `worker`, see `Dockerfile`) and starts all four services — `api`, `worker`, `redis`, and `postgres` (the `pgvector/pgvector:pg16` image, extension enabled automatically on startup — see `ensure_pgvector_extension()` in `db/database.py`). No local Python install, no manual Redis/Postgres setup. API available at `http://localhost:8000`.

Model weights (~350MB total, first run only) and the Postgres database persist in named Docker volumes (`deepface_weights`, `insightface_weights`, `postgres_data`) across restarts. `docker compose down -v` removes those volumes too — needed if a schema change requires recreating the database (see "Known limitation" below), not needed for a normal restart.

```powershell
docker compose logs -f api      # tail one service's logs
docker compose logs -f worker
docker compose down             # stop everything, keep volumes
docker compose down -v          # stop everything, also wipe volumes
```

### Option B: Manual (Python venv + standalone Redis)

#### Prerequisites
- Python 3.10+
- Virtual environment (recommended)
- Redis server

Redis is the message broker. You can run it locally with Docker without creating any cloud account:

```powershell
docker run --name snapfind-redis -p 6379:6379 redis:7
```

If you already have a Redis container with that name, start it again with:

```powershell
docker start snapfind-redis
```

### Setup

```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start the server

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `--host 0.0.0.0` makes the server accessible from your phone on the same Wi-Fi network
- `--reload` restarts the server automatically when you change code
- API docs available at `http://127.0.0.1:8000/docs`

### Start the worker

Open a second terminal, activate the same virtual environment, then run:

```powershell
python worker.py
```

To run more than one worker process at once (horizontal scaling — see "Why multiple worker processes?" below):

```powershell
$env:WORKER_COUNT=4; python worker.py
```

The API and worker must both be running:

- API process: handles HTTP requests and enqueues jobs
- Redis process: stores queued work messages
- Worker process: consumes queued jobs and runs face matching

On Windows, `worker.py` defaults to RQ's `SimpleWorker`, which avoids Unix-style process forking. On Linux or Docker, set `RQ_WORKER_CLASS=default` to use RQ's normal worker.

### First run note

On the very first processing request, DeepFace will download the ArcFace model weights (~100MB) and cache them in `storage/deepface/`. Subsequent runs load from cache and are much faster.

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Event Photo Finder API` | Title shown in API docs |
| `APP_VERSION` | `0.1.0` | Version shown in API docs |
| `DATABASE_URL` | `sqlite:///./app.db` | SQLAlchemy database URL. `docker-compose.yml` overrides this to `postgresql+psycopg2://snapfind:snapfind@postgres:5432/snapfind` — on Postgres, `EventPhoto.embedding` becomes a real pgvector `Vector(512)` column instead of JSON text (see Work log) |
| `UPLOAD_ROOT` | `storage/uploads` | Directory for uploaded files |
| `DEEPFACE_HOME` | `storage/deepface` | DeepFace model cache directory |
| `INSIGHTFACE_HOME` | `storage/insightface` | insightface model cache directory (the event-photo detector's ONNX weights) |
| `MAX_EVENT_PHOTOS` | `500` | Maximum photos allowed per ZIP |
| `SELFIE_DETECTOR` | `mtcnn` | Face detector for selfie (accurate) |
| `FACE_MATCH_WORKERS` | `4` | Parallel threads for event photo processing (within one worker process) |
| `SELFIE_MAX_DIMENSION` | `1024` | Selfie is downscaled to at most this many pixels on its longest side before inference |
| `EVENT_PHOTO_MAX_DIMENSION` | `800` | Same, for event photos — see "Why downscale images before inference?" |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string used by both API and worker |
| `RQ_QUEUE_NAME` | `face-matching` | Queue name used for face matching jobs |
| `RQ_JOB_TIMEOUT_SECONDS` | `1800` | Maximum runtime for one queued face matching job |
| `RQ_WORKER_CLASS` | `simple` on Windows, `default` elsewhere | RQ worker implementation. Use `simple` for local Windows testing and `default` for Linux/Docker production |
| `JOB_STALE_AFTER_SECONDS` | `3600` | How long a queued/processing job can sit before `/process` is allowed to requeue it |

Note: there is no `EVENT_PHOTO_DETECTOR` variable — the event-photo detector is fixed to `insightface`/SCRFD in code, not configurable. It went through two earlier choices first (`opencv`, `retinaface`); that history, the comparison data, and the retired detector code all live in `../benchmarks/` at the repo root, not here.

---

## Performance reality

| Scenario | Time estimate |
|---|---|
| First run, 20 photos, CPU laptop | 30–60 seconds |
| First run, 100 photos, CPU laptop | 3–8 minutes |
| Repeat run (embeddings cached) | 1–3 seconds |
| GPU server (e.g. NVIDIA T4) | ~5 seconds for 100 photos |
| Cloud API (AWS Rekognition) | ~500ms for 100 photos |

These original estimates predate this session's changes and were measured against the `opencv` detector — which, per the Work log below, turned out not to reliably detect faces in real photos at all, so treat the table above as historical/aspirational rather than current.

**Current measured reality** (this dev machine, 4 real sample photos, event-photo path only):

| Stage | Per-photo cost |
|---|---|
| Original (`opencv`, full resolution) | 2.73s — and detected 0/4 real faces |
| After downscaling only (`opencv`, resized) | 0.26s — still 0/4 faces, fast but useless |
| `retinaface` (accurate, resized) | ~11s — 4/4 faces, correct but slow |
| `insightface`/SCRFD (current default) | **~0.4s** — 4/4 faces, verified accurate (see Work log) |

All measured on a sandboxed dev machine whose TensorFlow build isn't using AVX2/FMA (visible in its own startup logs), so absolute numbers will differ on a normal machine — re-run the detector comparison in `benchmarks/detector_comparison.py` (repo root) on your own hardware for numbers you can cite directly. The *relative* pattern (opencv fast-but-broken, retinaface slow-but-correct, insightface fast-and-correct) should hold regardless of hardware, since it was measured identically across all three.

For a portfolio demo, test with 15–20 photos. The second run will always be fast thanks to embedding caching.

---

## Work log

Chronological record of changes made after the initial build, with the reasoning behind each — kept up to date so the "why" survives past the commit history, for interview prep.

### 2026-09-01 — Backend performance pass

**Problem raised:** upload + face matching felt slow end-to-end; needed concrete options before picking one.

**1. Downscale images before inference.** Phone photos are commonly 3000-4000px on the long side (test photos: 4096×1842, 3280×2460), but detection/embedding networks don't need that resolution — cost scales roughly with pixel count. `_load_resized_image()` in `services/face_matcher.py` shrinks each image to a configurable ceiling before it reaches the model — `SELFIE_MAX_DIMENSION` (1024, one image/job, accuracy matters most) and `EVENT_PHOTO_MAX_DIMENSION` (800, many images/job, speed matters most). **Measured** (4 real photos, opencv detector): 2.73s/photo full-res → 0.26s/photo resized = **10.48x**. Extrapolated: a 100-photo album's detection time drops from ~4.5min to ~26s. Measured on a dev machine whose TF build isn't using AVX2/FMA, so treat absolute seconds as machine-specific and the ratio as reliable — reproduce with `scripts/benchmark_downscale.py`. **Status: done.** No accuracy tradeoff expected (well above the ~150px minimum for reliable embeddings), no new dependency, no architectural change.

**2. Bug found while benchmarking: broken opencv-python install.** `DeepFace.represent(..., detector_backend="opencv")` failed with `cv2 has no attribute 'CascadeClassifier'` — the unpinned `opencv-python` had resolved to `5.0.0.93`, whose Windows wheel ships without the Haar cascade data or `objdetect` bindings at all. **Event-photo face detection was completely non-functional** before this fix, independent of the downscaling work. Fixed by pinning `opencv-python==4.10.0.84`. **Status: done.** Worth remembering: unpinned ML/CV dependencies are a real production risk — a transitive bump silently broke a core path with zero code change on our side, found only because we benchmarked instead of assuming it worked.

**3. Multiple worker processes (horizontal scaling).** One worker already parallelizes across `FACE_MATCH_WORKERS` threads (default 4), but Python's GIL + TF's own thread contention mean one process doesn't scale linearly with cores. `worker.py` reads `WORKER_COUNT`: at `1` (default) unchanged behavior; at `>1`, launches that many copies of itself as separate OS processes against the same RQ queue — safe since RQ's atomic dequeue already prevents double-claiming.
```powershell
python worker.py                        # 1 worker (default)
$env:WORKER_COUNT=4; python worker.py    # 4 worker processes
```
(Originally a separate `run_workers.py` launcher; folded into `worker.py` the same day — see "Project cleanup" below — since a launcher is just this script re-invoking itself.) **Status:** implemented, not yet load-tested with concurrent jobs.

**Not done from the original options list:** batch inference, GPU inference, ONNX/quantized models, a lighter embedding model — parked pending a decision on whether more speed is still needed.

### 2026-09-01 — Project cleanup

**Problem raised:** doc sprawl (8 markdown files across the repo with heavy overlap) and two separate worker scripts (`worker.py` + `run_workers.py`) where one was really just a launcher for the other.

**What we did:**
- Merged `run_workers.py` into `worker.py` itself (`WORKER_COUNT` env var: `1` = single worker, unchanged default behavior; `>1` = launches that many copies of itself as subprocesses). One file instead of two, same capability.
- Merged `Face_recognition/SUMMARY.md` into this README (its content was substantially redundant — an "interview-focused" restatement of what README already covered — kept only what was unique: the elevator-pitch cheat-sheet, now under "Interview cheat-sheet" below).
- Deleted the root `implementation_plan.md` (its original purpose — planning the Android doc set — was long complete); its live "open problems" tracker moved into "Cross-project status" below.
- On the Android side: merged `SnapFindAI/ARCHITECTURE.md`, `CODE_BREAKDOWN.md`, and `PROJECT_CONTEXT.md` into `SnapFindAI/README.md`, fixing several places where those docs had drifted from the actual code (they described a single `JobRepository.uploadAndProcessJob()` method that doesn't exist; the real code splits that across `JobRepository` + `FindFacesInPhotosUseCase`, which none of the old docs mentioned at all). Kept `ROADMAP.md` separate since forward-looking plans are a genuinely different kind of doc from current-state documentation.
- **Result:** 8 markdown files → 3 (`Face_recognition/README.md`, `SnapFindAI/README.md`, `SnapFindAI/ROADMAP.md`).

### 2026-09-01 — Event-photo detector history: opencv → retinaface → insightface

**Problem raised:** before trusting the perf-pass numbers, verified end-to-end whether the backend can actually detect faces and return matches, using the real sample photos already in `storage/uploads/`. This turned into three rounds, each measured against the same 4 real photos before moving on.

**Round 1 — found opencv was silently broken.** The selfie path (mtcnn) worked correctly — face detected, 512-dim embedding produced. The event-photo path (`opencv` Haar cascade, the original default) **detected 0 faces in all 4 real sample photos**, including one photo with two people clearly facing the camera. First ruled out image orientation as the cause (verified `cv2.imread` correctly applies EXIF rotation — it does, in this OpenCV version). The actual cause: Haar cascades are a 2001-era technique that fails easily on sunglasses, head angle, or a face that's small relative to the frame — exactly what's in these real phone photos. This was a **pre-existing accuracy bug**, not something introduced by the downscaling work — a photo the detector can't find a face in is simply skipped, by design (see the ML pipeline section above), so there was no error to notice.

**Round 2 — retinaface fixed accuracy, but was slow.** Switched to `retinaface` (a modern CNN-based detector) and tested: **4/4 detected**, including correctly finding both faces in the two-person photo. But measured **10.98s/photo** average on this sandboxed dev machine (vs opencv's 0.26s/photo) — a real, honest cost, not a free fix.

**Round 3 — asked "why is MobileNet-class so much faster, can we use that?", tested rather than assumed.** The published RetinaFace paper (Table 5, arXiv:1905.00641) shows a MobileNet-0.25-backbone variant running 1.4-25.6ms/image on a Tesla P40, vs 75-1742ms for the ResNet-152 variant — but DeepFace's own `retinaface` backend is hardcoded to ResNet50, so that speed isn't available through DeepFace at all, and a lighter backbone is a real accuracy risk (that's exactly why opencv, also "the fast one," failed in Round 1). Rather than swap blind:
- Installed `insightface` (pip-installable, ONNX Runtime, no heavy new ML framework) and tested its lightest detection pack (`buffalo_sc`, SCRFD-500MF — same weight class as RetinaFace-MobileNet-0.25) against the same 4 photos: **4/4 detected**, matching retinaface's face counts exactly (2, 1, 1, 1).
- Then specifically checked the accuracy risk, not just detection success: compared the actual ArcFace embeddings produced by the retinaface path vs. the insightface-detect + `norm_crop`-align + DeepFace-`skip`-embed path, for the *same* photos. Agreement (cosine distance between the two embeddings of the same face): **0.0142, 0.0055, 0.0053, 0.2374** — three near-identical, one more divergent but still well inside "same person, different photo" range per this project's own documented distance bands, and far below the 0.68 match threshold either way.
- Measured speed through the real, integrated `_embedding_for_image()` call (not a standalone script): **~386-472ms/photo** steady-state, after a one-time ArcFace warm-up cost that production already absorbs via the existing selfie-first warm-up pattern. That's roughly **25-30x faster than retinaface**, verified in-pipeline, not just in isolation.

**Fix:** integrated `insightface`'s SCRFD detector into `services/face_matcher.py` (`_event_photo_embedding()` — renamed 2026-09-02, see "Cleaned up function naming" below; was `_embedding_for_event_photo_fast()` at the time): detect + align via insightface, then embed via DeepFace's ArcFace with `detector_backend="skip"` since the crop is already aligned — so the embedding model itself never changed, only the detection/alignment front-end.

**Status:** done, integrated, verified against real photos through the actual code path. **Not yet tested** against a large/varied dataset (many more sunglasses/angle/low-light cases) to know the real-world false-negative rate at scale, and the one 0.2374 outlier is worth understanding better with more data rather than assuming it's noise.

### 2026-09-01 — Separated production code from the detector comparison history

**Problem raised:** `_embedding_for_image()` still had a config-driven fallback branch (`EVENT_PHOTO_DETECTOR=retinaface`/`opencv` would route back through plain DeepFace) left over from the round above. Useful as a rollback path, but it meant the production pipeline file mixed "what we actually use" with "what we used to use," and DeepFace's role in the project (still runs the selfie detector and every embedding, just not event-photo detection anymore) was easy to misread from the branching code.

**What we did:** removed the fallback branch and the `EVENT_PHOTO_DETECTOR` setting entirely — `services/face_matcher.py` now has exactly one path per role (selfie → mtcnn, event photo → insightface), no config branching. The retired `opencv`/`retinaface` detector code didn't get deleted, though — it moved to a new top-level `../benchmarks/` folder as a standalone, runnable comparison script (`detector_comparison.py`) with its own README, real measured numbers, and a pipeline diagram. That's now the canonical place to see the full opencv → retinaface → insightface evolution with reproducible data, separate from the code that actually ships.

**Status:** done. Verified the refactored production path still works end-to-end against real photos after removing the branch.

### 2026-09-02 — Docker + docker-compose

**Problem raised:** running this project meant three separately-managed things — a Python venv, a manually-installed Redis, and (about to become a fourth) a manually-installed Postgres — each able to drift out of sync with what the code actually expects, and none of it matching how the resume describes the project ("Docker").

**What we did:** one `Dockerfile` builds a single image containing the app and its dependencies; `docker-compose.yml` runs that image twice as separate services — `api` (default command: `uvicorn`) and `worker` (`command: python worker.py`) — alongside `redis:7` and `pgvector/pgvector:pg16` (chosen over plain `postgres:16` specifically so the pgvector extension would already be available for the next step, avoiding a second infra change). `depends_on: condition: service_healthy` on Postgres stops `api`/`worker` from starting before Postgres can accept connections. Named volumes (`deepface_weights`, `insightface_weights`, `postgres_data`, `uploads_data`) keep model weights, the database, and uploaded photos alive across container restarts, since containers themselves are disposable by design.

**Status:** done, verified — see the "full pipeline verified end-to-end" work below. The general practice this uses (why multi-container orchestration is a real thing companies do, not just us) is written up in `../production_learnings.md`.

### 2026-09-02 — PostgreSQL

**Problem raised:** SQLite is fine for solo local dev, but the resume specifically claims PostgreSQL, and now that `WORKER_COUNT` can run multiple worker processes writing concurrently, Postgres is also a genuine correctness improvement over SQLite's single-writer behavior.

**What we did:** almost nothing, and that's the point — `core/settings.py` already read `DATABASE_URL` from an environment variable with a SQLite default; `db/database.py` already branched its one SQLite-specific behavior (`connect_args`, and `ensure_runtime_schema()`'s SQLite-only `ALTER TABLE` logic) on `database_url.startswith("sqlite")`. The database layer was already written to be swappable. The only real gap: SQLAlchemy needs a driver per database, and SQLite's (`sqlite3`) ships in Python's standard library while Postgres's doesn't. Added `psycopg2-binary` (the precompiled variant, so no C compiler / `libpq` needed on the machine) to `requirements.txt`. `docker-compose.yml` sets `DATABASE_URL: postgresql+psycopg2://snapfind:snapfind@postgres:5432/snapfind` for `api`/`worker`; nothing else changed.

**Verified, not just assumed:** ran the real stack (`docker compose up`), then directly inspected the live database — `\dt` in `psql` showed all three tables created correctly, `\d upload_jobs` showed the native Postgres `jobstatus` enum type and correct column types/lengths/foreign keys, all generated by the exact same SQLAlchemy models that also generate SQLite's schema.

**Status:** done, verified against a live Postgres container, not just "it built."

### 2026-09-02 — Full pipeline verified end-to-end against the real Docker stack (not just "it starts")

**Problem raised:** containers starting and an API responding to a health check proves much less than it looks like — it doesn't prove uploads work, matching works, or the database round-trip is correct.

**What we did:** uploaded a real selfie + real event-photo zip through the actual running API (`curl` multipart POST, not a filesystem shortcut — the container's `storage/uploads` is a Docker volume, separate from the host), triggered processing, and let the real worker process it, including a genuine first-run download of ArcFace/mtcnn/insightface weights into the fresh volumes. Ran both a negative control (a selfie of a different person than the event photos — correctly 0 matches) and a positive control (the correct selfie — correctly matched all 4 real event photos, including the one photo that the original `opencv` detector had failed on completely, back at the very start of this project's perf work). Verified the match `download` endpoint serves the exact original file (byte-identical). Also directly inspected `event_photos.embedding` in Postgres via `psql` to confirm real 512-number embeddings were actually stored, not silently skipped.

**Status:** done. This is also where the technique of calling the app's real production functions directly inside the running container (instead of modifying code and redeploying to answer a diagnostic question) came from — written up in `../production_learnings.md`.

### 2026-09-02 — Real pgvector column instead of JSON text

**Problem raised:** `EventPhoto.embedding` was a `Text` column holding a JSON-serialized list of 512 floats, with cosine distance computed by hand in `_cosine_distance()`. That works, but it means "vector similarity search" wasn't literally true — there was no vector type or vector-aware storage anywhere, just a string.

**What we did:** changed the column to a real pgvector `Vector(512)` type on Postgres, via `Text().with_variant(Vector(512), "postgresql")` in `db/orm_models.py` — SQLAlchemy's mechanism for "use this type on this backend, that type elsewhere." SQLite (still the zero-setup local dev default) keeps the JSON-text fallback, since the `vector` extension and type don't exist there at all. `services/face_matcher.py` picks the right (de)serialization at runtime via `_USES_PGVECTOR` (derived from `DATABASE_URL`) — a plain list on Postgres, `json.dumps`/`json.loads` on SQLite. Detection and matching logic itself did not change at all; this is purely a storage-layer change, deliberately scoped that way (see the pgvector scope discussion below). `db/database.py` gained `ensure_pgvector_extension()` (`CREATE EXTENSION IF NOT EXISTS vector`), called in `main.py`'s startup before `Base.metadata.create_all()` — Postgres can't create a `vector` column until the extension exists.

Since there's no migration tool yet (no Alembic — see "Known limitation" below), changing an existing column's type meant recreating the dev database (`docker compose down -v` to drop the Postgres volume, rebuild, `up` again) rather than migrating in place. Fine for a dev database with only test data; would need a real migration for a database with real data to preserve.

**Verified, not just built:** ran a full job through the real HTTP API against the rebuilt stack — same known-good selfie/event-photo pair as the earlier full-pipeline verification above — and got **byte-for-byte identical match distances** to the pre-migration run (0.0066, 0.0083, 0.0278, 0.5937) — proof the migration changed nothing observable about matching. Directly confirmed in Postgres: `pg_typeof(embedding)` returns `vector`, `vector_dims(embedding)` returns `512`, and the actual stored numbers match what the same photo showed as JSON text earlier in this session.

**Scope decision — why this stopped at storage, not a SQL-side `ORDER BY` search:** the fuller version of pgvector adoption also moves the actual nearest-neighbor query into SQL (`ORDER BY embedding <-> :selfie_vector`) instead of comparing in a Python loop. Decided against that for now: at this project's actual scale (a few hundred photos per job), the Python loop is not a bottleneck — it's sub-second, dwarfed by the ~300ms/photo detection cost. A SQL-side ANN query mainly pays off at much larger scale (thousands of vectors, searching across jobs at once), which isn't this project's access pattern. Doing it anyway would be complexity added for a demonstration, not a real need — worth naming as a real future option, not doing reflexively.

**Status:** done, verified end-to-end against a live Postgres container with real data.

**Known limitation:** there's no real migration tool (Alembic or similar) — `main.py`'s startup just calls `Base.metadata.create_all()`, which creates missing tables but never alters an existing column's type. That's fine for a fresh database (including every test in this project so far) but means a schema change against a database with real data to preserve would need a hand-written migration, not just a code change. Worth adding Alembic before this project ever holds real user data.

### 2026-09-02 — Cleaned up function naming in `services/face_matcher.py`

**Problem raised:** `_embedding_for_event_photo_fast()` named itself relative to a comparison (faster than what, exactly?) that no longer existed in this file — the slower DeepFace/retinaface path it was "fast" *compared to* had already been deleted in the "Separated production code from the detector comparison history" cleanup above. The name had gone stale the moment that branch was removed, describing a contrast that no longer existed in the code a reader could see. Separately, three functions all starting with `_embedding_for_...` (`_embedding_for_image`, `_embedding_for_selfie`, `_embedding_for_event_photo_fast`) looked like peers/siblings, when really one is the public entry point and the other two are its private implementation details — the names didn't communicate that hierarchy.

**What we did:**
- `_embedding_for_event_photo_fast()` → `_event_photo_embedding()` — dropped the stale, relative "_fast" suffix; named for what it does, not a historical speed comparison.
- `_embedding_for_selfie()` → `_selfie_embedding()` — same word-order flip, for symmetry with the rename above.
- `_embedding_for_image()` — kept as the dispatcher name; flipping the two implementation helpers' word order now makes it visually distinct from them at a glance, instead of reading as a third sibling.
- `_process_single_photo()` → `_embed_and_score_event_photo()` — the old name only said "process," not what it actually does: gets an embedding (cached or fresh) *and* computes the distance score against the selfie. Renamed to say both.

**Status:** done. Verified the file still imports and wires together correctly after the rename.

### 2026-09-03 — Centralized error handling (`core/errors.py` + one FastAPI handler)

**Problem raised:** Every exception class in the codebase was independently reinvented — `JobServiceError`/`JobNotFoundError` in `job_service.py` carried a real `status_code`, but `UploadValidationError` (`upload_service.py`), `StorageError` (`storage_service.py`), and `FaceMatchError`/`SelfieFaceNotDetectedError` (`face_matcher.py`) were bare `class X(Exception): pass` with no structure at all. `api/routes.py` then had to hand-translate each one differently: a `_raise_job_http_error()` helper for `JobServiceError`, a separate inline `except UploadValidationError: raise HTTPException(400, ...)` for uploads — two different conventions for the same job, repeated in every route function.

**What we did:**
- New `core/errors.py`: one base `AppError(message, *, status_code=400, error_code="error")`. `error_code` is a short stable string (`"job_not_found"`, `"upload_validation_error"`) so the Android client can branch on a fixed identifier instead of parsing the English `detail` message.
- Every existing error class now inherits from it: `JobServiceError` (and new named subclasses replacing what used to be inline `JobServiceError("...", status_code=409)` calls at each raise site — `JobNotFoundError`, `MatchNotFoundError`, `MatchFileMissingError`, `JobAlreadyQueuedError`, `JobAlreadyProcessingError`, `NoEventPhotosError`, `QueueUnavailableError`), `UploadValidationError`, `FaceMatchError`/`SelfieFaceNotDetectedError`.
- `StorageError` deliberately stays a plain `Exception`, not an `AppError` — it's a filesystem utility with no concept of an HTTP status, always caught and re-raised as `UploadValidationError` by its one caller; documented in the file why it's the deliberate exception to the pattern.
- One handler in `main.py`: `@app.exception_handler(AppError)` — converts any `AppError` anywhere in the call stack to `{"detail", "error_code"}` JSON with the right status code.
- `api/routes.py` simplified: no `try/except` left in any route, `_raise_job_http_error()` deleted entirely — routes just call the service function and return its result.

**Verified for real:** ran the actual dev server (`uvicorn main:app`) and hit it with real HTTP requests — `GET /jobs/does-not-exist-123` → `404 {"detail":"Job not found","error_code":"job_not_found"}`, and a real multipart upload with a `.txt` selfie → `400 {"detail":"Selfie must be one of: ...","error_code":"upload_validation_error"}` — both through the full route → service → centralized handler path, not just import-checked.

**Status:** done.

### 2026-09-03 — Removed `scripts/benchmark_downscale.py` (broken, stale)

**Problem found during a stale-code audit:** the script called `settings.event_photo_detector`, an attribute removed from `core/settings.py` when the event-photo detector was centralized to insightface (see "Event-photo detector history" above) — running it would crash with `AttributeError`. Even fixed, it would only be exercising DeepFace's old detector-backend path, which production doesn't use for event photos anymore, so it wasn't measuring anything current.

**What we did:** deleted the file (and the now-empty `scripts/` directory) and the misplaced `services/Face_recognition.code-workspace` IDE file the same pass — user deleted both directly. Updated the two dangling README references (file tree, and the "re-run this benchmark yourself" pointer) to point at `benchmarks/detector_comparison.py`, which is the current, working, actually-representative benchmark.

**Status:** done.

### Idea, not yet started — drop DeepFace/TensorFlow entirely for event photos

Raised 2026-09-01, deliberately not implemented yet. The insightface `buffalo_sc` pack already downloaded for detection also ships its own ArcFace-trained embedding model (`w600k_mbf.onnx`, MobileFaceNet architecture) — visible in the pack's own load log, currently ignored (`allowed_modules=["detection"]`) in favor of routing crops back to DeepFace's ArcFace. Using it instead would let event photos skip TensorFlow entirely (detection + embedding both on ONNX Runtime), plausibly with a similar speedup to the one already measured for detection.

**Why this is a bigger step than the detector swap, not a repeat of it:**
- The detector swap could be validated by directly comparing embeddings from two detectors feeding the *same* embedder (near-0 distance = safe). A different **embedder** produces vectors in a different space entirely — that comparison doesn't apply. Would need validation against labeled same-person/different-person pairs instead, and likely a re-tuned match threshold (currently 0.68, calibrated for DeepFace's specific ArcFace model).
- Every cached embedding in `event_photos.embedding` is a DeepFace-ArcFace vector. Switching embedders invalidates all of it — needs a migration plan (wipe and recompute), not just a code change.
- `mtcnn` (still used for the selfie) is also TF-based; a full DeepFace removal would need to replace that too.

**Status:** documented, not started. Worth its own round of measurement (same discipline as the detector work above) before adopting.

---

## Cross-project status

This backend is one half of SnapFind AI — the Android client lives in `../SnapFindAI/` (its own README covers the app side). Tracking both projects' open items here since the reasoning behind them is shared:

1. **Backend performance** — in progress. Downscaling + horizontal worker scaling done; the event-photo detector went through a full opencv→retinaface→insightface cycle (see Work log) and is now both correct and fast. Still open: GPU inference, batch inference, larger-scale accuracy validation of the current detector.
2. **Android UI** — currently a bare-bones MVP (two buttons + a spinner); needs a redesign that reads as portfolio-quality. Not started.
3. **On-device / hybrid matching** — full on-device matching was investigated and found infeasible (model size + corpus size vs phone RAM/battery budget). Decided to reframe as a partial hybrid instead (client-side ML Kit pre-filtering, small-batch on-device toggle). Not started — deliberately sequenced after backend performance work since it's the most invasive change of everything on this list.
4. **Resume claims gap-check** — compared the resume's project bullets against actual repo state. True today: FastAPI, Python, face embeddings, REST API count (6 endpoints). The "25-30% improvement via preprocessing" claim now has real, measured backing — image downscaling + the detector swap together are a documented, reproducible speedup, not a guess. **Docker, PostgreSQL, and vector similarity search (pgvector) are now also true** — containerized, run and verified end-to-end against a live Postgres container with a real `vector(512)` column, not just written and hoped-for (see the Docker + PostgreSQL section and the pgvector Work log entry above). Still not yet true / roadmap-only: Google Drive ingestion, the "90%+" manual-effort-reduction number (still needs its own benchmark, unrelated to inference speed). **Decided 2026-09-02: not pursuing the 1000+ photo batch claim as engineering work.** 500 (the current `MAX_EVENT_PHOTOS` default) is already realistic for the actual use case (event/wedding albums rarely exceed a few hundred photos), and stress-testing to 1000+ would only have been to hit that specific resume number, not a real product need — cheaper and more honest to adjust the resume wording to "500+" than to build and validate for a scale the product doesn't actually need.

## Interview cheat-sheet

The elevator pitch: "This project uses FastAPI for the API layer, SQLite for persistent job/result storage, Redis as the message broker, and RQ as the Python job queue. Uploading photos creates a job in the database. When processing is requested, the API does not run face recognition in the request — it marks the job as queued, sends a message to Redis, and returns immediately. A separate worker consumes the queue, marks the job as processing, runs DeepFace matching, stores matches, and marks the job completed or failed. The client polls job status and fetches results when ready."

On reliability: "The database is the source of truth for job state. Atomic SQL updates prevent duplicate workers from processing the same job. Timestamps like `queued_at` and `processing_started_at` allow stale jobs to be retried, so a crash does not leave a job stuck in progress forever."

On the performance pass: "I measured before optimizing instead of guessing — wrote a benchmark script, found a 10x speedup was available just from downscaling images before inference, and in the process of benchmarking caught a broken dependency pin that had silently disabled event-photo face detection entirely."

On the detector swap: "When I found a fast detector was actually failing silently, I didn't just swap in whatever detector was slowest-but-safest — I tested a genuinely lightweight modern detector (SCRFD, same class as MobileNet-based RetinaFace) but specifically validated the accuracy risk before trusting it: I compared the embeddings it produced against the known-good pipeline's embeddings for the same photos, not just whether it 'detected a face.' That caught that alignment quality was preserved before I shipped it, instead of assuming a paper's speed number would translate directly to my setup."
