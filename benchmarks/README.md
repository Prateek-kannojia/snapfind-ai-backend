# Benchmarks

Measured data behind SnapFind AI's model choices, kept runnable and separate from
production. **Read [results.html](results.html)** — it's the actual write-up: what each
model does, why three separate comparisons were needed, and every finding. `RESULTS.md`
is the raw table output the scripts below regenerate on every run — not committed
(gitignored), since nobody reads it directly; results.html is built from it.

The current answer: SCRFD for the selfie (not just event photos), `w600k_mbf` as the
embedder. Measured by running the real API through four configurations — see the sweep
below.

---

## Reproduce

```powershell
venv\Scripts\python.exe benchmarks\build_corpus.py    # once — builds the synthetic corpus
venv\Scripts\python.exe benchmarks\seed_jobs.py        # once — uploads every job via the real API

# code changes to face_matcher.py/settings.py need a rebuild first — source is
# baked into the image (COPY . .), there's no bind mount:
docker compose build api worker
docker compose up -d

venv\Scripts\python.exe benchmarks\evaluate_corpus.py --sweep
```

The sweep runs the real 13-job corpus once per config, force-recreating the `worker`
container between runs so it picks up new settings, and wiping cached embeddings first
so each config scores distances it actually computed rather than a stale cache from
the last one.

**The config**, in `core/settings.py` / `services/face_matcher.py`:

| Setting | Values | Effect |
|---|---|---|
| `SELFIE_DETECTOR_MODE` | `legacy` (default) / `scrfd` | `legacy`: today's DeepFace+mtcnn. `scrfd`: same detector + crop-from-original event photos already use |
| `EMBEDDER` | `deepface` (default) / `mbf` / `r50` | Which model embeds the aligned crop, for **both** selfie and event photos |

Defaults reproduce today's exact production behaviour — nothing changes until these
are set explicitly. This is a measurement tool, not a migration.

---

## Why two older scripts still exist

The sweep runs through the real API, so it's the most trustworthy comparison here —
but it doesn't cover everything.

**`detector_comparison.py`** (opencv vs retinaface vs SCRFD) — the sweep never
re-tests detector choice; every config uses SCRFD for event photos. This script is
still the only evidence for that choice: SCRFD found 30/30 faces at 283ms/photo,
retinaface found 29/30 at 14.4s/photo, opencv found 16/30. Not superseded.

**`compare_embedders.py`** (DeepFace vs `w600k_mbf`, on the desktop) — its port
validation is now superseded: the sweep validates the same models *inside the actual
worker container*, against `face_matcher.py`'s real code, which is stronger evidence.
What isn't superseded is its accuracy-by-face-size breakdown (70px through 340px) —
the sweep only scores aggregate match/no-match, never by face size, and that's the
one place showing DeepFace and `w600k_mbf` tie exactly on the hardest (70px) bucket.

---

## The on-device contract

The Android leg doesn't exist yet — no real face detection has run on a phone, only
a timing spike. When it does, it needs to plug into the same scoring code every
number above already uses: a CSV with columns `job,file,d` (blank `d` = no face
found), one row per photo. `load_device_distances()` in `_common.py` reads it
straight into the same scorer as everything else — no separate on-device scoring
path to build later.

---

## Caveats

- The synthetic corpus (163 photos, 10 identities) is LFW faces composited onto
  generated backgrounds at controlled face sizes — it isolates face size cleanly but
  has none of the blur, lighting or angle variation of real event photos.
- The real-photo counts are small: 8 photos back the mbf-vs-r50 accuracy comparison,
  16 back the selfie-detector-fix comparison. Read differences of one photo as noise,
  not signal.
- All timings are CPU, on a dev machine without AVX2/FMA. Absolute numbers will
  differ on other hardware; relative ordering should hold since everything was
  measured identically.
- Nothing has run on a phone. The sweep proves which model to use, not that an
  Android port of it behaves identically — that's a separate port-validation
  question once real on-device detection exists.

---

## Files

| script | what it does |
|---|---|
| `build_corpus.py` | **Run first.** Builds the synthetic corpus into `sample_test_data/`. Real jobs there are untouched. |
| `seed_jobs.py` | **Run second.** Uploads every job through the real API so it exists in Postgres + object storage. Wipes and re-seeds by default (`--keep` to skip). |
| `evaluate_corpus.py` | Scores the seeded jobs against `/matches`. `--sweep` runs it once per detector/embedder config instead of just whatever's currently deployed. |
| `compare_embedders.py` | DeepFace vs `w600k_mbf` on identical crops, on the desktop. Now mainly useful for the face-size breakdown — see above. |
| `detector_comparison.py` | opencv vs retinaface vs insightface/SCRFD, identical input. Why SCRFD was picked. |

`_common.py` holds what they share: job discovery, label parsing (`p03_pos_240px.jpg`
= target present, 240px face — there's no separate ground-truth file to drift out of
sync), scoring, and the `RESULTS.md` section writer.

**Gitignored:** `sample_test_data/` (real photos + generated corpus), `*_log.txt`.
`RESULTS.md` is committed — each script rewrites its own section in place.