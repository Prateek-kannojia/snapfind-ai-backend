# Benchmarks

Measured data behind SnapFind AI's model choices, kept runnable and separate from the production backend. Two comparisons live here:

1. **[Face detector comparison](#1-face-detector-comparison)** — which detector finds faces in event photos. Settled; the winner shipped.
2. **[Face embedder comparison](#2-face-embedder-comparison)** — which model turns a face into a vector. Open; this is the gate for on-device matching.

They ask opposite questions and therefore measure different things — see [What each benchmark can prove](#what-each-benchmark-can-prove).

---

## 1. Face detector comparison

Which face detector should SnapFind AI use for event photos? Three were tried; this is the measured data behind the current choice.

## Pipeline

```mermaid
flowchart LR
    A[Event photo] --> B[Detect face]
    B --> C[Align to 112x112]
    C --> D[ArcFace embedding]
    D --> E[Cosine distance vs selfie]
```

Only the **Detect** step changed across the three rounds below — ArcFace has been the embedding model the whole time.

## Results

The table below is the original 4-photo run that drove the decision. Current
numbers, on a 30-photo corpus sample, are in [`RESULTS.md`](RESULTS.md) — the
ranking is unchanged.

To reproduce, build the corpus first, then from the `Face_recognition/` repo root:

```powershell
venv\Scripts\python.exe benchmarks\detector_comparison.py
# or point it at your own photos:
venv\Scripts\python.exe benchmarks\detector_comparison.py C:\some\folder
```

| Detector | Type | Detected | Avg time/photo (CPU) | Verdict |
|---|---|---|---|---|
| `opencv` (Haar cascade) | 2001, no deep learning | 0/4 | 111ms | Fast, but found nothing |
| `retinaface` (ResNet50) | 2019 CNN | 4/4 | 11.7s | Accurate, too slow |
| **`insightface` (SCRFD-500MF)** | Lightweight ONNX | **4/4** | **319ms** | Accurate and fast — current default |

insightface: ~37x faster than retinaface, same detection accuracy (same 4/4, same per-photo face counts).

## Accuracy check

Detecting a face isn't the same as embedding it well. Compared the ArcFace embeddings retinaface and insightface produced for the *same* photos — near 0 means both agree it's the same face, consistently processed:

| Photo | Agreement distance |
|---|---|
| IMG_20250301_101253092_HDR.jpg | 0.0142 |
| IMG_20250301_114959813_HDR.jpg | 0.0055 |
| IMG_20250301_115003398_HDR.jpg | 0.0053 |
| IMG_20250302_155231085_HDR.jpg | 0.2374 |

3/4 near-identical. The fourth is a real outlier — still well under the 0.68 match threshold, but unexplained. Needs a larger test set, not assumed to be noise.

## Why not MobileNet-0.25's 1.4ms?

The RetinaFace paper reports 1.4–25.6ms/image for a MobileNet-0.25 backbone — but that's a **GPU** number (Tesla P40), and DeepFace's `retinaface` backend is hardcoded to ResNet50 (MobileNet-0.25 isn't available through it at all). insightface's SCRFD-500MF is the same weight class, measured directly on this project's own CPU photos instead of borrowed from a different paper on different hardware.

## Caveats

- All numbers are CPU, from a sandboxed dev machine without AVX2/FMA — absolute ms will differ on real hardware; the *relative* ranking should hold. Re-run the script yourself for citable numbers.
- No GPU was available to measure anything here — the MobileNet-0.25 figure above is a published number, not something benchmarked in this project.
- Only 4 photos tested — no coverage yet of sunglasses, side profiles, low light, or group photos.

---

## 2. Face embedder comparison

Can SnapFind's face matching run entirely on an Android phone? That reduces to one question: the current embedding model can't (it's TensorFlow), so is there a mobile one that's good enough?

Two candidates:

| | DeepFace ArcFace | `w600k_mbf.onnx` |
|---|---|---|
| Architecture | ResNet, ArcFace-trained | MobileFaceNet, ArcFace-trained |
| Runtime | TensorFlow | ONNX Runtime |
| Size | 137 MB | **13.6 MB** |
| Runs on Android | no | **yes** |
| Status | current production | already in `models/insightface/models/buffalo_sc/` |

`w600k_mbf.onnx` ships *inside* the `buffalo_sc` pack downloaded for detection — it has been sitting on disk unused since 2026-09-01, because `face_matcher.py` passes `allowed_modules=["detection"]`.

```powershell
venv\Scripts\python.exe benchmarks\compare_embedders.py
```

Reads the corpus in `benchmarks/sample_test_data/`; build it with `build_corpus.py` first.

### Results

**Speed** — embed time for the whole corpus is in [`RESULTS.md`](RESULTS.md), regenerated on each run; the ratio has held around **30–40× in `w600k_mbf`'s favour** across runs, at one tenth the model size. Detection and alignment are shared and unchanged, so they cancel out of the comparison.

All CPU, on this dev machine. **Nothing has been measured on a phone** — see the caveats below.

**Accuracy** — live numbers are in [`RESULTS.md`](RESULTS.md); the script rewrites them on every run. Two populations, scored separately and never averaged:

- **Synthetic corpus** — labelled positives *and* negatives, so precision, F1 and the face-size breakdown are all defined. This is what picks each model's threshold.
- **Real photos** — the actual product input and the harder case. All-positive, so **recall only**; precision is undefined there and would read 1.000 by construction.

The headline is the interaction between the two. Pushing the threshold out until a model recovers every real photo, then asking what that costs back on the labelled corpus:

| Model | Threshold for 8/8 real | Synthetic precision there | Synthetic FP there |
|---|---|---|---|
| DeepFace ArcFace | 0.90 | 0.606 | **43** |
| **`w600k_mbf`** | **0.70** | **1.000** | **0** |

`w600k_mbf` recovers all eight real photos while still making zero mistakes on the labelled corpus. ArcFace can only get there by going so wide it produces 43 false positives — it cannot find these photos and stay usable at the same time.

> An earlier version of this section reported a **+0.118 separability gap** for `w600k_mbf` from 15 pairs of a single identity. The 153-photo corpus overturned it: both models have a *negative* gap (−0.62 and −0.24). `w600k_mbf` still wins, but it does not separate cleanly, and no threshold is perfect for either. Left here because the correction is the point — 15 pairs from one person was never enough to support that claim.

### Is the incumbent losing on merit, or because it's called wrong?

```powershell
venv\Scripts\python.exe benchmarks\compare_embedders.py --diagnose-incumbent
```

Two suspects, both read out of the installed `deepface` source rather than recalled. Only one is confirmed:

- **Scale — confirmed wrong.** `preprocessing.py:34`: `normalization="base"` returns the image untouched in `[0,1]`. ArcFace's own paper specifies `(x-127.5)/128` on `[0,255]`, which deepface ships as `normalization="ArcFace"` (`:66-71`). Production never passes the argument, so the model receives a pixel range it was not trained on.
- **Colour order — suspicious, not confirmed.** `representation.py:144` flips the input BGR→RGB, then `:173` flips it back, so the model receives whatever order it was handed — and `:43` documents the expected input as **BGR**. So deepface deliberately feeds its ArcFace weights BGR. Either that is a library-wide bug or those weights want BGR, and this repo cannot tell which. The sweep below doesn't settle it either: RGB is marginally better on the synthetic corpus, BGR is better on the real photos.

**Neither explains the loss.** Swept on the full corpus:

| config | thr | precision | recall | F1 | FP | real | all 8 at | FP there |
|---|---|---|---|---|---|---|---|---|
| **BGR + base (production)** | 0.65 | 0.955 | 0.829 | 0.887 | 3 | 6/8 | 0.90 | 43 |
| RGB + base | 0.70 | 0.969 | 0.829 | 0.894 | 2 | 4/8 | 0.80 | 17 |
| BGR + ArcFace | 0.70 | 0.955 | 0.829 | 0.887 | 3 | 6/8 | 0.85 | 22 |
| RGB + ArcFace *(correct)* | 0.65 | **1.000** | 0.803 | 0.891 | **0** | 3/8 | 0.85 | 26 |
| RGB + raw | 0.30 | 0.493 | 0.947 | 0.649 | 74 | 8/8 | 0.30 | 74 |
| **`w600k_mbf`** | 0.60 | **1.000** | **0.842** | **0.914** | **0** | 7/8 | **0.70** | **0** |

The best DeepFace can manage in any configuration is **F1 0.894**, against `w600k_mbf`'s **0.914** — and no configuration comes close to recovering all eight real photos at zero cost. **DeepFace loses on merit.**

**Two corrections this forced, both to claims made here earlier:**

1. *"Production uses the worst of the four configurations"* — **refuted.** That came from the 15-pair set. On the real corpus, production's BGR+base is mid-pack on synthetic and **joint-best on the real photos**.
2. Fixing the configuration makes real-photo recall **worse**, not better (6/8 → 3/8 for the correct RGB+ArcFace).

That second one looks backwards until you read it alongside the FP column. The sloppy configuration compresses distances, so everything matches more readily — on an all-positive set that reads as better recall, and on the mixed corpus it shows up as 3 false positives. RGB+ArcFace is the *conservative* and correct one: precision 1.000, zero false positives, lower recall. **A better score on an all-positive set is not evidence of a better model**, which is exactly why the real photos are never reported alone.

### Two bugs this turned up — both since fixed

| Bug | Then | Now |
|---|---|---|
| **The 800px downscale destroyed small faces.** Faces arrived at 318 / 835 / 755 px² (≈18×18 to 29×29) because crops were taken from the downscaled image | A 90px face in a 4000px photo became an 18px face | **Fixed.** `CROP_FROM_ORIGINAL=true` — detect on the downscale (cheap), crop from the original (detailed) |
| **Largest-face-only picked the wrong person.** One sample photo holds two people; the bystander's face was 8250 px² against the target's 7770 px² — 6% larger, so `max(faces, key=area)` scored the wrong one, and which face won flipped with resolution | Target unreachable in that photo | **Fixed.** Every detected face is scored; the closest wins |

A third bug from the same round — `det_size` being driven by `event_photo_max_dimension`, so raising the downscale also broke detection — is fixed too: `FACE_DETECTOR_SIZE` is now its own setting, pinned at 800.

### Caveats

- **Eight real photos, one identity.** They carry the weight of the headline claim, and eight is not many. The synthetic corpus supplies the scale (153 photos, 10 identities) but is LFW composites, not phone photos.
- The real photos are one person on one trip — one outfit, one hairstyle, similar lighting. No age, ethnicity or lighting diversity, which is where face models usually fail.
- The real set is all-positive, so it cannot measure false positives at all. Every precision figure quoted here comes from the synthetic corpus.
- Timings are this dev machine's CPU. **Nothing has run on a phone** — no thermal throttling, no full-size JPEG decode, different instruction set. The "runs on Android" row above is an inference from the file format, not a measurement.
- Only the *embedder* has been swapped in these runs. SCRFD detection has never been exercised through a mobile ONNX runtime.
- Nothing in the production pipeline was changed by this script — it calls the embedders directly rather than through the API.

---

## 3. Production verification (all sample jobs)

This is the run that found the bug, kept as a record of the *broken* pipeline —
the numbers below are from before the two-stage crop fix.

It used a script that ran the real `face_matcher.py` functions over every real
job on disk. Those jobs are now folded into the corpus instead, so
`evaluate_corpus.py` covers the same ground alongside the labelled data, and
that script is gone. Current numbers: [`RESULTS.md`](RESULTS.md).

| Job | Production (0.68) | On-device stack (0.74) |
|---|---|---|
| `24c8ea7c…` | **0 / 4** | 4 / 4 |
| `4092130d…` | **0 / 4** | 0 / 4 |
| `6fe1d707…` | **4 / 4** | 4 / 4 |

Two of the three are wrong. **Job `6fe1d707…`'s 4/4 is correct** — verified by zooming into the source photo, see below.

None of the three "selfies" is the close-up selfie the product asks for; they are distant full-body travel photos and one four-person group shot. What mtcnn hands the embedder:

| Job | Selfie face mtcnn used | What it is | Verdict |
|---|---|---|---|
| `24c8ea7c…` | 53×65 px | correct person, clearly usable | **0/4 is a real failure** |
| `4092130d…` | 190×235 px | **one of 4 people in a group photo**, picked arbitrarily (mtcnn returned 6 faces for 4 people) | wrong person selected |
| `6fe1d707…` | 33×42 px | correct person — a distant figure in profile wearing blue sunglasses | **4/4 is correct** |

### Root cause: the embedder false-accepts small faces

Job `6fe1d707…`'s 0.0066 is not a hallucinated face — the selfie is a 1842×4096 beach photo where the subject's face occupies **132×164 real pixels**, correctly detected. It is the **embedder** collapsing.

Controlled test, detector input pinned at `det_size=(800,800)` so only crop resolution varies. **`P1.f0` and `P2.f0` are different people:**

| Pair | Res | deepface | mbf |
|---|---|---|---|
| P1.f0 vs P2.f0 — **different people** | 800px | **0.0241** | 0.8357 |
| P1.f0 vs P2.f0 — **different people** | 3200px | 0.2200 | 0.8886 |
| P1.f0 vs P1.f1 — **different people** | 800px | **0.1596** | 0.6969 |
| P2 vs P4 — **same person**, small vs large face | — | 0.5688 | 0.4913 |

At production's 800px DeepFace rates two **different people** at **0.0241** against a 0.68 threshold. Every small face matches every other small face. `w600k_mbf` separates the same pair at 0.8357.

This explains all three jobs:

- **`6fe1d707…` 4/4** — 33×42 px selfie lands in the collapsed region with the 17–28 px event faces. **False positives from a real face.**
- **`24c8ea7c…` 0/4** — 53×65 px selfie sits *outside* the collapsed region, so it fails to match the small faces. Genuine false negative.
- **0.0090 / 0.0241 / 0.0314 photo-vs-photo** — three small faces, one a different person, collapsed together.

Failure mode: **false accepts on small faces, false rejects across a size gap.** Not a threshold problem.

### The obvious fix would have broken production

> **Since fixed.** `FACE_DETECTOR_SIZE` is now its own setting, pinned at 800, and crops come from the original image. The code below is what it looked like at the time.

At the time, `face_matcher.py` drove the detector's input size from the same setting as the downscale:

```python
app.prepare(ctx_id=-1, det_size=(settings.event_photo_max_dimension,) * 2)
```

Raising `event_photo_max_dimension` also raises SCRFD's input size, and its anchors are tuned for a fixed scale. On `IMG_20250302_155231085`:

| det_size | Faces found |
|---|---|
| 800 | 1 face, 33,088 px² |
| 1600 | 2 faces, 147,243 px² + 183 px² |
| **3200** | **730 px² + 505 px² — the 383×383 face is missed entirely** |

**Decouple these two before touching either.** Pin `det_size` at its tuned value; raise only the resolution the crop is sampled from. Section 2's sweep did exactly that (det_size 800, source 4000) and it worked.

### Withdrawn

The earlier claim that a **minimum-face-size gate** is needed. Job `6fe1d707…` works end-to-end from a ~33×41 px post-downscale face; a naive size gate would reject jobs that currently succeed. The problem is the embedder's behaviour on small faces, not their presence.

The selfie's `result[0]` bug is **latent, not firing**: the audit shows `[0]` was the largest of the 6 faces mtcnn returned for job `4092130d…`. Worth making explicit, not a current cause of failure.

### Caveat on the sample data

These three jobs are poor inputs. Part of what's measured here is test-data quality, not code quality.

---

## What each benchmark can prove

The two scripts hold opposite things constant, which changes what a comparison can mean:

| | Detector comparison | Embedder comparison |
|---|---|---|
| Varies | detector | embedder |
| Holds constant | embedder (ArcFace) | detector (SCRFD) |
| Embeddings comparable? | **yes** — same model both sides | **no** — different vector spaces |
| So it measures | agreement distance ≈ 0 | match/no-match decisions |

Embeddings are directly comparable only when produced by the same weights. That's why the detector work could validate itself with "agreement distance 0.0055" and the embedder work needs labelled pairs instead.

The same rule governs the Android port: running the *identical* `.onnx` on server and phone makes embeddings directly comparable again, so that comparison tests the port, not the model.

## Files

Five scripts. Each answers one question; the measuring ones write their own section of
[`RESULTS.md`](RESULTS.md), so they can be run separately and in any order.

| script | question it answers |
|---|---|
| `build_corpus.py` | **Run first.** Composites 10 LFW identities into phone-sized canvases at controlled face sizes, writing them into `sample_test_data/` alongside the real jobs already there. Only the `job*` folders are regenerated — the real ones cannot be rebuilt, so they are never touched. |
| `seed_jobs.py` | **Run second.** Uploads every job through the real API — init, presigned PUTs, multipart zip, complete — so they exist as real rows in Postgres with photos in object storage. Wipes first by default (`--keep` to skip). Run once, or again after a database wipe; not part of a measurement run. |
| `evaluate_corpus.py` | How accurate is the pipeline? Re-processes the seeded jobs and scores what `/matches` returns — precision/recall/F1 by face size, across a threshold sweep. Uploads nothing. |
| `compare_embedders.py` | DeepFace ArcFace vs `w600k_mbf` on identical crops, scoring **both** the synthetic corpus and the real photos. Validates the ONNX port against insightface's own reference before trusting a single number. Calls the embedders directly rather than through the API, so it compares any model without touching production. |
| `detector_comparison.py` | Which detector, and what did the two rejected ones actually cost? opencv → retinaface → insightface, all on identical input. |

`_common.py` holds what they share — job discovery, label parsing, scoring,
cosine distance, production's crop path, and the RESULTS.md section writer.

**There is no ground-truth file.** Labels live in the filenames
build_corpus.py writes (`p03_pos_240px.jpg` = target present, 240px face),
which the pipeline preserves as `original_filename`. The real photos carry no
such marker, so their labels — the one thing no script can derive — sit in
`REAL_JOBS` in `_common.py`.

Real and synthetic are scored separately, never averaged. The real jobs are
all-positive, so they measure recall and cannot measure precision; the
synthetic ones carry the face-size axis that real photos don't.

**Gitignored**
- `sample_test_data/` — real photos plus the generated corpus, kept out of a public repo

- `*_log.txt` — run output

`RESULTS.md` is committed: it's the readable output, and regenerated in place
rather than appended to.
