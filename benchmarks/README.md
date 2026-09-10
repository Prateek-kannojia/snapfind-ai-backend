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
| Status | current production | already in `storage/insightface/models/buffalo_sc/` |

`w600k_mbf.onnx` ships *inside* the `buffalo_sc` pack downloaded for detection — it has been sitting on disk unused since 2026-09-01, because `face_matcher.py` passes `allowed_modules=["detection"]`.

```powershell
venv\Scripts\python.exe benchmarks\embedder_comparison.py
```

Needs a selfie at `benchmarks/sample_photos/selfie/` in addition to the event photos.

### Results

**Speed** — same crops, same machine, CPU:

| | DeepFace ArcFace | `w600k_mbf` | Speedup |
|---|---|---|---|
| Embedding, all threads | 271 ms | **6.5 ms** | 41x |
| Embedding, 1 thread | 271 ms | **15.6 ms** | 17x |

Detection and alignment (shared, unchanged) add ~40-80 ms/photo. A 500-photo archive projects to well under a minute on one core of this machine.

**Accuracy** — 15 labelled pairs from 6 faces, one identity plus one bystander. A threshold only exists if the worst same-person pair scores closer than the best different-person pair:

| Embedder | Worst SAME | Best DIFFERENT | Gap | Verdict |
|---|---|---|---|---|
| DeepFace ArcFace | 0.9429 | 0.1375 | **−0.805** | no threshold separates these |
| **`w600k_mbf`** | 0.6788 | 0.7968 | **+0.118** | separable, e.g. **0.74** |

`w600k_mbf` classifies all 15 pairs correctly at 0.74. DeepFace scores two *different* people at 0.1375 while scoring the *same* person at 0.9429 — the ranges overlap completely, so no threshold works.

Before blaming the model, the incumbent was given its best configuration — production passes BGR crops with default normalization, and both are plausible bugs:

| Colour order | `normalization` | Gap |
|---|---|---|
| **BGR (production)** | **`base` (production)** | **−0.805** |
| BGR | `ArcFace` | −0.616 |
| RGB | `base` | −0.311 |
| RGB | `ArcFace` | −0.042 |

DeepFace fails in all four, but **production happens to use the worst one**. That's a live accuracy bug in `services/face_matcher.py`, independent of anything on-device.

### Two other things this turned up

**The 800px downscale destroys small faces.** `event_photo_max_dimension=800` turns a 90px face in a 4000px photo into an 18px face:

| Photo | face at 800px | face at 4000px |
|---|---|---|
| P1 | 318 px² (~18×18) | 8250 px² |
| P2 | 835 px² | 19504 px² |
| P3 | 755 px² | 18375 px² |

Raising it improves `w600k_mbf` materially (P2: 0.674 → 0.561) and DeepFace barely at all. The 10x speedup from downscaling was measured; its recall cost never was.

**Largest-face-only picks the wrong person.** P1 contains two people. The bystander's face is 8250 px²; the target's is 7770 px² — 6% smaller. `max(faces, key=area)` ([`face_matcher.py:174`](../services/face_matcher.py)) therefore scores the wrong person, and which one wins flips with resolution.

### Caveats

- **One identity and two different-person pairs.** Enough to show a direction, nowhere near enough to trust 0.74 as a number.
- All photos from a single person on a single trip — one outfit, one hairstyle, similar lighting. No age, ethnicity, or lighting diversity, which is where face models usually fail.
- Timings are this dev machine's CPU, not a phone: no thermal throttling, no full-size JPEG decode, different instruction set.
- Nothing in the production pipeline was changed.

---

---

## 3. Production verification (all sample jobs)

This is the run that found the bug, kept as a record of the *broken* pipeline —
the numbers below are from before the two-stage crop fix.

It used a script that ran the real `face_matcher.py` functions over every job
under `storage/uploads/`. Those jobs are now folded into the corpus instead, so
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

`face_matcher.py:45` drives the detector's input size from the same setting as the downscale:

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

Four scripts. Each answers one question and writes its own section of
[`RESULTS.md`](RESULTS.md), so they can be run separately and in any order.

| script | question it answers |
|---|---|
| `build_corpus.py` | **Run first.** Builds the labelled corpus: 10 LFW identities composited into phone-sized canvases at controlled face sizes, plus the real jobs already in `storage/uploads/`, with ground truth for both. |
| `evaluate_corpus.py` | How accurate is the pipeline? Precision/recall/F1 against that ground truth, by face size, across a threshold sweep. |
| `compare_embedders.py` | DeepFace ArcFace vs `w600k_mbf` on identical crops. Validates the ONNX port against insightface's own reference before trusting a single number. |
| `detector_comparison.py` | Which detector, and what did the two rejected ones actually cost? opencv → retinaface → insightface, all on identical input. |

`_common.py` holds what they share — scoring, cosine distance, production's
crop path, and the RESULTS.md section writer.

Real and synthetic are scored separately, never averaged. The real jobs are
all-positive, so they measure recall and cannot measure precision; the
synthetic ones carry the face-size axis that real photos don't.

**Gitignored**
- `sample_photos/` — real photos, kept out of a public repo
- `corpus/` — generated; rebuild with `build_corpus.py`
- `*_log.txt` — run output

`RESULTS.md` is committed: it's the readable output, and regenerated in place
rather than appended to.
