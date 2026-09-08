# Face Detector Comparison

Which face detector should SnapFind AI use for event photos? Three were tried; this is the measured data behind the current choice, kept runnable and separate from the production backend.

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

Measured against 4 real sample photos. Full output: [`results.json`](results.json).

To reproduce, drop a few photos into `benchmarks/sample_photos/` (gitignored — the originals aren't committed because they're photos of real people and this repo is public), then from the `Face_recognition/` repo root:

```powershell
venv\Scripts\python.exe benchmarks\detector_comparison.py
# or point it anywhere:
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

## Files

- `detector_comparison.py` — runnable script (needs this repo's venv)
- `results.json` — raw output, regenerated on each run
