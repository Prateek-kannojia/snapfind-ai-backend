"""Build a labelled test corpus from LFW.

The three real sample jobs are one person on one trip — no way to measure
precision/recall. LFW (Labeled Faces in the Wild) is the standard public
face-verification benchmark: many photos per identity, so ground truth is
known.

LFW tiles are 250x250 with the face centred, which does NOT exercise the
"small face in a big photo" path that broke production. So each tile is
composited onto a phone-sized canvas at a controlled face size, sometimes
with a second person as a distractor.

Synthetic backgrounds are a real limitation — see the caveats printed at the
end. What this DOES give is exact ground truth across many identities and a
controlled face-size sweep, neither of which the real photos provide.

    python pipeline_audit/build_corpus.py <path-to-extracted-lfw>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "corpus"
JOBS = 10
PHOTOS_PER_JOB = (12, 18)      # inclusive range
CANVAS = (3000, 2000)          # w, h — phone-photo shaped
FACE_PX = [70, 110, 160, 240, 340]   # face width in the canvas; sweeps the size axis
TILE_FACE_FRACTION = 0.44      # an LFW face spans ~110px of the 250px tile

random.seed(20260909)


def p(m=""):
    print(m, flush=True)


def load_identities(lfw_root: Path, min_photos: int) -> dict[str, list[Path]]:
    people = {}
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        shots = sorted(person_dir.glob("*.jpg"))
        if len(shots) >= min_photos:
            people[person_dir.name] = shots
    return people


def background(w: int, h: int) -> np.ndarray:
    """Cheap non-uniform background — a flat colour makes detectors behave oddly."""
    base = np.zeros((h, w, 3), np.uint8)
    c1 = np.array([random.randint(60, 170) for _ in range(3)], np.float32)
    c2 = np.array([random.randint(60, 170) for _ in range(3)], np.float32)
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    base[:] = (c1 * (1 - ramp) + c2 * ramp).astype(np.uint8)
    noise = np.random.default_rng().integers(-12, 12, (h, w, 3), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def paste(canvas: np.ndarray, tile_path: Path, face_px: int, slot: int, slots: int) -> None:
    tile = cv2.imread(str(tile_path))
    if tile is None:
        return
    scale = face_px / (tile.shape[1] * TILE_FACE_FRACTION)
    tw, th = max(8, int(tile.shape[1] * scale)), max(8, int(tile.shape[0] * scale))
    tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    ch, cw = canvas.shape[:2]
    band = cw // slots
    x = slot * band + random.randint(0, max(1, band - tw))
    y = random.randint(0, max(1, ch - th))
    x, y = min(x, cw - tw), min(y, ch - th)
    canvas[y:y + th, x:x + tw] = tile


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_corpus.py <path-to-extracted-lfw>")
    lfw_root = Path(sys.argv[1])
    if not lfw_root.exists():
        raise SystemExit(f"not found: {lfw_root}")

    people = load_identities(lfw_root, min_photos=20)
    p(f"identities with >=20 photos: {len(people)}")
    if len(people) < JOBS + 5:
        raise SystemExit("not enough identities")

    chosen = sorted(people, key=lambda n: -len(people[n]))[: JOBS + 12]
    targets = chosen[:JOBS]
    distractor_pool = chosen[JOBS:]

    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest = []
    for job_i, target in enumerate(targets):
        shots = list(people[target])
        random.shuffle(shots)
        selfie_src = shots.pop()

        job_id = f"job{job_i + 1:02d}_{target.replace(' ', '_')}"
        job_dir = OUT / job_id
        (job_dir / "event_photos").mkdir(parents=True)
        (job_dir / "selfie").mkdir()

        # selfie: composited too, so it has the same "face in a scene" shape
        sc = background(*CANVAS)
        paste(sc, selfie_src, 420, 0, 1)
        cv2.imwrite(str(job_dir / "selfie" / "selfie.jpg"), sc, [cv2.IMWRITE_JPEG_QUALITY, 92])

        n_photos = random.randint(*PHOTOS_PER_JOB)
        n_positive = n_photos // 2
        photos = []
        for i in range(n_photos):
            positive = i < n_positive
            face_px = FACE_PX[i % len(FACE_PX)]
            two_people = (i % 3 == 0)
            canvas = background(*CANVAS)
            present = []

            slots = 2 if two_people else 1
            if positive and shots:
                paste(canvas, shots[i % len(shots)], face_px, 0, slots)
                present.append(target)
                if two_people:
                    d = random.choice(distractor_pool)
                    paste(canvas, random.choice(people[d]), face_px, 1, slots)
                    present.append(d)
            else:
                d1 = random.choice(distractor_pool)
                paste(canvas, random.choice(people[d1]), face_px, 0, slots)
                present.append(d1)
                if two_people:
                    d2 = random.choice([x for x in distractor_pool if x != d1])
                    paste(canvas, random.choice(people[d2]), face_px, 1, slots)
                    present.append(d2)

            name = f"p{i:02d}_{'pos' if positive else 'neg'}_{face_px}px.jpg"
            cv2.imwrite(str(job_dir / "event_photos" / name), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            photos.append({
                "file": name,
                "contains_target": positive,
                "face_px": face_px,
                "people": present,
            })

        manifest.append({
            "job_id": job_id,
            "target_identity": target,
            "selfie": "selfie/selfie.jpg",
            "event_photos": photos,
            "positives": sum(1 for x in photos if x["contains_target"]),
            "negatives": sum(1 for x in photos if not x["contains_target"]),
        })
        p(f"  {job_id}: {len(photos)} photos "
          f"({manifest[-1]['positives']} pos / {manifest[-1]['negatives']} neg)")

    (OUT / "ground_truth.json").write_text(json.dumps(manifest, indent=2))
    total = sum(len(j["event_photos"]) for j in manifest)
    p(f"\n{len(manifest)} jobs, {total} event photos, ground truth -> {OUT / 'ground_truth.json'}")
    p("\nCAVEATS: composited faces on synthetic backgrounds. Real photos have")
    p("lighting/pose/motion-blur variation this does not reproduce. Use it to")
    p("measure precision/recall vs face size, not to claim real-world accuracy.")


if __name__ == "__main__":
    main()
