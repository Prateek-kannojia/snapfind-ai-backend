"""Did fixes 1-3 actually change the behaviour that was broken?

Before: one embedding per photo, from the largest face, cropped at 800px.
After:  every face, cropped from the original.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

from core.settings import settings  # noqa: E402
from services import face_matcher as fm  # noqa: E402

PHOTO = (
    BACKEND / "storage/uploads/6fe1d707-565a-4f7f-a5b8-4faf75fca594"
    / "event_photos/IMG_20250301_101253092_HDR.jpg"
)


def p(m=""):
    print(m, flush=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


p(f"settings: event_photo_max_dimension={settings.event_photo_max_dimension} "
  f"face_detector_size={settings.face_detector_size} "
  f"crop_from_original={settings.crop_from_original}")

original = cv2.imread(str(PHOTO))
detect_image = fm._downscale(original, settings.event_photo_max_dimension)
p(f"photo {original.shape[1]}x{original.shape[0]} -> detect on {detect_image.shape[1]}x{detect_image.shape[0]}\n")

p("AFTER (production path now): crop from original, embed every face")
after = fm._event_photo_embeddings(detect_image, original)
p(f"  faces embedded: {len(after)}")
if len(after) >= 2:
    d = cosine(after[0], after[1])
    p(f"  distance between the two different people: {d:.4f}")
    p(f"  -> {'SEPARATED (correct)' if d > 0.5 else 'STILL COLLAPSED'}")

p("\nBEFORE (old behaviour): crop from the 800px image")
before = fm._event_photo_embeddings(detect_image, detect_image)
p(f"  faces embedded: {len(before)}")
if len(before) >= 2:
    d = cosine(before[0], before[1])
    p(f"  distance between the two different people: {d:.4f}")
    p(f"  -> would be called the SAME person at threshold 0.68" if d <= 0.68 else "")
