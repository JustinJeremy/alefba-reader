"""Make phone_set/: the test photos as a phone might take them (TEST ONLY, never train on them).

Usage:  python make_phone_set.py          then:  python score_testset.py phone_set
A stand-in for real phone photos, not a replacement: each photo gets what a hand-held phone adds, with
random amounts (the same every time: seeded by the file name):
  - the camera held at a slant (perspective: the far side of the page smaller) and turned a little
  - lower resolution, then out of focus or shaken (blur)
  - uneven light (brighter on one side) and a soft shadow
  - sensor noise and JPEG compression
Folders:
  mild/     the one-letter photos (rehearsal_set/q95 and real_photos, as in make_turned_set.py), a little
  strong/   the same photos, more of everything
  marked/   the whole alphabet sheets with one letter marked (marked_set), mild
  several/  the photos of several letters (multi_set), mild
The paper is extended past each photo before it is turned (make_turned_set.extend_paper), so no part of
a letter is cut off. phone_set/ is made again from the photos, so it is not kept in git.
"""
import glob, os, re, shutil, sys, zlib
import cv2
import numpy as np
from common import load_image
from make_turned_set import SOURCES, extend_paper

OUT = "phone_set"
#              slant  turn  scale        blur  light  noise  jpeg quality
LEVELS = {"mild":   (0.06, 8,  (0.6, 1.0), 1.0, 0.25, 4,  (70, 90)),
          "strong": (0.12, 20, (0.4, 0.8), 2.0, 0.45, 9,  (50, 75))}


def phone_photo(img, rng, level):
    """img as a phone photo: slanted, turned a little, smaller, blurred, unevenly lit, noisy."""
    slant, turn, scale, blur, light, noise, quality = LEVELS[level]
    ext, (p, w, h) = extend_paper(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    H, W = ext.shape[:2]
    src = np.float32([[p, p], [p + w, p], [p + w, p + h], [p, p + h]])
    # the camera at a slant: one side of the page further away (smaller), plus a small turn
    d = slant * max(w, h)
    side = rng.integers(4)                                   # which side is further away
    dst = src.copy()
    a, b = [(0, 1), (1, 2), (2, 3), (3, 0)][side]
    centre = src.mean(0)
    for i in (a, b):
        dst[i] += (centre - dst[i]) / np.linalg.norm(centre - dst[i]) * rng.uniform(0.3, 1.0) * d
    m = cv2.getRotationMatrix2D(tuple(map(float, centre)), rng.uniform(-turn, turn), 1.0)
    dst = np.hstack([dst, np.ones((4, 1), np.float32)]) @ m.T.astype(np.float32)
    big = cv2.warpPerspective(ext, cv2.getPerspectiveTransform(src, dst.astype(np.float32)), (W, H),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = np.floor(dst.min(0)).astype(int)
    x1, y1 = np.ceil(dst.max(0)).astype(int)
    out = big[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
    # lower resolution (a letter seen from further away), but at least 100 px across: a phone photo
    # never has fewer pixels than the small crops some test photos are
    k = max(rng.uniform(*scale), 100 / min(out.shape[:2]))
    if k < 1:
        out = cv2.resize(out, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # out of focus, or a shaken hand (a short streak in one direction)
    s = rng.uniform(0, blur)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), max(s, 0.1))
    else:
        n = max(1, int(round(2 * s + 1)))
        kern = np.zeros((2 * n + 1, 2 * n + 1), np.float32)
        ang = rng.uniform(0, np.pi)
        for t in np.linspace(-n, n, 4 * n + 1):
            kern[int(round(n + t * np.sin(ang))), int(round(n + t * np.cos(ang)))] = 1
        out = cv2.filter2D(out, -1, kern / kern.sum())
    # uneven light: brighter towards one side, and a soft round shadow
    hh, ww = out.shape[:2]
    yy, xx = np.mgrid[:hh, :ww].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = ((xx / ww - 0.5) * np.cos(ang) + (yy / hh - 0.5) * np.sin(ang)) + 0.5
    gain = 1 - light * rng.uniform(0.4, 1.0) * ramp
    cx, cy, r = rng.uniform(0, ww), rng.uniform(0, hh), rng.uniform(0.3, 0.7) * max(hh, ww)
    gain *= 1 - light * 0.5 * rng.uniform(0, 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    out = out.astype(np.float32) * gain[..., None]
    out += rng.normal(0, rng.uniform(1, noise), out.shape).astype(np.float32)     # sensor noise
    out = out.clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*quality))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def photos(folder, single_only):
    for f in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        name = os.path.basename(f)
        if not f.lower().endswith((".jpg", ".jpeg", ".png")) or not re.match(r"\d{2}[_-]", name):
            continue
        if single_only and "-" in name.split("_")[0]:
            continue
        yield f


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    jobs = [(f, "mild", os.path.basename(src)) for src in SOURCES for f in photos(src, True)]
    jobs += [(f, "strong", os.path.basename(src)) for src in SOURCES for f in photos(src, True)]
    jobs += [(f, "marked", "marked") for f in photos("marked_set", False)]
    jobs += [(f, "several", "multi") for f in photos("multi_set", False)]
    n = 0
    for f, kind, tag in jobs:
        img = load_image(f)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if max(img.shape[:2]) > 1200:                       # keep the set small
            k = 1200 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        rng = np.random.default_rng(zlib.crc32((kind + f).encode()))
        d = os.path.join(OUT, kind)
        os.makedirs(d, exist_ok=True)
        stem = os.path.splitext(os.path.basename(f))[0]
        out_name = (f"{stem}_{tag}.jpg" if kind in ("mild", "strong") else
                    f"{stem}_{os.path.basename(os.path.dirname(f))}.jpg" if kind == "marked" else f"{stem}.jpg")
        cv2.imwrite(os.path.join(d, out_name), phone_photo(img, rng, "strong" if kind == "strong" else "mild"),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"{n} phone-style photos in {OUT}/ (mild, strong, marked, several)")


if __name__ == "__main__":
    sys.exit(main())
