"""Make turned_set/: the single-letter test photos turned to other angles (TEST ONLY, never train on them).

Usage:  python make_turned_set.py          then:  python score_testset.py turned_set
Sources: rehearsal_set/q95 (the chart letters) and every single-letter folder of real_photos/ (the team's
handwriting and saved test photos). Each photo is turned to ANGLES and saved as turned_set/<angle>/,
keeping its name, so the score is shown per angle. Each turned photo is cropped to the largest
rectangle without empty corners, like a photo taken with the camera held at that angle.
turned_set/ is made again from the photos, so it is not kept in git.
"""
import glob, os, re, shutil, sys
import cv2
import numpy as np
from common import load_image

ANGLES = (15, 30, 45, 90, 270, 315, 330, 345, 135, 180)   # tilts both ways, sideways; 135/180: further round
OUT = "turned_set"
SOURCES = ["rehearsal_set/q95"] + sorted(d for d in glob.glob("real_photos/*")
                                          if os.path.isdir(d) and not d.endswith("multi_letter_photos"))


def inner_size(w, h, angle):
    """Largest upright rectangle inside a w x h photo turned by angle: what a camera held at that angle
    would see, with no empty corners (filled corners look like a big shape to the clean-up)."""
    a = np.radians(angle % 180)
    sin, cos = abs(np.sin(a)), abs(np.cos(a))
    long_side, short_side = max(w, h), min(w, h)
    if short_side <= 2 * sin * cos * long_side or abs(sin - cos) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin, x / cos) if w >= h else (x / cos, x / sin)
    else:
        cos2 = cos * cos - sin * sin
        wr, hr = (w * cos - h * sin) / cos2, (h * cos - w * sin) / cos2
    return int(wr), int(hr)


def turn_photo(img, angle):
    """img turned by angle degrees (counter-clockwise), cropped to the largest rectangle without corners."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    big = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    iw, ih = inner_size(w, h, angle)
    x0, y0 = (nw - iw) // 2, (nh - ih) // 2
    return big[y0:y0 + ih, x0:x0 + iw]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    n = 0
    for src in SOURCES:
        tag = os.path.basename(src)
        for f in sorted(glob.glob(os.path.join(src, "*"))):
            name = os.path.basename(f)
            if not re.match(r"\d{2}_", name) or "-" in name.split("_")[0]:
                continue                                    # single letters with a known class only
            img = load_image(f)
            if img is None:
                continue
            if max(img.shape[:2]) > 900:                    # keep the set small
                k = 900 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            for a in ANGLES:
                d = os.path.join(OUT, f"{a:03d}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{os.path.splitext(name)[0]}_{tag}.jpg"), turn_photo(img, a),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
    print(f"{n} turned photos in {OUT}/ ({len(ANGLES)} angles: {', '.join(map(str, ANGLES))})")


if __name__ == "__main__":
    sys.exit(main())
