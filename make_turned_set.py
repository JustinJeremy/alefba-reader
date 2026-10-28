"""Make turned_set/: the single-letter test photos turned to other angles (TEST ONLY, never train on them).

Usage:  python make_turned_set.py             then:  python score_testset.py turned_set
        python make_turned_set.py --cropped   the batch 16-19 set (cropped corners), to compare old scores
Sources: rehearsal_set/q95 (the chart letters) and every single-letter folder of real_photos/ (the team's
handwriting and saved test photos). Each photo is turned to ANGLES and saved as turned_set/<angle>/,
keeping its name, so the score is shown per angle. The whole photo is kept and its empty corners are
filled with its paper colour (turn_whole), so no part of the letter is cut off. Until batch 20 each photo
was cropped to the largest rectangle without empty corners (turn_photo); at a slant that cut off parts
of many letters (the tail of te, the dots of se), so they could not be read even turned back exactly.
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


def turn_whole(img, angle):
    """img turned by angle degrees (counter-clockwise) with all of it kept, like a phone photo of a turned
    page: the paper is first extended past the photo (the ruled lines carry on sideways, plain paper above
    and below), then turned and cropped to where the photo went. Unlike turn_photo, no part of the letter
    is cut off, and the ruled lines still run across the whole photo."""
    if min(img.shape[:2]) > 40:
        img = img[2:-2, 2:-2]                     # some photos have a thin white rim; it must not be extended
    h, w = img.shape[:2]
    p = int(0.6 * max(h, w))
    paper = np.median(img.reshape(-1, img.shape[2] if img.ndim == 3 else 1), 0)        # mostly paper
    ext = cv2.copyMakeBorder(img, 0, 0, p, p, cv2.BORDER_REPLICATE).astype(np.float32)  # lines carry on
    rows = np.arange(p, dtype=np.float32)[:, None] / max(p * 0.15, 1)                # fade into plain paper
    fade = np.exp(-rows)[..., None] if img.ndim == 3 else np.exp(-rows)
    paper = paper.astype(np.float32) if img.ndim == 3 else np.float32(paper[0])
    top = ext[:1] * fade[::-1] + paper * (1 - fade[::-1])
    bottom = ext[-1:] * fade + paper * (1 - fade)
    ext = np.concatenate([top, ext, bottom]).clip(0, 255).astype(np.uint8)
    H, W = ext.shape[:2]
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    big = cv2.warpAffine(ext, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    corners = np.array([[p, p, 1], [p + w, p, 1], [p, p + h, 1], [p + w, p + h, 1]], np.float64) @ m.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    return big[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    turn = turn_photo if "--cropped" in sys.argv else turn_whole
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
                cv2.imwrite(os.path.join(d, f"{os.path.splitext(name)[0]}_{tag}.jpg"), turn(img, a),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
    print(f"{n} turned photos in {OUT}/ ({len(ANGLES)} angles: {', '.join(map(str, ANGLES))})")


if __name__ == "__main__":
    sys.exit(main())
