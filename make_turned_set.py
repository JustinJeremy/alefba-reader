"""Make turned_set/: the single-letter test photos turned to other angles (TEST ONLY, never train on them).

Usage:  python make_turned_set.py          then:  python score_testset.py turned_set
Sources: rehearsal_set/q95 (the chart letters) and every single-letter folder of real_photos/ (the team's
handwriting and saved test photos). Each photo is turned to ANGLES and saved as turned_set/<angle>/,
keeping its name, so the score is shown per angle. The empty corners are filled with the photo's own
paper colour. turned_set/ is made again from the photos, so it is not kept in git.
"""
import glob, os, re, shutil, sys
import cv2
import numpy as np
from common import load_image

ANGLES = (45, 90, 135, 180, 270)
OUT = "turned_set"
SOURCES = ["rehearsal_set/q95"] + sorted(d for d in glob.glob("real_photos/*")
                                          if os.path.isdir(d) and not d.endswith("multi_letter_photos"))


def turn_photo(img, angle):
    """img turned by angle degrees (counter-clockwise), the whole photo kept, corners in paper colour."""
    h, w = img.shape[:2]
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    paper = tuple(int(v) for v in np.median(edge.reshape(-1, img.shape[2] if img.ndim == 3 else 1), axis=0))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=paper)


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
