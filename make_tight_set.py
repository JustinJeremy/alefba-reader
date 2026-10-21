"""Make tight_set/: the single-letter test photos cropped tightly round the letter (TEST ONLY).

Usage:  python make_tight_set.py          then:  python score_testset.py tight_set
Sources: the same photos as make_turned_set.py. Each is cropped to the letter (found by the app's own
clean-up) with MARGINS percent of white space left round it, saved as tight_set/<margin>/, so the
score is shown per margin. 00 = no white space at all, like a screenshot cropped to the letter.
tight_set/ is made again from the photos, so it is not kept in git.
"""
import os, re, shutil, sys
import cv2
from common import load_image, preprocess
from make_turned_set import SOURCES

MARGINS = (0, 2, 5)
OUT = "tight_set"


def main():
    import glob
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    n = 0
    for src in SOURCES:
        tag = os.path.basename(src)
        for f in sorted(glob.glob(os.path.join(src, "*"))):
            name = os.path.basename(f)
            if not re.match(r"\d{2}_", name) or "-" in name.split("_")[0]:
                continue
            img = load_image(f)
            if img is None:
                continue
            info = {}
            try:
                preprocess(img, info=info)
            except ValueError:
                continue
            x, y, w, h = info["box"]
            for margin in MARGINS:
                m = int(margin * max(w, h) / 100)
                crop = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
                d = os.path.join(OUT, f"{margin:02d}")
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{os.path.splitext(name)[0]}_{tag}.png"), crop)   # png: no blur
                n += 1
    print(f"{n} tightly cropped photos in {OUT}/ (margins: {', '.join(f'{m}%' for m in MARGINS)})")


if __name__ == "__main__":
    sys.exit(main())
