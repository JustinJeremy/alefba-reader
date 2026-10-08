"""CLI:  python predict.py a.jpg b.jpg c.jpg [--ascii]"""
import argparse, os, sys, time
import cv2
import numpy as np
import torch
import marks
from common import (PersianCNN, preprocess, caption_suspect, load_image, _binary, _prepare_bw, SIZE,
                    LETTERS, LOOKUP, LOOKUP_ASCII, NAMES, NOT_A_LETTER)

# next to this file, so the app works no matter which folder it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persian_cnn.pt")

STRICT_BELOW = 0.99  # below this confidence the stricter crop (drops far-away marks such as a chart caption
                     # under the letter) is also tried; 0.99 fixed the last hard_set error, 0.9 did not

# Refusing photos without a letter (measured on all test sets, see build_profile.py):
MIN_CONTRAST = 12    # lightest minus darkest part of the (blurred) photo. Blank / dark / covered-camera
                     # frames 3-11, faintest real letter 14, hard_set 20+, real photos 96+
MAX_INK = 0.32       # share of the photo that looks like ink. Real photos up to 0.20 (noisiest synthetic
                     # print 0.30); random noise 0.34+, grainy dark camera frames 0.47+
MAX_DISTANCE = 0.65  # how unlike its letter's average a photo may look (0 = identical). Real photos up
                     # to 0.58, random noise and dark camera grain about 0.55-0.8
MIN_PERSIAN = 0.22   # on a photo with many shapes, the share that must look very much like a Persian letter
                     # (distance under 0.15). Persian alphabet sheets 0.32-0.36, English text 0.00-0.12
PROFILE_PATH = os.path.join(os.path.dirname(MODEL_PATH), "letter_profile.npz")

NO_LETTER_BLANK = "No Persian letter found: the photo is too dark or blank. Check the camera, the lighting and the lens."
NO_LETTER = "No Persian letter found in this photo. Make sure one letter is clearly visible and fills most of the photo."
NO_LETTER_EDGE = ("No complete Persian letter found: the shape found runs off the edge of the photo. "
                  "Centre the letter and leave some space around it.")
NOT_PERSIAN = "No Persian letter found: the writing in this photo does not look like Persian letters."

_model = None
_centres = None


def _letter_centres():
    """Average feature vector of each letter (None if letter_profile.npz is missing)."""
    global _centres
    if _centres is None and os.path.isfile(PROFILE_PATH):
        _centres = np.load(PROFILE_PATH)["centres"]
    return _centres


def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}. Run 'python train.py' first.")
        m = PersianCNN(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        _model = m
    return _model

def _probs(img, strict=False, captions=False):
    """-> (class probabilities, (cleaned SIZE x SIZE image the model looked at, letter box in img, features))"""
    info = {}
    view = preprocess(img, strict, captions, info)
    x = torch.tensor(view).unsqueeze(0).unsqueeze(0)
    m = load_model()
    with torch.no_grad():
        f = m.head[:3](m.features(x))                 # the 256 numbers the answer is computed from
        p = torch.softmax(m.head[3:](f), 1)[0]
    return p, (view, info["box"], torch.nn.functional.normalize(f, dim=1)[0].numpy())


def photo_contrast(img):
    """Lightest minus darkest part of the photo, after blurring away camera grain."""
    g = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY)
    k = min(1.0, 800 / max(g.shape))
    g = cv2.GaussianBlur(cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), (5, 5), 0)
    return float(np.percentile(g, 99.5) - np.percentile(g, 0.5))


def letter_distance(cls, feats):
    """How unlike the average of letter cls these features are (0 = identical), or 0 without a profile."""
    centres = _letter_centres()
    return 0.0 if centres is None else float(1 - feats @ centres[cls])

MANY_LETTERS = ("This photo shows many letters. Mark the one to read (circle it, highlight it or point "
                "an arrow at it), or drag a box around it.")


def predict_with_view(img, topk=3, select=None):
    """Read one letter from a photo.
    select: optional (x, y, w, h) box the user dragged around the letter.
    -> ([(class, confidence) x topk], (cleaned image, letter box in img), how the letter was found)"""
    ox = oy = 0
    how = "whole photo"
    if img is None or img.size == 0:
        raise ValueError("Image is empty or unreadable.")
    if photo_contrast(img) < MIN_CONTRAST:
        raise ValueError(NO_LETTER_BLANK)
    if (_binary(img) > 0).mean() > MAX_INK:          # speckle everywhere: noise, grain, a textured surface
        raise ValueError(NO_LETTER)
    if select is not None:
        x, y, w, h = (int(v) for v in select)
        x, y = max(0, x), max(0, y)
        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
        if w < 8 or h < 8:
            raise ValueError("The selected box is too small. Drag a box around the whole letter.")
        img, ox, oy, how = img[y:y + h, x:x + w], x, y, "your selection"
    target = marks.find_marked_letter(img)
    if target is not None:
        try:
            res, (view, (bx, by, bw, bh)) = _read_one(target.crop, topk)
            how = {"circle": "the circled letter", "highlight": "the highlighted letter",
                   "arrow": "the letter the arrow points at"}[target.kind]
            ox, oy = ox + target.offset[0], oy + target.offset[1]
            return res, (view, (bx + ox, by + oy, bw, bh)), how
        except ValueError:          # the "mark" held no letter: it was not a real mark, read the photo
            pass
    if select is None and marks.many_letters(img):
        raise ValueError(MANY_LETTERS if looks_persian(img) else NOT_PERSIAN)
    res, (view, (bx, by, bw, bh)) = _read_one(img, topk)
    if select is None and _runs_off_edge(img, (bx, by, bw, bh)):
        raise ValueError(NO_LETTER_EDGE)
    return res, (view, (bx + ox, by + oy, bw, bh)), how


def _runs_off_edge(img, box):
    """True when the shape that was read touches the photo edge. In every real letter photo tested
    (hard_set, rehearsal, real photos, 1000 dataset images) the letter never did; a shadow, a desk
    edge or a half-photographed letter does. Small, tightly cropped images are not judged."""
    H, W = img.shape[:2]
    if max(H, W) < 300:
        return False
    x, y, w, h = box
    m = 0.01 * max(H, W)
    return x <= m or y <= m or x + w >= W - m or y + h >= H - m


def looks_persian(img, limit=25):
    """Do the letter-sized shapes on a busy photo look like Persian letters (not English text)?"""
    centres = _letter_centres()
    if centres is None:
        return True
    bw = _prepare_bw(img)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < 6:
        return True
    areas = np.sort(st[1:, 4])[::-1]
    big = [i for i in range(1, n) if st[i][4] >= max(0.25 * areas[4], 0.0002 * bw.size)
           and st[i][3] >= 0.025 * bw.shape[0]]
    views = []
    for i in sorted(big, key=lambda i: -st[i][4])[:limit]:
        x, y, w, h, _ = st[i]
        side, pad = max(w, h), max(2, max(w, h) // 8)
        sq = np.zeros((side + 2 * pad, side + 2 * pad), np.uint8)
        oy, ox = (sq.shape[0] - h) // 2, (sq.shape[1] - w) // 2
        sq[oy:oy + h, ox:ox + w] = (lab[y:y + h, x:x + w] == i) * 255
        views.append(cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    if not views:
        return True
    m = load_model()
    with torch.no_grad():
        f = torch.nn.functional.normalize(m.head[:3](m.features(torch.tensor(np.array(views)).unsqueeze(1))), dim=1)
        out = m.head[3:](f)
        cls = out[:, :NOT_A_LETTER].argmax(1).numpy()          # nearest letter, for the distance
        refused = (out.argmax(1) == NOT_A_LETTER).numpy()      # the model's own "not a letter" answer
    dist = np.where(refused, 1.0, 1 - (f.numpy() * centres[cls]).sum(1))
    return float((dist < 0.15).mean()) >= MIN_PERSIAN


def _read_one(img, topk):
    """The photo (or crop) holds one letter -> top-k, (cleaned image, letter box)."""
    p, view = _probs(img)
    if caption_suspect(img):                # a caption (small Latin text) sits under the letter
        try:
            q, v = _probs(img, captions=True)  # same crop with the caption text removed
            if q.max() >= 0.5:
                p, view = q, v
        except ValueError:
            pass
    if p.max() < STRICT_BELOW:              # not sure enough -> also try the stricter crop
        try:
            q, v = _probs(img, strict=True)
            if q.max() > p.max():
                p, view = q, v
        except ValueError:
            pass
    view, box, feats = view
    if p[NOT_A_LETTER] >= 0.5:                      # the model itself says: this is not a Persian letter
        raise ValueError(NO_LETTER)
    p = p[:NOT_A_LETTER]                            # only the 32 letters are ever shown as answers
    if letter_distance(int(p.argmax()), feats) > MAX_DISTANCE:
        raise ValueError(NO_LETTER)
    conf, idx = p.topk(topk)
    return [(int(i), float(c)) for c, i in zip(conf, idx)], (view, box)

def predict_array(img, topk=3):
    return predict_with_view(img, topk)[0]

def predict_file(path):
    img = load_image(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return predict_array(img)

def utf8_console():
    """Windows consoles default to cp1252, which cannot print Persian letters."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

if __name__ == "__main__":
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ascii", action="store_true", help="plain letters, no diacritics (e.g. sh instead of š)")
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args()
    table = LOOKUP_ASCII if a.ascii else LOOKUP
    out = []
    for path in a.images:
        try:
            t = time.time(); (i, c), *_ = predict_file(path)
        except (ValueError, OSError) as e:          # unreadable / blank image: report it, keep going
            print(f"{path}: ERROR - {e}", file=sys.stderr); out.append("?")
            continue
        warn = "  [LOW CONFIDENCE]" if c < a.min_conf else ""
        print(f"{path}: {LETTERS[i]} ({NAMES[i]}) -> {table[i]}  conf={c:.2f}  {1000*(time.time()-t):.0f} ms{warn}")
        out.append(table[i])
    print("Output:", " ".join(out))