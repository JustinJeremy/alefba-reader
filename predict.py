"""CLI:  python predict.py a.jpg b.jpg c.jpg [--ascii]"""
import argparse, os, sys, time
import torch
import marks
from common import PersianCNN, preprocess, caption_suspect, load_image, LETTERS, LOOKUP, LOOKUP_ASCII, NAMES

# next to this file, so the app works no matter which folder it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persian_cnn.pt")

STRICT_BELOW = 0.99  # below this confidence the stricter crop (drops far-away marks such as a chart caption
                     # under the letter) is also tried; 0.99 fixed the last hard_set error, 0.9 did not

_model = None
def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}. Run 'python train.py' first.")
        m = PersianCNN(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        _model = m
    return _model

def _probs(img, strict=False, captions=False):
    """-> (class probabilities, (cleaned SIZE x SIZE image the model looked at, letter box in img))"""
    info = {}
    view = preprocess(img, strict, captions, info)
    x = torch.tensor(view).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(load_model()(x), 1)[0], (view, info["box"])

MANY_LETTERS = ("This photo shows many letters. Mark the one to read (circle it, highlight it or point "
                "an arrow at it), or drag a box around it.")


def predict_with_view(img, topk=3, select=None):
    """Read one letter from a photo.
    select: optional (x, y, w, h) box the user dragged around the letter.
    -> ([(class, confidence) x topk], (cleaned image, letter box in img), how the letter was found)"""
    ox = oy = 0
    how = "whole photo"
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
        raise ValueError(MANY_LETTERS)
    res, (view, (bx, by, bw, bh)) = _read_one(img, topk)
    return res, (view, (bx + ox, by + oy, bw, bh)), how


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
    conf, idx = p.topk(topk)
    return [(int(i), float(c)) for c, i in zip(conf, idx)], view

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