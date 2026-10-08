"""CLI:  python predict.py a.jpg b.jpg c.jpg [--ascii]"""
import argparse, os, sys, time
import torch
from common import PersianCNN, preprocess, caption_suspect, load_image, LETTERS, LOOKUP, LOOKUP_ASCII, NAMES

# next to this file, so the app works no matter which folder it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persian_cnn.pt")

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
    x = torch.tensor(preprocess(img, strict, captions)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(load_model()(x), 1)[0]

def predict_array(img, topk=3):
    p = _probs(img)
    if caption_suspect(img):                # a caption (small Latin text) sits under the letter
        try:
            q = _probs(img, captions=True)  # same crop with the caption text removed
            if q.max() >= 0.5:
                p = q
        except ValueError:
            pass
    if p.max() < 0.9:                       # still unsure -> try the stricter crop
        try:
            q = _probs(img, strict=True)
            if q.max() > p.max():
                p = q
        except ValueError:
            pass
    conf, idx = p.topk(topk)
    return [(int(i), float(c)) for c, i in zip(conf, idx)]

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