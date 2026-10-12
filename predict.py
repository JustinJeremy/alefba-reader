"""CLI:  python predict.py a.jpg b.jpg c.jpg [--ascii]"""
import argparse, os, sys, time
import cv2
import numpy as np
import torch
import marks
from segment import find_letters
from common import (PersianCNN, preprocess, caption_suspect, load_image, _binary, _prepare_bw, _grain,
                    _flatten_lighting, colourful, ruled_lines, SIZE,
                    LETTERS, LOOKUP, LOOKUP_ASCII, NAMES, NOT_A_LETTER)

# next to this file, so the app works no matter which folder it is started from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persian_cnn.pt")

STRICT_BELOW = 0.99  # below this confidence the stricter crop (drops far-away marks such as a chart caption
                     # under the letter) is also tried; 0.99 fixed the last hard_set error, 0.9 did not
COLOUR_BELOW = 0.9   # below this confidence (or when refused), a photo with colour in it is also read with
                     # the colour clean-up (common._colour_gray): pastel and cartoon letters, dots in another colour
COLOUR_MIN = 0.5     # ... and that reading replaces the usual one only when it is at least this sure (and surer).
                     # Measured: an app icon (no letter) was read as che at 0.25 by the colour clean-up

# Refusing photos without a letter (measured on all test sets, see build_profile.py):
MIN_CONTRAST = 12    # lightest minus darkest part of the (blurred) photo. Blank / dark / covered-camera
                     # frames 3-11, faintest real letter 14, hard_set 20+, real photos 96+
MAX_INK = 0.32       # share of the photo that looks like ink. Real photos up to 0.20 (noisiest synthetic
                     # print 0.30); random noise 0.34+, grainy dark camera frames 0.47+
MAX_DISTANCE = 0.65  # how unlike its letter's average a photo may look (0 = identical). Real photos up
                     # to 0.58, random noise and dark camera grain about 0.55-0.8
MAX_GRAIN = 25      # pixel-to-pixel jitter (common._grain) above which the photo is noise, not a grainy
                    # photo: real photos up to 5.4, film-grain training images up to 12, random noise 38+
MIN_PERSIAN = 0.22   # on a photo with many shapes, the share that must look very much like a Persian letter
PERSIAN_CLOSE = 0.25 # (distance under PERSIAN_CLOSE). Batch 10 model: Persian alphabet sheets 0.40-0.44,
                     # English text 0.00 (batch 9 model used 0.15: sheets 0.32-0.36, English 0.00-0.12)
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

def _probs(img, strict=False, captions=False, colour=False):
    """-> (class probabilities, (cleaned SIZE x SIZE image the model looked at, letter box in img, features))"""
    info = {}
    view = preprocess(img, strict, captions, info, colour)
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


def photo_grain(img):
    """How grainy the photo is (see common._grain), measured at the size the clean-up works at."""
    g = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY)
    k = min(1.0, 1024 / max(g.shape))
    g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g
    return _grain(_flatten_lighting(g))


def letter_distance(cls, feats):
    """How unlike the average of letter cls these features are (0 = identical), or 0 without a profile."""
    centres = _letter_centres()
    return 0.0 if centres is None else float(1 - feats @ centres[cls])

MANY_LETTERS = ("This photo shows many letters. Mark the one to read (circle it, highlight it or point "
                "an arrow at it), or drag a box around it.")


MAX_LETTERS = 20     # a photo with 2..MAX_LETTERS separate letters is read letter by letter
MULTI_MIN_CONF = 0.85  # and each such letter must be at least this sure. Measured: multi_set letters 0.95+,
                       # letter-like pieces of screenshots and English text up to 0.81 (when not refused anyway)
SEVERAL_TRIES = 3    # how often a photo of several letters is looked at (again without the refused shapes)
CAPTION_ROW = 0.65  # a row whose biggest shape is smaller than this (x the biggest letter) is caption text.
                    # Measured: caption letters 0.44-0.55; every row of real letters has one of 0.7 or more
MULTI_MAX_DISTANCE = 0.42  # stricter "unlike every letter" limit for each letter of a photo with several letters
                           # (more pieces = more chances that one only looks like a letter). Measured on the
                           # batch 10 model: multi_set letters up to 0.38, pieces of a game screenshot 0.46+


def predict_with_view(img, topk=3, select=None):
    """Read one letter from a photo (the first one, if the photo holds several: see read_photo).
    select: optional (x, y, w, h) box the user dragged around the letter.
    -> ([(class, confidence) x topk], (cleaned image, letter box in img), how the letter was found)"""
    letters, how = read_photo(img, topk, select)
    res, view = letters[0]
    return res, view, how


def read_photo(img, topk=3, select=None):
    """Read every letter in a photo: one letter, a marked letter on a sheet, or a few separate letters.
    -> ([(top-k, (cleaned image, box in img)), ...] in reading order (right to left), how they were found)"""
    ox = oy = 0
    how = "whole photo"
    if img is None or img.size == 0:
        raise ValueError("Image is empty or unreadable.")
    if photo_contrast(img) < MIN_CONTRAST:
        raise ValueError(NO_LETTER_BLANK)
    if (_binary(img) > 0).mean() > MAX_INK or photo_grain(img) > MAX_GRAIN:   # speckle everywhere: noise,
        raise ValueError(NO_LETTER)                                         # grain, a textured surface
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
            return [(res, (view, (bx + ox, by + oy, bw, bh)))], how
        except ValueError:          # the "mark" held no letter: it was not a real mark, read the photo
            pass
    if select is None and marks.many_letters(img):
        raise ValueError(MANY_LETTERS if looks_persian(img) else NOT_PERSIAN)
    several = _read_several(img, topk)
    if several:
        n = len(several)
        how = f"{n} letters, right to left" + (" (your selection)" if select is not None else "")
        return [(res, (view, (bx + ox, by + oy, bw, bh))) for res, (view, (bx, by, bw, bh)) in several], how
    res, (view, (bx, by, bw, bh)) = _read_one(img, topk)
    if select is None and _runs_off_edge(img, (bx, by, bw, bh)):
        raise ValueError(NO_LETTER_EDGE)
    return [(res, (view, (bx + ox, by + oy, bw, bh)))], how


def _read_several(img, topk):
    """A photo of a few separate letters -> each letter read on its own, in reading order; None when
    fewer than two real letters are found (shapes the model refuses, like a caption, do not count),
    so a photo of one letter is always read exactly as before.
    When shapes are refused (a shadow, a printed form box), the letters are looked for again without
    them: small letters were measured against them and dropped as dots, and a dot was not joined to its
    letter (notepad photo 30-18-10-21). Photos where every shape is a letter are read in one go."""
    refused, read = set(), {}                      # boxes the model refused / read before (box -> result)
    lined = [None]                                 # the photo without ruled lines, made when first needed
    for _ in range(SEVERAL_TRIES):
        letters = _several_candidates(img, refused)
        if letters is None:
            return None
        out, before = [], len(refused)
        for letter in letters:
            if letter.box not in read:
                read[letter.box] = _read_candidate(img, letter.box, topk, lined)
            if read[letter.box] is None:
                refused.add(letter.box)
            else:
                out.append(read[letter.box])
        if len(refused) == before:                 # nothing new refused: the letters are final
            break
    return out if len(out) >= 2 else None


def _several_candidates(img, skip):
    """The shapes of a photo of several letters that may be letters (caption rows dropped), or None
    when the photo does not look like letters written side by side."""
    letters = find_letters(img, skip=skip)
    if not 2 <= len(letters) <= MAX_LETTERS:
        return None
    size = lambda l: max(l.box[2], l.box[3])
    biggest = max(size(l) for l in letters)
    rows = {r: [l for l in letters if l.row == r] for r in {l.row for l in letters}}
    letters = [l for l in letters if max(size(m) for m in rows[l.row]) >= CAPTION_ROW * biggest]
    rows = [l.row for l in letters]                 # (a row of only small shapes is caption text: dropped)
    if not letters or max(rows.count(r) for r in set(rows)) < 2:   # letters are written side by side; a small
        return None                                                 # shape alone under a letter is its caption
    return letters


def _read_candidate(img, box, topk, lined):
    """One shape of a photo of several letters -> (top-k, (view, box in img)), or None when refused."""
    H, W = img.shape[:2]
    x, y, w, h = box
    p = int(0.3 * max(w, h)) + 4                   # paper around it, as in a photo of one letter (a small
    x0, y0 = max(0, x - p), max(0, y - p)          # he cut tightly was read as "not a letter"); a neighbour
                                                   # peeking in touches the crop edge and is dropped
    x1, y1 = min(W, x + w + p), min(H, y + h + p)
    got = _read_letter(img[y0:y1, x0:x1], topk)
    if got is None:                                # a ruled line through a small letter: in the small crop
        if lined[0] is None:                       # the line is too short to be recognised as one
            lined[0] = _without_ruled_lines(img)
        got = _read_letter(lined[0][y0:y1, x0:x1], topk)
    if got is None:
        return None
    res, (view, (bx, by, bw, bh)) = got
    return res, (view, (bx + x0, by + y0, bw, bh))


def _read_letter(crop, topk):
    """One letter of a photo of several letters, or None when it is not read surely enough."""
    try:
        res, view = _read_one(crop, topk, MULTI_MAX_DISTANCE)
    except ValueError:
        return None
    return (res, view) if res[0][1] >= MULTI_MIN_CONF else None


def _without_ruled_lines(img):
    """The photo with its ruled lines painted over in the paper colour. On the whole photo a notebook
    line clearly runs across the page; in the small crop around one letter it may not, and a short
    line through a small letter (he on the line) made it 'not a letter'. Only used for a letter that
    could not be read otherwise: painting can also cover pen ink lying along a line."""
    lines = ruled_lines(img)
    if not lines.any():
        return img
    out = img.copy()
    paper = _binary(img) == 0
    if paper.shape != lines.shape:
        paper = cv2.resize(paper.astype(np.uint8), lines.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0
    out[lines] = np.median(img[paper & ~lines], axis=0)
    return out


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
    return float((dist < PERSIAN_CLOSE).mean()) >= MIN_PERSIAN


def _usable(p, view, max_distance):
    """Would this reading be shown (not refused as 'not a letter' or as unlike every letter)?"""
    return p[NOT_A_LETTER] < 0.5 and letter_distance(int(p[:NOT_A_LETTER].argmax()), view[2]) <= max_distance


def _unsure(p, view, max_distance):
    return not _usable(p, view, max_distance) or p[:NOT_A_LETTER].max() < COLOUR_BELOW


def _read_one(img, topk, max_distance=MAX_DISTANCE):
    """The photo (or crop) holds one letter -> top-k, (cleaned image, letter box)."""
    try:
        p, view = _probs(img)
    except ValueError:                      # nothing darker than the paper: a light letter in colour?
        if not colourful(img):
            raise
        p, view = _probs(img, colour=True)
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
    if _unsure(p, view, max_distance) and colourful(img):   # a letter in colour: let colour count as ink
        for strict in (False, True):
            try:
                q, v = _probs(img, strict=strict, colour=True)
            except ValueError:
                continue
            if not _unsure(q, v, max_distance) or (q[:NOT_A_LETTER].max() > max(COLOUR_MIN, p[:NOT_A_LETTER].max())
                                                    and _usable(q, v, max_distance)):
                p, view = q, v
                break
    view, box, feats = view
    if p[NOT_A_LETTER] >= 0.5:                      # the model itself says: this is not a Persian letter
        raise ValueError(NO_LETTER)
    p = p[:NOT_A_LETTER]                            # only the 32 letters are ever shown as answers
    if letter_distance(int(p.argmax()), feats) > max_distance:
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
            t = time.time()
            img = load_image(path)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            letters, how = read_photo(img)
        except (ValueError, OSError) as e:          # unreadable / blank image: report it, keep going
            print(f"{path}: ERROR - {e}", file=sys.stderr); out.append("?")
            continue
        ms = 1000 * (time.time() - t)
        for k, (res, _) in enumerate(letters):
            i, c = res[0]
            warn = "  [LOW CONFIDENCE]" if c < a.min_conf else ""
            where = f" letter {k + 1}/{len(letters)}" if len(letters) > 1 else ""
            print(f"{path}{where}: {LETTERS[i]} ({NAMES[i]}) -> {table[i]}  conf={c:.2f}  {ms:.0f} ms{warn}")
        out.append(" ".join(table[r[0][0]] for r, _ in letters))
    print("Output:", " | ".join(out) if any(" " in o for o in out) else " ".join(out))