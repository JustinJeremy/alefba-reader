"""CLI:  python predict.py a.jpg b.jpg c.jpg [--ascii]"""
import argparse, contextlib, os, sys, threading, time
import cv2
import numpy as np
try:
    import torch
except ImportError:          # the online copy runs without PyTorch: numpy_cnn.py does the same maths
    torch = None
import marks
from numpy_cnn import NumpyCNN, softmax, normalize
from segment import find_letters
from common import (PersianCNN, preprocess, caption_suspect, load_image, _binary, _prepare_bw, _grain,
                    _flatten_lighting, colourful, ruled_lines, ruled_tilt, SIZE,
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
# the second model (batch 16), trained on turned letters too: asked only when the first model is unsure or
# refuses a photo of one letter (python train.py --tilt makes these three files)
TILT_MODEL_PATH = os.path.join(os.path.dirname(MODEL_PATH), "persian_cnn_tilt.pt")
TILT_PROFILE_PATH = os.path.join(os.path.dirname(MODEL_PATH), "letter_profile_tilt.npz")
TILT_WEIGHTS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "persian_cnn_tilt.npz")
BACKUP_BELOW = 0.97 # a one-letter reading less sure than this (or a refusal) is also read by the tilt model;
                    # the surer of the two wins. Measured: upright scores unchanged, turned photos 440 -> 811/1390

NO_LETTER_BLANK = "No Persian letter found: the photo is too dark or blank. Check the camera, the lighting and the lens."
NO_LETTER = "No Persian letter found in this photo. Make sure one letter is clearly visible and fills most of the photo."
NO_LETTER_EDGE = ("No complete Persian letter found: the shape found runs off the edge of the photo. "
                  "Centre the letter and leave some space around it.")
NOT_PERSIAN = "No Persian letter found: the writing in this photo does not look like Persian letters."

_model = None
_net = None
_centres = None
_tilt = {}              # the tilt model: net, centres (loaded when first needed)
_use = threading.local()  # which model this thread reads with: "main" (default) or "tilt"
# which maths runs the model: PyTorch when it is installed (the laptop), else NumPy (the online copy).
# ALEFBA_BACKEND=numpy forces NumPy, to check on the laptop that both give the same answers.
BACKEND = "numpy" if torch is None or os.environ.get("ALEFBA_BACKEND") == "numpy" else "torch"


def has_tilt_model():
    weights = TILT_WEIGHTS_PATH if BACKEND == "numpy" else TILT_MODEL_PATH
    return os.path.isfile(weights) and os.path.isfile(TILT_PROFILE_PATH)


@contextlib.contextmanager
def _using(which):
    """Read with the main or the tilt model inside this block (per thread, so the web app's threads
    reading two photos at once never mix them up)."""
    before = getattr(_use, "which", "main")
    _use.which = which
    try:
        yield
    finally:
        _use.which = before


def _letter_centres():
    """Average feature vector of each letter (None if letter_profile.npz is missing)."""
    if getattr(_use, "which", "main") == "tilt":
        if "centres" not in _tilt:
            _tilt["centres"] = np.load(TILT_PROFILE_PATH)["centres"]
        return _tilt["centres"]
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


class _TorchNet:
    """The PyTorch model with the same forward() as NumpyCNN: (N, 64, 64) -> (256 features, 33 scores)."""
    def __init__(self, m):
        self.m = m

    def forward(self, x):
        with torch.no_grad():
            f = self.m.head[:3](self.m.features(torch.tensor(np.asarray(x, np.float32)).unsqueeze(1)))
            return f.numpy(), self.m.head[3:](f).numpy()


def warm_up():
    """Load the model(s) before the first photo (the web app does this when it starts)."""
    network()
    if has_tilt_model():
        with _using("tilt"):
            network(); _letter_centres()


def network():
    """The model, ready to read letters (loaded once)."""
    if getattr(_use, "which", "main") == "tilt":
        if "net" not in _tilt:
            if BACKEND == "numpy":
                _tilt["net"] = NumpyCNN(TILT_WEIGHTS_PATH)
            else:
                m = PersianCNN(); m.load_state_dict(torch.load(TILT_MODEL_PATH, map_location="cpu")); m.eval()
                _tilt["net"] = _TorchNet(m)
        return _tilt["net"]
    global _net
    if _net is None:
        _net = NumpyCNN() if BACKEND == "numpy" else _TorchNet(load_model())
    return _net

def _probs(img, strict=False, captions=False, colour=False):
    """-> (class probabilities, (cleaned SIZE x SIZE image the model looked at, letter box in img, features))"""
    info = {}
    view = preprocess(img, strict, captions, info, colour)
    f, out = network().forward(view[None])     # f: the 256 numbers the answer is computed from
    return softmax(out)[0], (view, info["box"], normalize(f)[0])


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
MIN_SHOW = 0.4      # a single letter read less surely than this is refused. Batch 13 model: right answers
                    # 0.64 and up on every test set; the weakest guesses (0.13-0.31) were wrong or not letters
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


LEVEL_BELOW = 1.01  # tilted ruled paper is always also read straightened (a turned letter can be read
                    # wrongly but surely, e.g. a tilted be as pe at 90%)
LEVEL_SURE = 0.9    # the straightened reading is used only when at least this sure,
LEVEL_BETTER = 0.0  # and at least as sure as the reading as it is
LEVEL_HOW = " (straightened along the ruled lines)"


PAD_BELOW = 0.9     # a reading less sure than this (or a refusal): also read the photo with a border added
                    # (0.97 also re-read a team photo cut tightly from its sheet, zal at 95%, as alef)
PAD_SHARE = 0.25    # border on each side, x the photo's longer side (paper colour: the photo's median)
PAD_EDGE = 0.07     # only for a tight crop: the letter comes this close (x the photo's side) to the edge
PAD_SIDES = 3       # on at least this many sides (a normal photo with space round the letter is never padded)
PAD_SURE = 0.9      # and the reading with the border is used only when at least this sure (a picture card
                    # cut close round its picture was re-read wrongly at 71%)


def read_photo(img, topk=3, select=None):
    """Read every letter in a photo (see _read_two_models). A letter cropped tightly (no white space round
    it, as a cropped screenshot often is) touches the photo's edges, and the clean-up removes shapes
    along the edges (desk, shadow, frame). So a photo read unsurely or refused is also read with a
    border of its own paper colour added; the surer one-letter reading wins, so a sure reading never
    changes. Measured on tightly cropped test letters: 84.9% -> 97.8% right."""
    try:
        first, err = _read_two_models(img, topk, select), None
    except ValueError as e:
        first, err = None, e
    sure = lambda r: min(x[0][1] for x, _ in r[0])
    if select is not None or img is None or img.size == 0 or (first and (len(first[0]) > 1
                                                                         or sure(first) >= PAD_BELOW)):
        if err:
            raise err
        return first
    if not _tight_crop(img):
        if err:
            raise err
        return first
    p = max(8, int(PAD_SHARE * max(img.shape[:2])))
    try:
        second = _read_two_models(_with_border(img, p), topk, None)
    except ValueError:
        second = None
    # a tight crop is one plain letter: the border must not let it be read as a marked (circled) letter
    # on a sheet (a photo of a person was read as "the circled letter" with the border round it)
    if (second is not None and len(second[0]) == 1 and sure(second) >= PAD_SURE
            and second[1].startswith("whole photo") and (first is None or sure(second) > sure(first))):
        return [(res, (view, _box_unpad(box, p, img.shape))) for res, (view, box) in second[0]], second[1]
    if err:
        raise err
    return first


def _tight_crop(img):
    """Is the photo cropped tightly round its writing (the ink comes close to PAD_SIDES of its edges)?"""
    pts = cv2.findNonZero(_binary(img))
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(pts)
    H, W = _binary(img).shape[:2]
    gaps = (x / W, y / H, (W - x - w) / W, (H - y - h) / H)
    return sum(g <= PAD_EDGE for g in gaps) >= PAD_SIDES


def _with_border(img, p):
    """img with p pixels of its own paper colour (the median colour) added on every side."""
    if img.ndim == 3 and img.shape[2] == 4:                     # transparent PNG: on white, like the clean-up
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    paper = ([int(v) for v in np.median(img.reshape(-1, img.shape[2]), axis=0)] if img.ndim == 3
             else int(np.median(img)))
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=paper)


def _box_unpad(box, p, shape):
    """A box in the photo with the border -> the same box in the photo as given (kept inside it)."""
    x, y, w, h = box
    x0, y0 = min(max(x - p, 0), shape[1] - 1), min(max(y - p, 0), shape[0] - 1)
    x1, y1 = min(max(x + w - p, x0 + 1), shape[1]), min(max(y + h - p, y0 + 1), shape[0])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _read_two_models(img, topk=3, select=None):
    """Read every letter in a photo (see _read_level). A photo of one letter that the main model reads
    unsurely (below BACKUP_BELOW) or refuses is also read by the tilt model (trained on turned letters
    too); the surer reading wins, so a sure upright reading never changes. Photos of several letters
    are rows read right to left, so they stay with the main model."""
    try:
        first, err = _read_level(img, topk, select), None
    except ValueError as e:
        first, err = None, e
    sure = lambda r: min(x[0][1] for x, _ in r[0])
    if not has_tilt_model() or (first and (len(first[0]) > 1 or sure(first) >= BACKUP_BELOW)):
        if err:
            raise err
        return first
    with _using("tilt"):
        try:
            second = _read_level(img, topk, select)
        except ValueError:
            second = None
    # only one letter from the tilt model: it is asked about one-letter photos, and several "letters"
    # from it are things the main model refused (a game screenshot read as 5 letters)
    if second is not None and len(second[0]) == 1 and (first is None or sure(second) > sure(first)):
        return second
    if err:
        raise err
    return first


def _read_level(img, topk=3, select=None):
    """Read every letter in a photo (see _read_upright). Ruled paper photographed at a tilt (or sideways)
    is also read straightened along its lines, when the photo as it is reads unsurely or not at all;
    the surer reading wins, so a confident reading never changes."""
    try:
        first, err = _read_upright(img, topk, select), None
    except ValueError as e:
        first, err = None, e
    if select is not None or (first and min(r[0][1] for r, _ in first[0]) >= LEVEL_BELOW):
        if err:
            raise err
        return first
    # colourful photos are not straightened: the edges in a picture card's drawing can look like ruling,
    # and notebook paper is (nearly always) not colourful
    tilt = None if colourful(img) else ruled_tilt(img)
    if tilt is None:
        if err:
            raise err
        return first
    level, back = _level(img, tilt)
    try:
        letters, how = _read_upright(level, topk, None)
    except ValueError:
        if err:
            raise err
        return first
    sure = min(r[0][1] for r, _ in letters)
    if sure < LEVEL_SURE or (first and sure < min(r[0][1] for r, _ in first[0]) + LEVEL_BETTER):
        if err:
            raise err
        return first
    return [(res, (view, _box_back(box, back, img.shape))) for res, (view, box) in letters], how + LEVEL_HOW


def _level(img, angle):
    """img turned back by angle (the ruled lines level), on a canvas big enough for all of it.
    -> (levelled image, matrix from levelled back to img coordinates)"""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    level = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return level, cv2.invertAffineTransform(m)


def _box_back(box, back, shape):
    """A box (x, y, w, h) in the levelled image -> the box around it in the original photo."""
    x, y, w, h = box
    pts = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float64)
    p = pts @ back[:, :2].T + back[:, 2]
    x0, y0 = np.clip(p.min(0), 0, [shape[1] - 1, shape[0] - 1]).astype(int)
    x1, y1 = np.clip(p.max(0), 0, [shape[1], shape[0]]).astype(int)
    return int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))


def _read_upright(img, topk=3, select=None):
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
    try:
        letters, how = _read_plain(img, topk, select, how)
    except ValueError:
        letters = None
        if select is None and colourful(img):          # nothing found: maybe a coloured letter beside a picture
            card = _read_picture_card(img, topk)
            if card is not None:
                return [_moved(card, ox, oy)], CARD_HOW
        raise
    if select is None and colourful(img):
        card = _read_picture_card(img, topk)
        if card is not None and min(r[0][1] for r, _ in letters) < CARD_UNSURE                 and _inside_much_smaller(card[1][1], [b for _, (_, b) in letters]):
            return [_moved(card, ox, oy)], CARD_HOW    # the reading swallowed the picture with the letter
    return [_moved(l, ox, oy) for l in letters], how


def _moved(letter, ox, oy):
    res, (view, (bx, by, bw, bh)) = letter
    return res, (view, (bx + ox, by + oy, bw, bh))


def _read_plain(img, topk, select, how):
    """read_photo after marks: a few separate letters, or one letter. Boxes relative to img."""
    several = _read_several(img, topk)
    if several:
        n = len(several)
        return several, f"{n} letters, right to left" + (" (your selection)" if select is not None else "")
    res, (view, box) = _read_one(img, topk)
    if select is None and _runs_off_edge(img, box):
        raise ValueError(NO_LETTER_EDGE)
    if res[0][1] < MIN_SHOW:                       # a guess this weak is no answer (an app icon read at 13%)
        raise ValueError(NO_LETTER)
    return [(res, (view, box))], how


def _inside_much_smaller(box, boxes):
    """Does box lie inside what was read (all boxes together), and is it much smaller than that?"""
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    m = 0.1 * max(box[2], box[3])
    inside = box[0] >= x0 - m and box[1] >= y0 - m and box[0] + box[2] <= x1 + m and box[1] + box[3] <= y1 + m
    return inside and (x1 - x0) * (y1 - y0) >= CARD_SMALLER * box[2] * box[3]         and max(box[2], box[3]) >= CARD_NOT_SLIVER * max(x1 - x0, y1 - y0)


CARD_HOW = "the coloured letter next to the picture"
CARD_SMALLER = 1.8  # the coloured letter is used when it is this much smaller than what was read,
CARD_UNSURE = 0.97   # the plain reading was less sure than this (pastel letters, tiles: read surely, kept)
CARD_NOT_SLIVER = 0.35  # and it is at least this long (x what was read): not a shiny stripe or outline
CARD_SAT = 90       # picture cards: a pixel is strongly coloured above this saturation (HSV, 0-255); the
                    # pale disc behind the picture and letter stays below it
CARD_MIN = 0.2      # a letter is at least this big (x the photo's longer side)
CARD_CONF = 0.7    # the letter must be read at least this surely
CARD_WORK = 400     # bigger photos are looked at this size (longer side), for speed


def _colour_shapes(img):
    """Shapes of one strong colour (with the small dots of any colour above / below them) -> masks.
    A cartoon letter is one colour; the picture next to it is many colours with outlines."""
    hsv = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    sat = ((hsv[:, :, 1] > CARD_SAT) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    hue = hsv[:, :, 0].astype(int)                 # 0..179 around the colour circle
    ns, slab, sst, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    shapes = []
    for c in range(0, 180, 10):
        d = np.minimum(abs(hue - c), 180 - abs(hue - c))
        m = (sat & (d <= 12)).astype(np.uint8)
        if m.sum() < 0.01 * H * W:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # shiny highlights on the letter
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            s = max(w, h)
            if s < CARD_MIN * max(H, W) or a < 0.12 * w * h:      # small bits, thin lines and outlines
                continue
            shape = lab == i
            if any((shape & o).sum() > 0.8 * shape.sum() for o in shapes):  # (found in the next colour too)
                continue
            full = shape.copy()
            for j in range(1, ns):                 # its dots: small, within its width, above or below
                bx, by, bw, bh, _ = sst[j]
                if max(bw, bh) < 0.35 * s and bx >= x - 0.15 * s and bx + bw <= x + w + 0.15 * s                         and by >= y - 0.6 * s and by + bh <= y + h + 0.6 * s:
                    full |= slab == j
            shapes.append(full)
    return shapes, sat


def _read_picture_card(img, topk, with_mask=False):
    """A coloured letter next to a picture (a children's alphabet card) -> (top-k, (view, box)) or None.
    Each shape of one strong colour is drawn black on white and read on its own; the surest letter wins,
    but only when it is sure (CARD_CONF). read_photo decides whether to use it."""
    k = min(1.0, CARD_WORK / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    shapes, _ = _colour_shapes(small)
    best = None
    for m in shapes:
        drawn = np.full(m.shape + (3,), 255, np.uint8)
        drawn[m] = 0
        drawn = cv2.copyMakeBorder(drawn, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        try:
            res, (view, _) = _read_one(drawn, topk)
        except ValueError:
            continue
        if res[0][0] == 0:                     # alef is one stroke: any strip of colour looks like it
            continue
        if res[0][1] >= CARD_CONF and (best is None or res[0][1] > best[0][0][1]):
            best = (res, view, m)
    if best is None:
        return None
    res, view, m = best
    ys, xs = np.nonzero(m)
    box = tuple(int(round(v / k)) for v in (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    return (res, (view, box), m) if with_mask else (res, (view, box))


def _read_several(img, topk):
    """A photo of a few separate letters -> each letter read on its own, in reading order; None when
    fewer than two real letters are found (shapes the model refuses, like a caption, do not count),
    so a photo of one letter is always read exactly as before.
    When shapes are refused (a shadow, a printed form box), the letters are looked for again without
    them: small letters were measured against them and dropped as dots, and a dot was not joined to its
    letter (notepad photo 30-18-10-21). Photos where every shape is a letter are read in one go."""
    refused, read = set(), {}                      # boxes the model refused / read before (box -> result)
    read_any, last_tried = False, set()
    lined = [None]                                 # the photo without ruled lines, made when first needed
    for _ in range(SEVERAL_TRIES):
        letters = _several_candidates(img, refused)
        if letters is None:
            return None
        before = len(refused)
        for letter in letters:
            if letter.box not in read:
                read[letter.box] = _read_candidate(img, letter.box, topk, lined)
        if not read_any and not any(read[l.box] for l in letters):
            return None                            # the first look found no letter at all: not a photo of
        read_any = True                            # letters (block puzzle pieces, batch 13), however sure later
        out = []                                   # looks are about the shapes that are left
        for letter in letters:
            if read[letter.box] is None and letter.box not in last_tried:   # a photo of letters: try the
                last_tried.add(letter.box)                                    # refused shapes once more
                read[letter.box] = _read_candidate(img, letter.box, topk, lined, last=True)
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


def _read_candidate(img, box, topk, lined, last=False):
    """One shape of a photo of several letters -> (top-k, (view, box in img)), or None when refused."""
    H, W = img.shape[:2]
    x, y, w, h = box
    p = int(0.3 * max(w, h)) + 4                   # paper around it, as in a photo of one letter (a small
    x0, y0 = max(0, x - p), max(0, y - p)          # he cut tightly was read as "not a letter"); a neighbour
                                                   # peeking in touches the crop edge and is dropped
    x1, y1 = min(W, x + w + p), min(H, y + h + p)
    got = None if last else _read_letter(img[y0:y1, x0:x1], topk)
    if got is None and not last:                   # a ruled line through a small letter: in the small crop
        if lined[0] is None:                       # the line is too short to be recognised as one
            lined[0] = _without_ruled_lines(img)
        got = _read_letter(lined[0][y0:y1, x0:x1], topk)
        if got is None:
            return None
    if got is None:                       # last try: the letter cut from the cleaned WHOLE photo. A
        if len(lined) < 2:                         # ruled line through a small letter (he as a circle) is
            try:                                   # removed there, but not in the small crop or above
                lined.append(_prepare_bw(img))
            except ValueError:
                lined.append(None)
        if lined[1] is not None:
            k = lined[1].shape[1] / W
            cx0, cy0 = int(round(x0 * k)), int(round(y0 * k))
            mask = lined[1][cy0:int(round(y1 * k)), cx0:int(round(x1 * k))]
            if mask.any():
                got = _read_letter(cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR), topk)
                if got is not None:
                    res, (view, (bx, by, bw, bh)) = got
                    return res, (view, (x0 + int(bx / k), y0 + int(by / k), int(bw / k), int(bh / k)))
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
    f, out = network().forward(np.array(views))
    f = normalize(f)
    cls = out[:, :NOT_A_LETTER].argmax(1)                       # nearest letter, for the distance
    refused = out.argmax(1) == NOT_A_LETTER                     # the model's own "not a letter" answer
    dist = np.where(refused, 1.0, 1 - (f * centres[cls]).sum(1))
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
    idx = np.argsort(-p, kind="stable")[:topk]
    return [(int(i), float(p[i])) for i in idx], (view, box)

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