"""Shared pieces: classes, lookup table, preprocessing, model."""
import cv2
import numpy as np
import torch.nn as nn

# 32 classes in standard Persian alphabetical order (index = class id)
LETTERS = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")

# Deterministic lookup table (class index -> Latin string), copied from the instructor's
# alphabet chart (project slides, p.3) so the output matches the expected answer.
LOOKUP = ["-", "b", "p", "t", "s̱", "j", "ch", "ḥ", "kh", "d", "ẕ", "r", "z",
          "zh", "s", "š", "ṣ", "ż", "ṭ", "ẓ", "ʿ", "gh", "f", "q", "k",
          "g", "l", "m", "n", "w", "h", "y"]
# Plain-keyboard version of the same table (no diacritics), for terminals / --ascii
LOOKUP_ASCII = ["a", "b", "p", "t", "s", "j", "ch", "h", "kh", "d", "z", "r", "z",
                "zh", "s", "sh", "s", "z", "t", "z", "'", "gh", "f", "q", "k",
                "g", "l", "m", "n", "w", "h", "y"]
# Letter names, shown next to the result so the exact letter is never ambiguous
NAMES = ["alef", "be", "pe", "te", "se", "jim", "che", "he (jimi)", "khe", "dal", "zal", "re", "ze",
         "zhe", "sin", "shin", "sad", "zad", "ta", "za", "eyn", "gheyn", "fe", "qaf", "kaf",
         "gaf", "lam", "mim", "nun", "vav", "he (do-cheshm)", "ye"]
assert len(LETTERS) == len(LOOKUP) == len(LOOKUP_ASCII) == len(NAMES) == 32

# The model has one extra answer: "this is not a Persian letter" (Latin letters, digits, objects, shapes)
NOT_A_LETTER = 32
NUM_CLASSES = 33

SIZE = 64          # input side in pixels (dots stay 3-4 px wide instead of 1-2)


def _strip_ruled_lines(bw):
    """Erase long thin lines that cross the whole image (notebook rules), keep letter strokes."""
    h, w = bw.shape
    fg = bw > 0
    rows = np.arange(h)[:, None]
    last0 = np.maximum.accumulate(np.where(~fg, rows, -1), axis=0)
    next0 = np.minimum.accumulate(np.where(~fg, rows, h)[::-1], axis=0)[::-1]
    vrun = next0 - last0 - 1                      # vertical run length through each pixel
    near = cv2.dilate(bw, np.ones((3, 3), np.uint8)) > 0
    found = cv2.HoughLines(bw, 1, np.pi / 720, max(20, int(0.7 * w)),
                           min_theta=np.pi / 2 - 0.21, max_theta=np.pi / 2 + 0.21)
    if found is None:
        return bw
    out, done = bw.copy(), []
    xs = np.linspace(0, w - 1, 60).astype(int)
    for rho, th in found[:, 0]:
        ys = np.round((rho - xs * np.cos(th)) / np.sin(th)).astype(int)
        ok = (ys >= 0) & (ys < h)
        if ok.mean() < 0.9 or abs(ys[ok].mean() - np.mean(ys[ok][[0, -1]])) > h:
            continue
        x_, y_ = xs[ok], ys[ok]
        if near[y_, x_].mean() < 0.9:               # must be ink along (almost) the whole width
            continue
        mid = int(y_.mean())
        if any(abs(mid - m) <= 3 for m in done):
            continue
        med = float(np.median(vrun[y_, x_]))
        if med > max(6, 0.03 * h):                  # too thick to be a ruled line
            continue
        done.append(mid)
        band = np.zeros_like(bw)
        cv2.polylines(band, [np.stack([x_, y_], 1).astype(np.int32)], False, 255, int(med) + 4)
        out[(band > 0) & (vrun <= med + 3)] = 0     # remove line pixels only
    return out


def _flatten_lighting(gray):
    """Even out shadows / gradients so one global threshold works. Does nothing on evenly lit images."""
    inv = np.median(gray) < 110                    # light letter on dark paper -> work on the inverse
    g = 255 - gray if inv else gray
    h, w = g.shape
    k = max(15, int(max(h, w) * 0.12)) | 1         # window must be wider than the strokes
    bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)       # brightness of the paper around each pixel (ink ignored)
    # Judge the lighting from this paper estimate, not from an Otsu split: with a shadow, Otsu
    # separates lit paper from shaded paper and the lit half alone looks evenly lit.
    if np.subtract(*np.percentile(bg, [95, 5])) < 25:
        return gray                                # paper is evenly lit: leave untouched
    out = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    return 255 - out if inv else out


def _binarize(gray):
    """Letter pixels = 255. Otsu normally; falls back to local contrast when Otsu clearly split the paper itself."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (bw > 0).mean() <= 0.30 or np.median(gray) < 110:
        return bw                                  # plausible amount of ink (or light-on-dark: handled later)
    h, w = gray.shape                              # >30% "ink": thin ink on uneven / low-contrast paper
    k = max(15, int(max(h, w) * 0.12)) | 1
    cap = np.uint8(min(255, int(np.median(gray)) + 35))             # ignore white margins / glare when judging the paper
    bg = cv2.dilate(np.minimum(gray, cap), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    diff = np.clip(bg.astype(np.int16) - gray, 0, 255).astype(np.uint8)   # how much darker than nearby paper
    strong = diff > max(40, 0.6 * np.percentile(diff, 99.9))        # dark pen ink: always kept
    weak = (diff > 25).astype(np.uint8)                             # faint pixels: dots, thin ends... and ruled lines
    ln = max(15, max(h, w) // 6)
    rules = cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ln, 1)))
    rules |= cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ln)))
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))            # long straight faint lines = paper rulings
    return (((weak > 0) & (rules == 0)) | strong).astype(np.uint8) * 255


def _remove_frames(bw):
    """Erase hollow rectangular borders (photo/scan frames) that run along the image edges."""
    H, W = bw.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    bx, by = max(2, int(0.08 * W)), max(2, int(0.08 * H))
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.6 * W or h < 0.6 * H:
            continue
        m = lab[y:y + h, x:x + w] == i
        sides = [m[:by].any(0).mean(), m[-by:].any(0).mean(),
                 m[:, :bx].any(1).mean(), m[:, -bx:].any(1).mean()]
        centre = m[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
        if sum(s >= 0.85 for s in sides) >= 3 and centre < 0.02:   # ring-shaped: a frame, not a letter
            out[lab == i] = 0
    return out


# how far (as a fraction of the letter size) a dot / broken piece may sit from the letter body
KEEP = dict(above=0.9, below=0.45, side=0.2, part_ratio=0.3, part_gap=0.8)         # default: protects real letters
KEEP_STRICT = dict(above=0.9, below=0.3, side=0.2, part_ratio=0.3, part_gap=0.5)   # fallback: also drops captions


def _keep_main_letter(bw, cfg=None):
    """Keep the main letter body plus its dots; drop captions, neighbours and specks."""
    cfg = cfg or KEEP
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 2:
        return bw
    H, W = bw.shape
    def touches(i):
        x, y, w, h, _ = st[i]
        return x == 0 or y == 0 or x + w >= W or y + h >= H
    body = max(range(1, n), key=lambda i: st[i][4] * (0.4 if touches(i) else 1.0))
    bx, by, bw_, bh, ba = st[body]
    size = max(bw_, bh)
    keep = np.zeros(bw.shape, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if i != body:
            if a < max(4, 0.01 * ba):                                  # noise speck
                continue
            gx = max(bx - (x + w), x - (bx + bw_), 0)
            gy = max(by - (y + h), y - (by + bh), 0)
            gap = max(gx, gy) / size
            if touches(i) and not touches(body) and gap > 0.1:      # clutter at the edge, letter is centred
                continue
            if a >= cfg["part_ratio"] * ba:                           # another piece of a broken letter
                ok = gap <= cfg["part_gap"]
            else:                                                      # dot-sized piece
                limit = cfg["above"] if y + h <= by else cfg["below"]
                ok = gap <= limit and gx <= cfg["side"] * size
            if not ok:
                continue
        keep |= lab == i
    return keep.astype(np.uint8) * 255


def load_image(src):
    """Read a file path or raw bytes as a BGR(A)/gray array, honouring the phone's EXIF rotation."""
    import io
    from PIL import Image
    data = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    try:
        o = Image.open(io.BytesIO(data)).getexif().get(0x0112, 1)
    except Exception:
        o = 1
    if o in (2, 4, 5, 7):
        img = cv2.flip(img, 1)
    if o in (3, 4):
        img = cv2.rotate(img, cv2.ROTATE_180)
    if o in (5, 6):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if o in (7, 8):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def _binary(img_bgr):
    """Image -> raw binary mask (ink = 255), before ruled lines, frames and clutter are removed."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Image is empty or unreadable.")
    if img_bgr.ndim == 3:
        if img_bgr.shape[2] == 4:  # transparent PNG: composite on white
            a = img_bgr[:, :, 3:4] / 255.0
            img_bgr = (img_bgr[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    # shrink huge photos so Otsu/blur behave consistently
    h, w = gray.shape
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gray = _ignore_margin(gray)
    gray = _flatten_lighting(gray)
    gray = cv2.medianBlur(gray, 3)
    bw = _binarize(gray)
    # if the border is foreground, polarity is wrong -> flip
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127 and (bw > 0).mean() > 0.4:   # (a frame hugging the edge is not a dark background)
        bw = 255 - bw
    return bw


def _strip_all_lines(bw):
    bw = _strip_ruled_lines(bw)
    return np.ascontiguousarray(_strip_ruled_lines(np.ascontiguousarray(bw.T)).T)


def _prepare_bw(img_bgr):
    """Image -> clean binary mask (letter = 255) before the main-letter selection."""
    return _remove_edge_clutter(_remove_frames(_strip_all_lines(_binary(img_bgr))))


def ruled_lines(img_bgr):
    """Mask (255) of the notebook / ruled lines in an image (same size as the image)."""
    bw = _binary(img_bgr)
    lines = ((bw > 0) & (_strip_all_lines(bw) == 0)).astype(np.uint8) * 255
    h, w = img_bgr.shape[:2]
    return lines if lines.shape == (h, w) else cv2.resize(lines, (w, h), interpolation=cv2.INTER_NEAREST)


def _ignore_margin(gray):
    """Overwrite the outermost pixels with their inner neighbours. Screenshots, crops and webcams
    often leave a 1-3 px white or black line on the edge; it fooled the shadow correction."""
    h, w = gray.shape
    m = max(2, int(0.01 * max(h, w)))
    if min(h, w) <= 8 * m:
        return gray
    return cv2.copyMakeBorder(gray[m:h - m, m:w - m], m, m, m, m, cv2.BORDER_REPLICATE)


def _remove_edge_clutter(bw):
    """Erase shapes that run along a long stretch of the image edge (desk, paper edge, shadow,
    part of a frame). A letter may touch the edge too, but only where a stroke ends."""
    n, lab, _, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 2:
        return bw
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    contact = np.bincount(edge, minlength=n)
    clutter = [i for i in range(1, n) if contact[i] > 0.15 * edge.size]
    if not clutter or len(clutter) == n - 1:       # nothing to drop, or it would drop everything
        return bw
    return np.where(np.isin(lab, clutter), 0, bw).astype(np.uint8)


def _caption_like(piece, body):
    """Tall, hollow, stroke-like piece well below the letter = small Latin caption text.
    Real dots are small and solid; broken halves of a letter are wide, so neither matches."""
    x, y, w, h, ar = piece
    bx, by, bww, bhh, _ = body
    size = max(bww, bhh)
    gap, rh, rw, fill = (y - (by + bhh)) / size, h / size, w / size, ar / (w * h)
    return gap > 0.2 and rw <= 0.8 and rh >= 0.25 and (fill <= 0.65 or rw <= 0.12)


def _pieces(kept):
    n, lab, st, _ = cv2.connectedComponentsWithStats(kept.astype(np.uint8), connectivity=8)
    body = max(range(1, n), key=lambda i: st[i][4]) if n > 1 else None
    return n, lab, st, body


def caption_suspect(img_bgr):
    """True when the strict crop removes a caption-like piece. Used by predict.py to decide
    whether the image carries a caption under the letter."""
    try:
        mask = _prepare_bw(img_bgr)
    except ValueError:
        return False
    kept = _keep_main_letter(mask, KEEP) > 0
    dropped = kept & ~(_keep_main_letter(mask, KEEP_STRICT) > 0)
    if not dropped.any():
        return False
    n, lab, st, body = _pieces(kept)
    if n < 3:
        return False
    for i in range(1, n):
        if i != body and (((lab == i) & dropped).sum() > 0.5 * st[i][4]) and _caption_like(st[i], st[body]):
            return True
    return False


def _drop_captions(kept):
    """Remove every caption-like piece (all of the caption text), keep letter body and dots."""
    n, lab, st, body = _pieces(kept > 0)
    out = kept.copy()
    for i in range(1, n):
        if i != body and _caption_like(st[i], st[body]):
            out[lab == i] = 0
    return out


def preprocess(img_bgr: np.ndarray, strict: bool = False, captions: bool = False, info: dict = None) -> np.ndarray:
    """BGR/gray image -> float32 SIZE x SIZE array, white letter on black, [0,1].
    If a dict is passed as info, info["box"] = (x, y, w, h) of the letter in the input image."""
    bw = _prepare_bw(img_bgr)
    bw = _keep_main_letter(bw, KEEP_STRICT if (strict or captions) else KEEP)
    if captions:                                   # strict crop, plus any caption text it still left
        bw = _drop_captions(bw)
    pts = cv2.findNonZero(bw)
    if pts is None:
        raise ValueError("No letter found (image looks blank).")
    x, y, w, h = cv2.boundingRect(pts)  # uses ALL foreground, so dots stay
    if info is not None:                           # undo the shrink done for huge photos
        k = max(img_bgr.shape[:2]) / max(bw.shape)
        info["box"] = tuple(int(round(v * k)) for v in (x, y, w, h))
    crop = bw[y:y + h, x:x + w]
    side = max(w, h)
    pad = max(2, side // 8)
    sq = np.zeros((side + 2 * pad, side + 2 * pad), np.uint8)
    oy, ox = (sq.shape[0] - h) // 2, (sq.shape[1] - w) // 2
    sq[oy:oy + h, ox:ox + w] = crop
    out = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32) / 255.0


class PersianCNN(nn.Module):
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))
        # 64 -> 32 -> 16 -> 8 -> 4: four blocks, so the head sees 128 x 4 x 4 like the 32 px design
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256),
                                  nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))