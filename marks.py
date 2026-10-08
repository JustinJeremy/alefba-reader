"""Find the one letter someone marked on a photo of a whole alphabet sheet.

A mark is a strongly coloured, bright stroke that is not the writing itself:
  - a circle or box drawn around the letter  -> read what is inside it
  - a highlighter stroke over the letter     -> read the letter under it
  - an arrow pointing at the letter          -> read the letter at its tip
Pen ink is dark, marker and highlighter colours are bright and saturated, so the two separate by
colour even when they have the same hue (light-blue marker on dark-blue writing). Any marker
colour works. Marks touching the photo edge are ignored (fingers, desk, coloured background).
"""
from dataclasses import dataclass
import cv2
import numpy as np
from common import _prepare_bw, ruled_lines

WORK = 1000                 # long side used for finding marks and letters (speed)
MIN_MARK = 0.0005           # smallest mark, as a fraction of the photo area
SOLID = 0.13                # stroke thickness / mark size above which a mark is a highlighter
STANDOUT = 50               # a mark must be this much more saturated than the paper. Measured: real
                            # marks +101 (highlighter) and +238 (marker); greenish webcam paper +14, +41


@dataclass
class Target:
    kind: str               # "circle", "highlight" or "arrow"
    crop: np.ndarray        # the marked letter, cut out of the photo, mark and ruled lines painted out
    offset: tuple           # (x, y) of the crop in the photo


def _hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, i].astype(np.int16) for i in range(3))


def _hue_gap(H, hue):
    return np.minimum(abs(H - hue), 180 - abs(H - hue))


def _ink_hue(bgr):
    """Hue of coloured pen ink, or None for black / grey writing."""
    H, S, V = _hsv(bgr)
    dark = (V < 140) & (S > 50)
    return int(np.median(H[dark])) if dark.sum() > 50 else None


def _paper_saturation(bgr):
    """How colourful the paper is (white paper ~16, a tinted webcam photo ~60)."""
    H, S, V = _hsv(bgr)
    return int(np.median(S[V >= np.percentile(V, 40)]))


def _mark_pixels(bgr, ink_hue, paper_s=0):
    """Bright, saturated pixels that are not the ink colour."""
    H, S, V = _hsv(bgr)
    gap = _hue_gap(H, ink_hue) if ink_hue is not None else np.full(H.shape, 90, np.int16)
    return ((S >= max(70, paper_s + 40)) & (V >= 150) & ((gap > 12) | (S >= 200))).astype(np.uint8) * 255


def _with_soft_edge(bgr, strong):
    """Add the paler, less saturated rim of a marker stroke (same hue, just as bright)."""
    if not strong.any():
        return strong
    H, S, V = _hsv(bgr)
    hue, bright = int(np.median(H[strong > 0])), int(np.median(V[strong > 0]))
    r = max(3, int(0.006 * max(bgr.shape[:2])))
    near = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    soft = near & (_hue_gap(H, hue) <= 15) & (S >= 30) & (V >= bright - 40)   # dark ink stays
    return ((strong > 0) | soft).astype(np.uint8) * 255


def _paper_colour(bgr, mark):
    H, S, V = _hsv(bgr)
    ok = (S < 60) & (V > 120) & (mark == 0)
    return np.median(bgr[ok], axis=0) if ok.sum() > 20 else np.array([255, 255, 255])


def _find_mark(small, ink_hue, paper_s=0):
    """-> mask of the biggest mark in the small image, or None."""
    m = cv2.morphologyEx(_mark_pixels(small, ink_hue, paper_s), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < MIN_MARK * h * w or a > 0.3 * h * w:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:        # finger, desk, coloured border
            continue
        if best is None or a > st[best][4]:
            best = i
    if best is None:
        return None
    mask = (lab == best).astype(np.uint8) * 255
    _, S, _ = _hsv(small)
    return mask if np.median(S[mask > 0]) >= paper_s + STANDOUT else None


def _stroke_width(mask):
    """Average width of the mark's stroke, in pixels."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2 * (mask > 0).sum() / max(1.0, perimeter)


def _thickness(mask):
    """Stroke width of a mark relative to its size: ~0.06-0.1 for a pen line, ~0.17+ for a highlighter."""
    x, y, w, h = cv2.boundingRect(mask)
    return _stroke_width(mask) / max(w, h)


def _letters(small, mark, solid):
    """Ink shapes in the small image with the mark painted out -> (labels, stats)."""
    clean = small.copy()
    paint = mark if solid else cv2.dilate(mark, np.ones((5, 5), np.uint8))   # writing lies under a highlighter
    clean[paint > 0] = _paper_colour(small, mark)
    bw = _prepare_bw(clean)
    if bw.shape != mark.shape:
        bw = cv2.resize(bw, mark.shape[::-1], interpolation=cv2.INTER_NEAREST)
    _, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    return lab, st


def _arrow_tip(mask):
    """-> (tip point, unit direction) of an arrow, or None if the mark does not look like one.
    The head end holds more ink than the tail end (two barbs or a filled triangle), even when the
    tail is curved. A real arrow measured 3.4x longer than wide with 2.7x more ink at the head."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
    u = vecs[:, 1]                                               # main direction of the stroke
    t = (pts - c) @ u
    end = 0.3 * (t.max() - t.min())
    plus, minus = (t >= t.max() - end).sum(), (t <= t.min() + end).sum()
    if np.sqrt(vals[1] / max(vals[0], 1e-6)) < 1.8 or max(plus, minus) < 1.3 * max(1, min(plus, minus)):
        return None                                              # e.g. a letter written in coloured pen
    return (pts[np.argmax(t)], u) if plus >= minus else (pts[np.argmin(t)], -u)


def _inside(lab, n, region):
    return np.bincount(lab[region > 0].ravel(), minlength=n)


def _pick_at_tip(st, tip, direction, arrow_area):
    """The letter-sized ink shape closest to the arrow tip, in front of it."""
    best, best_d = None, None
    for i in range(1, len(st)):
        x, y, w, h, a = st[i]
        if a < max(40, 0.15 * arrow_area):                       # dots and specks (an arrow is about letter-sized)
            continue
        if np.dot(np.array([x + w / 2, y + h / 2]) - tip, direction) < -0.25 * max(w, h):
            continue                                             # behind the arrow
        nearest = np.array([np.clip(tip[0], x, x + w), np.clip(tip[1], y, y + h)], np.float32)
        d = np.linalg.norm(nearest - tip)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def _with_close_pieces(st, i):
    """Box of shape i plus the pieces right next to it (a letter whose stroke broke in two, such as
    the head and stem of mim on a small photo). Neighbouring letters on a sheet are further away."""
    x, y, w, h = (int(v) for v in st[i][:4])
    x1, y1 = x + w, y + h
    gap = 0.12 * max(w, h)                                     # measured: 0.10-0.15 all work
    for _ in range(2):                                           # a piece next to a piece
        for j in range(1, len(st)):
            jx, jy, jw, jh, ja = (int(v) for v in st[j])
            if ja < 0.05 * st[i][4]:
                continue
            dx = max(x - (jx + jw), jx - x1, 0)
            dy = max(y - (jy + jh), jy - y1, 0)
            if max(dx, dy) <= gap:
                x, y, x1, y1 = min(x, jx), min(y, jy), max(x1, jx + jw), max(y1, jy + jh)
    return x, y, x1 - x, y1 - y


def _blank_outside(crop, region, paper, near=None):
    """Paint out ink shapes lying mostly outside the marked region (whole shapes, not pixels).
    near: a wider region where small separate parts (a dot the highlighter missed) may still be
    kept, unless they are cut off by the crop edge, which means they belong to neighbouring writing."""
    try:
        bw = _prepare_bw(crop)
    except ValueError:
        return
    if bw.shape != region.shape:
        bw = cv2.resize(bw, region.shape[::-1], interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    inside = _inside(lab, n, region)
    close = _inside(lab, n, near) if near is not None else inside
    H, W = region.shape
    def cut_off(i):
        x, y, w, h, _ = st[i]
        return x == 0 or y == 0 or x + w >= W or y + h >= H
    out = [i for i in range(1, n) if inside[i] < 0.5 * st[i][4]
           and (close[i] < 0.5 * st[i][4] or cut_off(i))]
    if out:
        crop[cv2.dilate(np.isin(lab, out).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = paper


def find_marked_letter(img):
    """-> Target for the marked letter, or None when the photo has no mark."""
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        return None
    bgr = np.ascontiguousarray(img[:, :, :3])
    k = min(1.0, WORK / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    ink_hue, paper_s = _ink_hue(small), _paper_saturation(small)
    mark = _find_mark(small, ink_hue, paper_s)
    if mark is None:
        return None
    hull = cv2.convexHull(cv2.findNonZero(mark))
    hx, hy, hw, hh = cv2.boundingRect(hull)
    solid = _thickness(mark) >= SOLID
    lab, st = _letters(small, mark, solid)
    region = np.zeros(lab.shape, np.uint8)
    cv2.fillConvexPoly(region, hull, 1)
    inside = _inside(lab, len(st), region)
    enclosed = [i for i in range(1, len(st)) if inside[i] >= 0.7 * st[i][4] and st[i][4] >= 0.02 * hw * hh]

    if solid:            # highlighter: the writing under it turns pale, so it is found inside the crop
        kind, box = "highlight", (hx, hy, hw, hh)
    elif enclosed:       # a thin line with a letter inside it: read just inside the line, so neither
        kind = "circle"  # the line nor its pale rim is in the crop and ruled lines cross it edge to edge
        inner = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=int(1.5 * _stroke_width(mark)) + 2)
        if not inner.any():
            inner = region
        hull = cv2.convexHull(cv2.findNonZero(inner))
        box = cv2.boundingRect(hull)
    else:                # a thin line with nothing inside: an arrow
        arrow = _arrow_tip(mark)
        if arrow is None:
            return None
        i = _pick_at_tip(st, *arrow, (mark > 0).sum())
        if i is None:
            return None
        kind, box = "arrow", _with_close_pieces(st, i)

    # crop from the full-size photo, with room for dots above / below
    x, y, w, h = box
    p = {"circle": 0.0, "highlight": 0.15, "arrow": 0.4}[kind] * max(w, h)
    x0, y0 = max(0, int((x - p) / k)), max(0, int((y - p) / k))
    x1, y1 = min(bgr.shape[1], int((x + w + p) / k) + 1), min(bgr.shape[0], int((y + h + p) / k) + 1)
    crop = bgr[y0:y1, x0:x1].copy()

    # ruled lines: found on the whole page, where they still cross the full width
    lines = ruled_lines(small)
    lines = cv2.resize(cv2.dilate(lines, np.ones((3, 3), np.uint8)), bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]

    # paint out the mark (with its soft rim) and the ruled lines in the paper colour
    full_mark = _with_soft_edge(crop, _mark_pixels(crop, ink_hue, paper_s))
    if solid:            # under a highlighter the writing is tinted too: only paint the highlighter's own colour
        H, S, V = _hsv(small)
        hue, bright = int(np.median(H[mark > 0])), int(np.median(V[mark > 0]))
        H, S, V = _hsv(crop)
        full_mark &= ((_hue_gap(H, hue) <= 15) & (V >= bright - 40)).astype(np.uint8) * 255
    else:
        full_mark = cv2.dilate(full_mark, np.ones((3, 3), np.uint8))
    paper = _paper_colour(crop, full_mark | lines)
    crop[(full_mark > 0) | (lines > 0)] = paper

    if kind != "arrow":                                          # neighbours peeking into the corners
        reg = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(reg, np.round(hull[:, 0, :] / k - [x0, y0]).astype(np.int32), 1)
        near = None
        if kind == "highlight":                                  # letter ends may stick out a little
            r = max(3, int(0.12 * max(w, h) / k))
            near = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        _blank_outside(crop, reg, paper, near)
    _remove_line_debris(crop, paper)
    return Target(kind, crop, (x0, y0))


def _remove_line_debris(crop, paper):
    """Paint out pieces of a ruled line that painting the mark has broken into dashes: flat pieces
    on one row that together span at least half the crop. (A dot, even a dash-shaped one, never does.)"""
    try:
        bw = _prepare_bw(crop)
    except ValueError:
        return
    H, W = bw.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    flat = [i for i in range(1, n) if st[i][3] <= max(3, 0.04 * H) and st[i][2] >= 2 * st[i][3]]
    tol = max(3, 0.02 * H)
    done = set()
    for i in flat:
        if i in done:
            continue
        row = [j for j in flat if abs((st[j][1] + st[j][3] / 2) - (st[i][1] + st[i][3] / 2)) <= tol]
        span = max(st[j][0] + st[j][2] for j in row) - min(st[j][0] for j in row)
        if span >= 0.5 * W:
            done.update(row)
    if done:
        m = cv2.dilate(np.isin(lab, list(done)).astype(np.uint8), np.ones((3, 3), np.uint8))
        if m.shape != crop.shape[:2]:
            m = cv2.resize(m, crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        crop[m > 0] = paper


def many_letters(img, limit=35):
    """True when the photo shows a whole sheet of letters rather than one letter.
    Counts letter-sized shapes: at least a quarter of the 5th largest shape (so one long word does
    not set the scale) and at least 2.5% of the image tall (so specks of noise do not count).
    Measured: single-letter photos 1-6 (noisiest synthetic printed 26), alphabet sheets 44-50.
    Images under 400 px are never sheets (32 legible letters need more room); in tiny, very noisy
    images the noise itself breaks into hundreds of blobs."""
    if img is None or max(img.shape[:2]) < 400:
        return False
    try:
        bw = _prepare_bw(img)
    except ValueError:
        return False
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= limit:
        return False
    areas = np.sort(st[1:, 4])[::-1]
    big = st[1:, 4] >= max(0.25 * areas[4], 0.0002 * bw.size)
    tall = st[1:, 3] >= 0.025 * bw.shape[0]
    return int((big & tall).sum()) >= limit
