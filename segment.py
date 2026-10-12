"""Find the separate letters in a photo of a few letters (or a whole sheet), in reading order.

  find_letters(img) -> [Letter, ...]   rows from top to bottom, each row right to left (Persian order)

A letter = one main stroke shape plus the dots and small pieces next to it. Shapes are taken from
the same clean-up the model uses (ruled lines, frames, desk edges removed), so lines through the
letters do not join them together.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from common import _prepare_bw

DOT_REACH = 0.7     # a dot may sit this far (x letter height) above / below its letter
PIECE_GAP = 0.15    # two big pieces closer than this (x letter height) are one letter (e.g. gaf's top bar)
MIN_SIZE = 0.35     # a letter is at least this big compared with the biggest letter in the photo.
                    # Measured on multi_set: real letters 0.45+ (he, dal); stray pieces / caption text 0.33 or less.
                    # A small handwritten he next to a tall gheyn measured 0.38 (real_photos/multi_letter_photos)


@dataclass
class Letter:
    box: tuple          # (x, y, w, h) in the photo's own pixels
    row: int            # 0 = top row


def _gap(a, b):
    """Gaps (x, y) between two boxes (x0, y0, x1, y1); 0 where they overlap."""
    return max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0)


def _join(a, b):
    a[:] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _groups(bw):
    """Connected shapes -> letters (list of boxes x0, y0, x1, y1), dots and pieces joined to their letter."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return []
    H, W = bw.shape
    edge = {i for i in range(1, n) if st[i][0] == 0 or st[i][1] == 0
            or st[i][0] + st[i][2] >= W or st[i][1] + st[i][3] >= H}  # desk, paper edge, half a neighbour
    pen = max(1.0, _stroke_width(np.isin(lab, [i for i in range(1, n) if i not in edge])))
    boxes = [[st[i][0], st[i][1], st[i][0] + st[i][2], st[i][1] + st[i][3]] for i in range(1, n)
             if i not in edge and st[i][4] >= max(4, 0.5 * pen * pen)  # not specks
             and not (st[i][3] <= 3 * pen and st[i][2] >= 4 * st[i][3])]    # nor flat bits of a ruled line
    long_ = [b for b in boxes if max(b[2] - b[0], b[3] - b[1]) >= 4 * pen]   # strokes, not dots
    if not long_:
        return []
    size = np.median([max(b[2] - b[0], b[3] - b[1]) for b in long_])  # typical letter size
    groups = [list(b) for b in long_]
    merged = True                                                     # join pieces of one letter
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if max(_gap(groups[i], groups[j])) <= PIECE_GAP * size:
                    _join(groups[i], groups.pop(j))
                    merged = True
                    break
            if merged:
                break
    small = [g for g in groups if max(g[2] - g[0], g[3] - g[1]) < 0.35 * size]   # e.g. three dots in a clump
    groups = [g for g in groups if g not in small]
    for b in [b for b in boxes if b not in long_] + small:            # dots: to the letter above / below
        best, best_d = None, None
        for g in groups:
            gx, gy = _gap(b, g)
            if gx <= 0.25 * size and gy <= DOT_REACH * size:
                d = gx * 2 + gy
                if best_d is None or d < best_d:
                    best, best_d = g, d
        if best is not None:
            _join(best, b)
    return groups


def _stroke_width(mask):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return 2 * float(m.sum()) / max(1.0, sum(cv2.arcLength(c, True) for c in contours))


def _rows(groups):
    """Sort letters into rows (by vertical overlap), each row right to left."""
    rows = []
    for g in sorted(groups, key=lambda g: (g[1] + g[3]) / 2):
        cy, h = (g[1] + g[3]) / 2, g[3] - g[1]
        for r in rows:
            ry0, ry1 = np.median([x[1] for x in r]), np.median([x[3] for x in r])
            if ry0 - 0.25 * h <= cy <= ry1 + 0.25 * h:
                r.append(g)
                break
        else:
            rows.append([g])
    return [sorted(r, key=lambda g: -(g[0] + g[2])) for r in rows]


def find_letters(img, min_size=MIN_SIZE):
    """-> letters in reading order. Shapes much smaller than the biggest letter in the photo (specks,
    caption text) and shapes cut off by the photo edge (a neighbour half in the photo) are left out."""
    try:
        bw = _prepare_bw(img)
    except ValueError:
        return []
    return letters_in_mask(bw, max(img.shape[:2]) / max(bw.shape), min_size)


def letters_in_mask(bw, k=1.0, min_size=MIN_SIZE):
    """Same as find_letters, on a ready-made ink mask (ink = 255); boxes are scaled by k."""
    H, W = bw.shape
    groups = _groups(bw)
    if not groups:
        return []
    biggest = max(max(g[2] - g[0], g[3] - g[1]) for g in groups)
    m = 1
    groups = [g for g in groups if max(g[2] - g[0], g[3] - g[1]) >= min_size * biggest
              and g[0] >= m and g[1] >= m and g[2] <= W - m and g[3] <= H - m]
    out = []
    for r, row in enumerate(_rows(groups)):
        for g in row:
            out.append(Letter(tuple(int(round(v * k)) for v in (g[0], g[1], g[2] - g[0], g[3] - g[1])), r))
    return out
