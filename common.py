"""Shared pieces: classes, lookup table, preprocessing, model."""
import cv2
import numpy as np
try:
    import torch.nn as nn
except ImportError:          # the online copy has no PyTorch; it reads with numpy_cnn.py instead
    nn = None

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
SMALL = 256        # photos smaller than this are enlarged before the clean-up (a 3 px blur would erase dots)


def _vertical_runs(fg):
    """Length of the vertical run of ink through each pixel (a ruled line is only 1-5 px thick)."""
    h = fg.shape[0]
    rows = np.arange(h)[:, None]
    last0 = np.maximum.accumulate(np.where(~fg, rows, -1), axis=0)
    next0 = np.minimum.accumulate(np.where(~fg, rows, h)[::-1], axis=0)[::-1]
    return next0 - last0 - 1


def _strip_ruled_lines(bw, gray=None):
    """Erase long thin lines that cross the whole image (notebook rules), keep letter strokes."""
    bw = _strip_broken_lines(_strip_straight_lines(bw), gray)
    w = bw.shape[1]
    if w >= 600 and gray is not None:              # whole page: a bent or folded page curves its lines,
        for x0 in (0, w // 3, 2 * w // 3):         # but over a third of the width they are nearly straight
            x1 = min(w, x0 + w // 3 + 20)
            bw[:, x0:x1] = _strip_broken_lines(np.ascontiguousarray(bw[:, x0:x1]), np.ascontiguousarray(gray[:, x0:x1]))
    return bw


def _strip_straight_lines(bw):
    """Lines that are perfectly straight and unbroken across (almost) the whole width."""
    h, w = bw.shape
    fg = bw > 0
    vrun = _vertical_runs(fg)                     # vertical run length through each pixel
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


def _ridge_across(gray, fit, t, ink):
    """True when the grey image shows a thin dark line along the fitted line over most of the
    width and near both edges: darker than the paper just above and below it. A ruled line runs
    across the paper; a straight pen stroke (a letter's flat base) ends where the letter ends."""
    h, w = gray.shape
    g = gray.astype(np.int16)
    xs = np.arange(w)
    yc = np.round(np.polyval(fit, xs)).astype(int)
    off = t + 3
    ok = (yc - off >= 0) & (yc + off < h)
    if ok.mean() < 0.8:
        return False
    x, y = xs[ok], yc[ok]
    on = np.min(np.stack([g[np.clip(y + d, 0, h - 1), x] for d in (-2, -1, 0, 1, 2)]), axis=0)
    side = np.minimum(g[y - off, x], g[y + off, x])          # paper just above and below
    step = max(8, 0.1 * (np.median(gray) - ink))             # a tenth of the paper-to-pen contrast
    dark = (side - on) >= step
    paper = np.nonzero(side >= np.median(gray) - 4 * step)[0]   # columns with paper around the line (not desk)
    if len(paper) < 0.5 * w:
        return False
    dark = dark[paper[0]:paper[-1] + 1]
    e = max(2, int(0.15 * len(dark)))
    return dark.mean() >= 0.6 and dark[:e].mean() >= 0.3 and dark[-e:].mean() >= 0.3


def _strip_broken_lines(bw, gray=None):
    """Ruled lines the straight-line search misses: slightly slanted or curved (paper not flat),
    or cut into pieces where a letter is written over them. A piece counts as a ruled line when,
    together with the pieces in line with it, it runs across the image (checked in the grey image
    when given, because faint parts of a line may not have become ink). Only the thin line pixels
    are erased, so strokes crossing the line stay; where the pen is clearly darker than the printed
    line, pen ink lying ON the line stays too."""
    h, w = bw.shape
    fg = bw > 0
    if w < 40 or h < 20 or not fg.any():
        return bw
    vrun = _vertical_runs(fg)
    tmax = max(3, int(0.02 * h))                  # thickest ruled line looked for
    thin = (fg & (vrun <= tmax)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    cand = [i for i in range(1, n) if st[i][2] >= max(12, 0.05 * w) and st[i][2] >= 4 * st[i][3]]   # (dashes too)
    out, used = bw.copy(), set()
    xs_all = np.arange(w)
    ink = float(np.percentile(gray[fg], 10)) if gray is not None else 0.0
    for i in sorted(cand, key=lambda i: -st[i][2]):
        if i in used:
            continue
        ys, xs = np.nonzero(lab == i)
        a, b = np.polyfit(xs, ys, 1)
        for _ in range(3):                         # refit on the straight part only (a letter may touch it)
            keep = np.abs(ys - (a * xs + b)) <= 1.5
            if keep.sum() < 10:
                break
            a, b = np.polyfit(xs[keep], ys[keep], 1)
        keep = np.abs(ys - (a * xs + b)) <= 1.5
        if abs(a) > 0.15 or len(np.unique(xs[keep])) < max(12, 0.05 * w):
            continue                               # slanted or curved: a pen stroke, not a ruled line
        group = [j for j in cand if j not in used and
                 abs(st[j][1] + st[j][3] / 2 - (a * (st[j][0] + st[j][2] / 2) + b)) <= tmax + 2]
        gm = np.isin(lab, group)
        ys, xs = np.nonzero(gm)
        near = np.abs(ys - (a * xs + b)) <= tmax + 2   # only the pixels along the line, not a touching letter
        gm[ys[~near], xs[~near]] = False
        ys, xs = ys[near], xs[near]
        cols = np.unique(xs)
        if len(cols) < 2:                          # nothing left along the line (a curved stroke whose
            continue                               # middle is off the fitted line): not a ruled line
        if gray is not None:
            fit = np.polyfit(xs, ys, 1)
            if not _ridge_across(gray, fit, tmax, ink):
                continue
        else:
            if len(cols) < 0.5 * w or cols[0] > 0.12 * w or cols[-1] < 0.88 * w:
                continue                           # does not run across the image
            fit = np.polyfit(xs, ys, 2 if cols[-1] - cols[0] > 0.5 * w else 1)
            if np.std(ys - np.polyval(fit, xs)) > 0.6 * tmax:
                continue
        used.update(group)
        med = float(np.median(vrun[gm]))
        band = np.zeros_like(bw)
        pts = np.stack([xs_all, np.polyval(fit, xs_all)], 1).round().astype(np.int32)
        cv2.polylines(band, [pts], False, 255, int(med) + 4)
        erase = (band > 0) & fg & (vrun <= med + 3)
        if gray is not None:                       # protect pen ink that lies on the line
            g_line = float(np.median(gray[gm]))
            pen = fg & (band == 0)
            if pen.sum() > 20:
                g_pen = float(np.percentile(gray[pen], 30))
                if g_pen < g_line - 25:
                    erase &= gray > (g_pen + g_line) / 2
        out[erase] = 0
    return out


GRAIN = 2.5        # grain level (grey levels) above which the photo is smoothed before thresholding.
                   # Measured: real photos and clean test sets 0-1.7, hard_set/shadow_noise ~5, film grain 1-5


def _grain(gray):
    """How grainy the photo is: typical pixel-to-pixel jitter on the paper, in grey levels
    (average absolute difference to the 3x3 average, away from the edges of strokes)."""
    g = gray.astype(np.float32)
    d = np.abs(g - cv2.blur(g, (3, 3)))
    edges = cv2.dilate((cv2.Canny(gray, 60, 150) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    calm = ~edges
    return float(np.mean(d[calm] if calm.mean() > 0.2 else d))   # (pure noise is "edges" everywhere)


BG_SHRINK = 4      # the paper brightness is a smooth gradient: on big photos (BG_SHRINK_FROM px or more on the
BG_SHRINK_FROM = 400   # short side) it is measured on a 4x smaller copy, ~16x less work (marked sheets)
RIM = 5            # phone sharpening leaves a thin bright rim around strokes. Taken as paper, it made a grey
                   # halo that was thresholded into stripes (real_photos/Kaye/09_dal_sheet.jpg read as zal).
                   # Pen-on-paper photos only: on coloured cartoon cards it shuffled the results (bus_test_set)


def _paper_brightness(g, k, rim=1):
    """Brightness of the paper around each pixel (ink ignored): the brightest pixel in a k-wide circle,
    smoothed. On big photos measured on a smaller copy (taking the brightest pixel of each 4x4 block,
    so thin ink still disappears) and scaled back up."""
    h, w = g.shape
    if rim > 1:
        g = cv2.erode(g, np.ones((rim, rim), np.uint8))  # thin bright rims (phone sharpening) are not paper
    if min(h, w) < BG_SHRINK_FROM:
        bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return cv2.GaussianBlur(bg, (0, 0), k / 3)
    f = BG_SHRINK
    small = cv2.dilate(g, np.ones((f, f), np.uint8))[f // 2::f, f // 2::f]
    ks = max(3, k // f) | 1
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bg = cv2.GaussianBlur(bg, (0, 0), ks / 3)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def _light_on_dark(gray):
    """Light letter on dark paper (chalk on a board)? The paper is dark, AND the letter is lighter than it:
    a dim photo of dark ink on grey paper (Mark's qaf sheet, paper at grey 108) has its ink far below the
    paper, chalk lies far above it. (Before, every photo with dark paper was treated as chalk and
    turned inside out.)"""
    lo, med, hi = np.percentile(gray, [1, 50, 99])
    return med < 110 and hi - med > med - lo


def _flatten_lighting(gray, rim=1):
    """Even out shadows / gradients so one global threshold works. Does nothing on evenly lit images."""
    inv = _light_on_dark(gray)                     # light letter on dark paper -> work on the inverse
    g = 255 - gray if inv else gray
    h, w = g.shape
    k = max(15, int(max(h, w) * 0.12)) | 1         # window must be wider than the strokes
    bg = _paper_brightness(g, k, rim)              # brightness of the paper around each pixel (ink ignored)
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
    bg = _paper_brightness(np.minimum(gray, cap), k)
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
    return _remove_rings(out)


def _remove_rings(bw):
    """Erase a closed ring around all the other ink: the edge of a coloured disc, circle or badge
    the letter is printed on. A letter's own loop (he, mim, ta) never has the rest of the letter
    inside it, and an open bowl (nun, jim) is not closed."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < 3:
        return bw
    H, W = bw.shape
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.5 * W or h < 0.5 * H:
            continue
        ring = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        flood = cv2.copyMakeBorder(ring, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(flood, None, (0, 0), 2)                    # flood the outside: what stays 0 is the hole
        hole = flood[1:-1, 1:-1] == 0
        if hole.sum() < 0.5 * (hole.sum() + a):
            continue                                             # mostly ink: not a thin ring
        others = (bw[y:y + h, x:x + w] > 0) & (ring == 0)
        rest = int((bw > 0).sum()) - int(a)
        if rest > 0 and (others & hole).sum() >= 0.95 * rest:
            bw = bw.copy()
            bw[lab == i] = 0
            return bw
    return bw


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
    inner = [i for i in range(1, n) if not touches(i) and st[i][4] >= max(30, 0.0003 * bw.size)]
    blocks = set()
    if inner:                                      # solid blocks at the edge (desk, finger, dark paper edge)
        pen = _stroke_width(lab[st_box(st, max(inner, key=lambda i: st[i][4]))] > 0)
        blocks = {i for i in range(1, n) if touches(i) and _stroke_width(lab[st_box(st, i)] == i) > 3 * max(pen, 2)}
    if inner and max(H, W) >= 300:                 # big photo: a shape touching the edge would be refused
        body = max(inner, key=lambda i: st[i][4])  # anyway (predict.py), so read the letter inside the photo
    else:
        body = max((i for i in range(1, n) if i not in blocks),
                   key=lambda i: st[i][4] * (0.4 if touches(i) else 1.0))
    bx, by, bw_, bh, ba = st[body]
    size = max(bw_, bh)
    keep = np.zeros(bw.shape, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if i != body:
            if i in blocks or a < max(4, 0.01 * ba):                   # desk / finger, or a noise speck
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
    return _binary_gray(img_bgr)[0]


COLOUR_INK = 2.5   # colour clean-up: grey levels of "ink" per unit of colour difference (Lab a/b) from
                   # the paper. A pastel letter (~35 units from white paper) becomes as dark as grey 165


def colourful(img_bgr):
    """True when some of the photo is clearly coloured (a coloured letter, a picture, coloured paper)."""
    if img_bgr is None or img_bgr.ndim != 3:
        return False
    small = cv2.resize(img_bgr[:, :, :3], (64, 64), interpolation=cv2.INTER_AREA)
    return float(np.percentile(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], 95)) >= 80


def _colour_gray(bgr, gray):
    """Grey image in which coloured parts count as ink too: each pixel is as dark as it is darker OR
    more differently coloured than the paper (the photo edge). A yellow, pink or light-blue letter is
    hardly darker than white paper, and a dot may be another colour than its letter: in plain grey
    both nearly vanish, here they become ink. Shadows change brightness, not colour, so they stay paper."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:]
    m = max(2, int(0.02 * max(gray.shape)))
    edge = np.concatenate([ab[:m].reshape(-1, 2), ab[-m:].reshape(-1, 2), ab[:, :m].reshape(-1, 2), ab[:, -m:].reshape(-1, 2)])
    paper = np.median(edge, axis=0)
    diff = np.sqrt(((ab - paper) ** 2).sum(2))
    return np.minimum(gray, np.clip(255 - COLOUR_INK * diff, 0, 255).astype(np.uint8))


def _binary_gray(img_bgr, colour=False):
    """-> (raw binary mask, grey image of the same size where ink is dark).
    colour=True: coloured parts count as ink as well as dark ones (see _colour_gray)."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Image is empty or unreadable.")
    if img_bgr.ndim == 3:
        if img_bgr.shape[2] == 4:  # transparent PNG: composite on white
            a = img_bgr[:, :, 3:4] / 255.0
            img_bgr = (img_bgr[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if colour:
            gray = _colour_gray(img_bgr, gray)
    else:
        gray = img_bgr
    # shrink huge photos so Otsu/blur behave consistently
    h, w = gray.shape
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    elif max(h, w) < SMALL:                        # small photo: enlarge so the clean-up keeps thin strokes and dots
        s = SMALL / max(h, w)
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
    gray = _ignore_margin(gray)
    gray = _flatten_lighting(gray, RIM if img_bgr.ndim == 2 or not colourful(img_bgr) else 1)
    grain = _grain(gray)
    gray = cv2.medianBlur(gray, 3)
    if grain > GRAIN:                              # film grain / camera noise: smooth it before thresholding
        gray = cv2.GaussianBlur(gray, (0, 0), min(3.0, grain / GRAIN))
    bw = _binarize(gray)
    # if the border is foreground, polarity is wrong -> flip
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127 and (bw > 0).mean() > 0.4:   # (a frame hugging the edge is not a dark background)
        bw = 255 - bw
    if np.median(gray[bw > 0]) > np.median(gray[bw == 0]) if (bw > 0).any() and (bw == 0).any() else False:
        gray = 255 - gray                           # light ink on dark paper: make ink the dark side
    return bw, gray


def _strip_all_lines(bw, gray=None):
    bw = _strip_ruled_lines(bw, gray)
    gt = None if gray is None else np.ascontiguousarray(gray.T)
    return np.ascontiguousarray(_strip_ruled_lines(np.ascontiguousarray(bw.T), gt).T)


TICK_ROW = 3       # at least this many small marks, evenly spaced along one ruled line, are the paper's
TICK_GAP = 5       # printed ticks (spaced at least TICK_GAP x their size apart), not dots: a letter's dots are
                   # close together. A tick above Kaye's dal made it zal (real_photos/Kaye/09_dal_sheet.jpg)


def _remove_line_ticks(bw, lines):
    """Erase the small printed ticks that some notebooks have along their ruled lines.
    bw: the mask without ruled lines; lines: the ruled lines that were taken out."""
    if not lines.any():
        return bw
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    near = cv2.dilate(lines, np.ones((9, 9), np.uint8)) > 0     # ticks may hang a few px off the line
    body = st[1:, 4].max() if n > 1 else 0
    marks = [i for i in range(1, n) if st[i][4] < 0.1 * body and near[lab == i].any()]
    out = bw
    used = set()
    for i in sorted(marks, key=lambda i: st[i][0]):
        if i in used:
            continue
        size = max(st[i][2], st[i][3])
        cy = st[i][1] + st[i][3] / 2
        row = [j for j in marks if j not in used and abs(st[j][1] + st[j][3] / 2 - cy) <= max(3, size)]
        if len(row) < TICK_ROW:
            continue
        xs = np.sort([st[j][0] + st[j][2] / 2 for j in row])
        gaps = np.diff(xs)
        if gaps.min() >= TICK_GAP * size and gaps.std() <= 0.2 * gaps.mean():
            out = out.copy() if out is bw else out
            out[np.isin(lab, row)] = 0
            used.update(row)
    return out


def _prepare_bw(img_bgr, colour=False):
    """Image -> clean binary mask (letter = 255) before the main-letter selection."""
    bw, gray = _binary_gray(img_bgr, colour)
    clean = _strip_all_lines(bw, gray)
    clean = _remove_line_ticks(clean, ((bw > 0) & (clean == 0)).astype(np.uint8))
    return _remove_edge_clutter(_remove_frames(clean))


def ruled_lines(img_bgr):
    """Mask (255) of the notebook / ruled lines in an image (same size as the image)."""
    bw, gray = _binary_gray(img_bgr)
    lines = ((bw > 0) & (_strip_all_lines(bw, gray) == 0)).astype(np.uint8) * 255
    h, w = img_bgr.shape[:2]
    return lines if lines.shape == (h, w) else cv2.resize(lines, (w, h), interpolation=cv2.INTER_NEAREST)


TILT_MIN = 10       # ruled paper tilted less than this (degrees) is read as it is: the line removal copes
TILT_LINES = 2      # at least this many long parallel lines (the ruling), so one straight stroke is not enough
TILT_SPREAD = 2.0   # the lines agree on their angle within this (degrees)


def ruled_tilt(img_bgr):
    """Angle (degrees, counter-clockwise, -90..90) of the ruled lines when the paper is turned at least
    TILT_MIN, else None. Long straight lines, nearly parallel, spread over the photo: notebook ruling.
    Turning the photo back by this angle makes the lines level, and the letter upright again
    (on ruled paper the lines show which way is up, apart from upside down)."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    k = 600 / max(gray.shape)
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    segs = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=40, minLineLength=int(0.25 * min(h, w)), maxLineGap=10)
    if segs is None:
        return None
    segs = segs.reshape(-1, 4).astype(float)                # (x1, y1, x2, y2) per line, any OpenCV version
    ang = np.degrees(np.arctan2(-(segs[:, 3] - segs[:, 1]), segs[:, 2] - segs[:, 0]))   # image y points down
    ang = (ang + 90) % 180 - 90                                                         # -90..90
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    best, best_len = None, 0
    for a in ang:                                     # the angle most of the long line length agrees on
        d = np.abs((ang - a + 90) % 180 - 90)
        tot = length[d <= TILT_SPREAD].sum()
        if tot > best_len:
            best, best_len = a, tot
    def separate_lines(at):
        """How many separate lines (not pieces of one line) run at this angle."""
        near = np.abs((ang - at + 90) % 180 - 90) <= TILT_SPREAD
        nx, ny = -np.sin(np.radians(at)), -np.cos(np.radians(at))
        mid = (segs[near][:, :2] + segs[near][:, 2:]) / 2
        offsets = np.sort(mid[:, 0] * nx + mid[:, 1] * ny)
        return (1 + int((np.diff(offsets) > 0.04 * min(h, w)).sum()) if len(offsets) else 0), near

    if separate_lines(0)[0] >= TILT_LINES:            # level ruling is there: the paper is straight already
        return None                                   # (a long upright stroke, like lam, is not ruling)
    lines, near = separate_lines(best)
    if lines < TILT_LINES:
        return None
    a = float(np.average(ang[near], weights=length[near]) if np.ptp(ang[near]) < 90 else best)
    return a if abs(a) >= TILT_MIN else None


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


def st_box(st, i):
    """Slice of component i's bounding box."""
    x, y, w, h = (int(v) for v in st[i][:4])
    return slice(y, y + h), slice(x, x + w)


def _stroke_width(mask):
    """Average stroke width of a shape, in pixels (2 x area / outline length)."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return 2 * float(m.sum()) / max(1.0, sum(cv2.arcLength(c, True) for c in contours))


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


DOT_SIZE = 0.25    # a piece smaller than this (x the letter body's area) is a dot. Cartoon dots are big
                   # (a ring-shaped test dot measured 0.16); pieces of a broken letter are 0.3+ (KEEP)


def _fill_dots(bw):
    """Fill the hole in hollow dots: a shiny cartoon dot (white highlight in it), or a dot written as
    a small circle. The model knows dots as solid blobs; a ring-shaped one was read as a small letter."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < 3:
        return bw
    body = int(st[1:, 4].max())
    out = bw
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= DOT_SIZE * body:
            continue
        piece = cv2.copyMakeBorder((lab[y:y + h, x:x + w] == i).astype(np.uint8), 1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT, value=0)
        cv2.floodFill(piece, None, (0, 0), 2)
        hole = piece[1:-1, 1:-1] == 0
        if hole.any():
            out = out.copy() if out is bw else out
            out[y:y + h, x:x + w][hole] = 255
    return out


def preprocess(img_bgr: np.ndarray, strict: bool = False, captions: bool = False, info: dict = None,
               colour: bool = False) -> np.ndarray:
    """BGR/gray image -> float32 SIZE x SIZE array, white letter on black, [0,1].
    If a dict is passed as info, info["box"] = (x, y, w, h) of the letter in the input image.
    colour=True: coloured parts count as ink too (a pastel or cartoon letter, see _colour_gray)."""
    bw = _prepare_bw(img_bgr, colour)
    bw = _keep_main_letter(bw, KEEP_STRICT if (strict or captions) else KEEP)
    if captions:                                   # strict crop, plus any caption text it still left
        bw = _drop_captions(bw)
    bw = _fill_dots(bw)
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


class PersianCNN(nn.Module if nn else object):    # (without PyTorch it only exists by name)
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