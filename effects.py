"""Photo effects that make a clean letter image harder to read, for training and for the hard test set.

  ruled_lines(gray, rng)   notebook lines drawn across the paper, through the letter
  film_grain(gray, rng)    clumpy camera / film noise
  blackout(gray, rng)      light letter on black that fades into the dark (like "BLACKOUT" title text),
                           with scan-line stripes in the faded part, but still readable

Every function takes a grey image (dark letter on light paper, uint8) and returns a new grey image.
"""
import cv2
import numpy as np


def _ink(gray):
    """Ink amount per pixel (0 = paper, 1 = darkest ink)."""
    paper, dark = np.percentile(gray, 90), np.percentile(gray, 1)
    return np.clip((paper - gray.astype(np.float32)) / max(paper - dark, 1), 0, 1)


def _with_margin(gray, rng, k=(0.35, 0.7)):
    """The letter redrawn on a fresh, bigger sheet of paper (photo size, room around it for lines).
    Redrawing (instead of padding) leaves no visible box where the original image ended."""
    ink = _ink(gray)
    ink[ink < 0.15] = 0                                                 # the original paper's own texture
    pts = cv2.findNonZero((ink > 0).astype(np.uint8))
    if pts is not None:                                                 # just the letter, not its old canvas
        x, y, w, h = cv2.boundingRect(pts)
        ink = ink[y:y + h, x:x + w]
    s = rng.uniform(150, 350) / max(ink.shape)                          # letter size in a phone photo
    ink = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC).clip(0, 1)
    m = int(max(ink.shape) * rng.uniform(*k))
    ink = cv2.copyMakeBorder(ink, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    paper, pen = rng.uniform(180, 250), rng.uniform(10, 70)
    return (paper - (paper - pen) * ink).astype(np.uint8)


def ruled_lines(gray, rng):
    g = _with_margin(gray, rng).astype(np.float32)
    h, w = g.shape
    ink = _ink(g.astype(np.uint8))
    ys = np.nonzero(ink.max(1) > 0.5)[0]
    top, bottom = (ys.min(), ys.max()) if len(ys) else (h // 3, 2 * h // 3)
    paper = np.percentile(g, 90)
    gap = rng.uniform(0.25, 0.6) * (bottom - top + 1) + 6               # notebook line spacing
    start = rng.uniform(top, top + gap)                                 # at least one line crosses the letter
    first = start - gap * int(start / gap)
    tone = paper - (paper - np.percentile(g, 1)) * rng.uniform(0.25, 0.9)   # faint blue print .. as dark as pen
    thick = int(rng.choice([1, 1, 2, 2, 3]) * max(1, h // 200))
    slope, bend = rng.uniform(-0.03, 0.03), rng.uniform(-0.04, 0.04)
    x = np.arange(w)
    lines = np.zeros((h, w), np.uint8)
    for y0 in np.arange(first, h, gap):
        y = y0 + slope * (x - w / 2) + bend * h * ((x / w - 0.5) ** 2)  # slightly slanted / curved paper
        pts = np.stack([x, y], 1).round().astype(np.int32)
        cv2.polylines(lines, [pts], False, 255, thick, cv2.LINE_AA)
    a = lines.astype(np.float32) / 255
    g = np.minimum(g, g * (1 - a) + tone * a)                           # ink stays as dark as it was
    return np.clip(g, 0, 255).astype(np.uint8)


def film_grain(gray, rng):
    g = _with_margin(gray, rng, (0.15, 0.4)).astype(np.float32)
    sigma = rng.uniform(10, 35)
    grain = rng.normal(0, 1, g.shape).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), rng.uniform(0.6, 1.6))      # clumps of grain, not single pixels
    grain *= sigma / max(grain.std(), 1e-6)
    lum = 0.6 + 0.4 * (g / 255)                                         # film grain shows most in light parts
    g = g + grain * lum
    if rng.random() < 0.4:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.0))
    return np.clip(g, 0, 255).astype(np.uint8)


def blackout(gray, rng):
    g = _with_margin(gray, rng, (0.2, 0.45))
    ink = _ink(g)
    h, w = g.shape
    bg = rng.uniform(5, 40)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)                                     # fade in any direction
    t = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5)) + 0.5
    t = np.clip((t - t.min()) / max(np.ptp(t), 1e-6), 0, 1)
    lo, hi = rng.uniform(70, 120), rng.uniform(200, 255)                # faded end stays visible (bg + 40 or more)
    bright = lo + (hi - lo) * t ** rng.uniform(0.6, 1.6)
    if rng.random() < 0.6:                                              # scan-line stripes in the faded part
        period = int(rng.integers(3, 7))
        stripes = ((yy.astype(int) % period) < period // 2).astype(np.float32)
        bright *= 1 - (1 - t) * 0.5 * stripes
    out = bg + (bright - bg) * ink
    out += rng.normal(0, rng.uniform(2, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


EFFECTS = {"ruled_lines": ruled_lines, "film_grain": film_grain, "blackout": blackout}


if __name__ == "__main__":
    # python effects.py  ->  hard_set/ruled_lines, hard_set/film_grain, hard_set/blackout:
    # the 32 chart letters (rehearsal_set/q95, TEST ONLY) with each effect, same file names
    import glob, os
    for name, effect in EFFECTS.items():
        out = os.path.join("hard_set", name)
        os.makedirs(out, exist_ok=True)
        for i, path in enumerate(sorted(glob.glob(os.path.join("rehearsal_set", "q95", "*.jpg")))):
            g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = effect(g, np.random.default_rng(1000 + i))         # fixed seeds: the same set every time
            cv2.imwrite(os.path.join(out, os.path.basename(path)), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(out)
