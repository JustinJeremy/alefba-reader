"""Make the printed training images.

  data/printed/<class>/<font>_<n>.png   the 32 letters in many fonts
  data/negative/32/<kind>_<n>.png       things that are NOT a Persian letter (class 32)

Fonts: every .ttf/.otf in fonts/ plus the plain Arabic-script fonts that ship with Windows
(Arial, Tahoma, Times New Roman, Segoe UI, ...). Windows fonts are read in place, never copied.
Each sample is made to look like a phone photo: random size, stroke weight, tilt, perspective,
uneven lighting, blur, noise and JPEG compression.
"""
import glob, os, random, shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import LETTERS, preprocess

SAMPLES_PER_FONT = 12
OUT = os.path.join("data", "printed")
NEG_OUT = os.path.join("data", "negative", "32")
NOT_A_LETTER = 32

# plain text fonts only (outline / shadow / decorative Windows fonts would teach the wrong shapes)
WINDOWS_FONTS = ["arial", "arialbd", "tahoma", "tahomabd", "times", "timesbd", "cour", "courbd",
                 "micross", "segoeui", "segoeuib", "segoeuil", "segoeuisl", "seguisb",
                 "calibri", "calibrib", "calibril", "DUBAI-REGULAR", "DUBAI-BOLD", "DUBAI-LIGHT",
                 "DUBAI-MEDIUM", "DTNASKH0", "DTNASKH1"]
# (DTNASKH2/3 are left out: they draw several letters in connected or swash forms)

# Arabic code points that are the same letter: text typed on an Arabic keyboard uses these
VARIANTS = {LETTERS.index("\u06a9"): ["\u0643"],             # kaf: Arabic kaf
            LETTERS.index("\u06cc"): ["\u064a", "\u0649"]}   # ye: Arabic yeh, alef maksura

# Latin characters used as "not a letter". Left out on purpose because they look like a Persian
# letter: l I 1 i j | ! (alef), o O 0 (he), c C (eyn), u U v V n (nun / be), r (re), 9 g q (vav, qaf),
# 2 3 5 7 S s z Z (various), and punctuation that looks like dots.
NOT_PERSIAN_CHARS = "ABDEFGHKMNPQRTWXYabdefhkmptwxy468#@&%$?+=*"


def find_fonts():
    fonts = glob.glob("fonts/*.ttf") + glob.glob("fonts/*.otf")
    windir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for name in WINDOWS_FONTS:
        for ext in (".ttf", ".TTF"):
            p = os.path.join(windir, name + ext)
            if os.path.isfile(p):
                fonts.append(p)
                break
    return sorted(set(fonts))


def font_tag(path):
    return os.path.splitext(os.path.basename(path))[0]


def _ink(font, ch):
    im = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(im).text((80, 80), ch, font=font, fill=255, anchor="mm")
    return np.asarray(im)


def has_glyph(font, ch):
    """False when the font draws its 'missing glyph' box (or nothing) for this character."""
    a = _ink(font, ch)
    return a.any() and not np.array_equal(a, _ink(font, "\ue000"))


def text_ink(font_path, ch, rng):
    """A character as an ink mask (1 = ink) on a square canvas."""
    size = rng.randint(50, 140)
    font = ImageFont.truetype(font_path, size)
    W = size * 3
    img = Image.new("L", (W, W), 0)
    stroke = rng.choice([0, 0, 0, 1, 2]) * size // 60                 # thicker strokes / bold print
    ImageDraw.Draw(img).text((W / 2, W / 2), ch, font=font, fill=255, anchor="mm",
                             stroke_width=stroke, stroke_fill=255)
    ink = np.asarray(img).astype(np.float32) / 255
    if rng.random() < 0.2:                                             # thinner strokes
        ink = cv2.erode(ink, np.ones((3, 3), np.uint8))
    return ink


def photo_like(ink, rng):
    """Ink mask -> grey image that looks like a phone photo of it printed on paper."""
    W = ink.shape[0]
    # tilt + perspective (a phone held not quite straight)
    j = W * 0.08
    src = np.float32([[0, 0], [W, 0], [W, W], [0, W]])
    dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    R = cv2.getRotationMatrix2D((W / 2, W / 2), rng.uniform(-8, 8), 1.0)
    M = np.vstack([R, [0, 0, 1]]) @ M
    ink = cv2.warpPerspective(ink, M, (W, W))

    # paper with uneven light, dark ink
    paper = rng.uniform(150, 255)
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, W))
    shade = 1 - rng.uniform(0, 0.35) * np.clip(gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1), 0, None)
    bg = paper * shade
    ink_level = rng.uniform(0, min(90, paper - 60))
    g = bg * (1 - ink) + ink_level * ink

    if rng.random() < 0.5:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 2.0))
    g = g + rng.normal(0, rng.uniform(0, 12), g.shape)
    if rng.random() < 0.3:                                             # low-resolution camera
        s = rng.uniform(0.25, 0.6)
        g = cv2.resize(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (W, W))
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.05:                                            # light letter on dark background
        g = 255 - g
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def render(font_path, ch, rng):
    return photo_like(text_ink(font_path, ch, rng), rng)


# ---------- things that are not a Persian letter ----------

def shape_ink(rng, W=300, combo=True):
    """A random non-letter shape as an ink mask: tiles, boxes, blobs (object / face silhouettes),
    stars, grids or scribbles. Nothing here should look like one of the 32 letters."""
    m = np.zeros((W, W), np.uint8)
    kind = rng.choice(["tile", "box", "blob", "blob", "star", "grid", "scribble"] + (["combo"] if combo else []))
    c, r = W // 2, int(W * rng.uniform(0.18, 0.32))
    if kind == "tile":                                   # filled (rounded) squares, sometimes split up
        for _ in range(rng.randint(1, 4)):
            x, y = rng.randint(40, W - 120, 2)
            w, h = rng.randint(50, 130, 2)
            cv2.rectangle(m, (int(x), int(y)), (int(x + w), int(y + h)), 255, -1)
        if rng.random() < 0.5:                           # cut lines through it, like game tiles / panels
            for _ in range(rng.randint(1, 4)):
                p = int(rng.randint(40, W - 40))
                cv2.line(m, (p, 0), (p, W), 0, int(rng.randint(3, 9))) if rng.random() < .5 else \
                    cv2.line(m, (0, p), (W, p), 0, int(rng.randint(3, 9)))
    elif kind == "box":                                  # outlined rectangle or several of them
        for _ in range(rng.randint(1, 3)):
            x, y = rng.randint(30, W - 150, 2)
            w, h = rng.randint(70, 150, 2)
            cv2.rectangle(m, (int(x), int(y)), (int(x + w), int(y + h)), 255, int(rng.randint(4, 16)))
    elif kind == "blob":                                 # smooth random silhouette (object, face, icon)
        noise = cv2.GaussianBlur(rng.rand(W, W).astype(np.float32), (0, 0), rng.uniform(14, 35))
        yy, xx = np.mgrid[:W, :W]
        centre = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2 * (W * rng.uniform(.18, .3)) ** 2)))
        field = noise / noise.max() + 0.5 * centre      # mostly noise: irregular, not a disc
        m = (field > np.percentile(field, rng.uniform(70, 88))).astype(np.uint8) * 255
        # (no holes and no rings: a loop looks like the letter he)
    elif kind == "star":
        n = int(rng.randint(5, 9))
        a = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) + rng.uniform(0, 1)
        rad = np.where(np.arange(2 * n) % 2 == 0, r, r * rng.uniform(.35, .6))
        pts = np.stack([c + rad * np.cos(a), c + rad * np.sin(a)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255) if rng.random() < .6 else cv2.polylines(m, [pts], True, 255, int(rng.randint(5, 12)))
    elif kind == "grid":
        n, step = int(rng.randint(2, 5)), int(rng.randint(35, 60))
        x0 = c - n * step // 2
        for i in range(n + 1):
            cv2.line(m, (x0 + i * step, x0), (x0 + i * step, x0 + n * step), 255, int(rng.randint(3, 8)))
            cv2.line(m, (x0, x0 + i * step), (x0 + n * step, x0 + i * step), 255, int(rng.randint(3, 8)))
    elif kind == "scribble":                              # tangled pen lines
        pts = (c + rng.randn(int(rng.randint(8, 16)), 2) * r).astype(np.int32)
        cv2.polylines(m, [pts], False, 255, int(rng.randint(4, 12)))
    else:                                                 # two different shapes together
        return np.maximum(shape_ink(rng, W, combo=False), shape_ink(rng, W, combo=False))
    # keep big filled shapes small: covering much of the photo, the clean-up keeps only their
    # outline, and an outline is a loop (which looks like the letter he)
    frac = (m > 0).mean()
    if frac > 0.12:
        k = np.sqrt(0.12 / frac)
        small = cv2.resize(m, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        m = np.zeros((W, W), np.uint8)
        o = (W - small.shape[0]) // 2
        m[o:o + small.shape[0], o:o + small.shape[1]] = small
    return m.astype(np.float32) / 255


def _is_loop(img):
    """True when the cleaned image is one shape around a big hole (like the letter he)."""
    try:
        v = (preprocess(img) > 0.5).astype(np.uint8)
    except ValueError:
        return True                                      # blank after clean-up: not useful either
    n, _, st, _ = cv2.connectedComponentsWithStats(v, connectivity=8)
    holes, _, hst, _ = cv2.connectedComponentsWithStats(1 - v, connectivity=4)
    inner = [hst[i][4] for i in range(1, holes) if hst[i][0] > 0 and hst[i][1] > 0
             and hst[i][0] + hst[i][2] < v.shape[1] and hst[i][1] + hst[i][3] < v.shape[0]]
    return n - 1 <= 2 and bool(inner) and max(inner) > 0.04 * v.size


def make_negatives(fonts, rng, n_shapes=2400, per_char=2):
    """data/negative/32/: Latin letters / digits / symbols in every font, and random shapes."""
    shutil.rmtree(NEG_OUT, ignore_errors=True)
    os.makedirs(NEG_OUT)
    n = 0
    for f in fonts:
        probe = ImageFont.truetype(f, 60)
        for ch in NOT_PERSIAN_CHARS:
            if not has_glyph(probe, ch):
                continue
            for _ in range(per_char):
                r = np.random.RandomState(rng.randrange(2**31))
                cv2.imwrite(os.path.join(NEG_OUT, f"char_{n}.png"), render(f, ch, r)); n += 1
    made = dropped = 0
    while made < n_shapes:
        r = np.random.RandomState(rng.randrange(2**31))
        img = photo_like(shape_ink(r), r)
        if _is_loop(img):                                # never teach "a loop is not a letter"
            dropped += 1
            continue
        cv2.imwrite(os.path.join(NEG_OUT, f"shape_{made}.png"), img); made += 1
    print(f"not-a-letter images: {n} characters + {n_shapes} shapes ({dropped} loop-like shapes dropped)")


def main():
    fonts = find_fonts()
    if not fonts:
        raise SystemExit("No fonts found: put Persian .ttf/.otf fonts in fonts/")
    shutil.rmtree(OUT, ignore_errors=True)
    rng = random.Random(0)
    np.random.seed(0)
    skipped = []
    for idx, ch in enumerate(LETTERS):
        os.makedirs(os.path.join(OUT, str(idx)), exist_ok=True)
        for f in fonts:
            probe = ImageFont.truetype(f, 60)
            for v, c in enumerate([ch] + VARIANTS.get(idx, [])):
                if not has_glyph(probe, c):
                    skipped.append(f"{font_tag(f)}:{c}")
                    continue
                n = SAMPLES_PER_FONT if v == 0 else SAMPLES_PER_FONT // 2
                for k in range(n):
                    rng_np = np.random.RandomState(rng.randrange(2**31))
                    img = render(f, c, rng_np)
                    cv2.imwrite(os.path.join(OUT, str(idx), f"{font_tag(f)}_{v}_{k}.png"), img)
    print(f"done: {len(fonts)} fonts:", ", ".join(font_tag(f) for f in fonts))
    if skipped:
        print(f"{len(skipped)} font/letter pairs skipped (glyph missing):", ", ".join(skipped[:20]))
    make_negatives(fonts, random.Random(1))


if __name__ == "__main__":
    main()
