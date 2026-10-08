"""Render the 32 letters in many fonts -> data/printed/<class>/<font>_<n>.png

Fonts: every .ttf/.otf in fonts/ plus the plain Arabic-script fonts that ship with Windows
(Arial, Tahoma, Times New Roman, Segoe UI, ...). Windows fonts are read in place, never copied.
Each sample is made to look like a phone photo of a printed letter: random size, stroke
weight, tilt, perspective, uneven lighting, blur, noise and JPEG compression.
"""
import glob, os, random, shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import LETTERS

SAMPLES_PER_FONT = 12
OUT = os.path.join("data", "printed")

# plain text fonts only (outline / shadow / decorative Windows fonts would teach the wrong shapes)
WINDOWS_FONTS = ["arial", "arialbd", "tahoma", "tahomabd", "times", "timesbd", "cour", "courbd",
                 "micross", "segoeui", "segoeuib", "segoeuil", "segoeuisl", "seguisb",
                 "calibri", "calibrib", "calibril", "DUBAI-REGULAR", "DUBAI-BOLD", "DUBAI-LIGHT",
                 "DUBAI-MEDIUM", "DTNASKH0", "DTNASKH1"]
# (DTNASKH2/3 are left out: they draw several letters in connected or swash forms)

# Arabic code points that are the same letter: text typed on an Arabic keyboard uses these
VARIANTS = {LETTERS.index("\u06a9"): ["\u0643"],             # kaf: Arabic kaf
            LETTERS.index("\u06cc"): ["\u064a", "\u0649"]}   # ye: Arabic yeh, alef maksura


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


def render(font_path, ch, rng):
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


if __name__ == "__main__":
    main()
