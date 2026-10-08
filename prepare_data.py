"""Render the 32 letters in every font found in fonts/ -> data/printed/<class>/.
Put .ttf/.otf Persian fonts (Vazirmatn, Noto Naskh Arabic, Amiri, ...) in fonts/."""
import glob, os, random
from PIL import Image, ImageDraw, ImageFont
from common import LETTERS


def main():
    fonts = glob.glob("fonts/*.ttf") + glob.glob("fonts/*.otf")
    if not fonts:
        raise SystemExit("Put at least one Persian .ttf/.otf font in fonts/")
    random.seed(0)
    for idx, ch in enumerate(LETTERS):
        os.makedirs(f"data/printed/{idx}", exist_ok=True)
        n = 0
        for f in fonts:
            for size in (60, 90, 120):
                font = ImageFont.truetype(f, size)
                for _ in range(4):
                    img = Image.new("L", (size * 3, size * 3), random.randint(200, 255))
                    d = ImageDraw.Draw(img)
                    d.text((size * 1.5, size * 1.5), ch, font=font, anchor="mm",
                           fill=random.randint(0, 60))
                    img = img.rotate(random.uniform(-6, 6), fillcolor=255)
                    img.save(f"data/printed/{idx}/{n}.png"); n += 1
    print("done:", len(fonts), "fonts")


if __name__ == "__main__":
    main()
