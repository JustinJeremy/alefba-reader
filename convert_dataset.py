"""Convert the Yousefnezhad Persian handwriting dataset to data/handwritten/<0..31>/.
Usage: python convert_dataset.py <folder containing Random1 ... Random10>
Image 'aa (N).bmp' has class (N-1)//100, in the same order as LETTERS in common.py
(1-100 = alef, 101-200 = be, ... 3101-3200 = ye). Verified visually."""
import glob, os, re, sys
import cv2

src = sys.argv[1] if len(sys.argv) > 1 else "."
files = glob.glob(os.path.join(src, "Random*", "*.bmp"))
if not files:
    raise SystemExit("No Random*/ *.bmp files found in: " + src)
count = [0] * 32
for p in files:
    n = int(re.search(r"\((\d+)\)", os.path.basename(p)).group(1))
    c = (n - 1) // 100
    if not 0 <= c < 32:
        continue
    folder = os.path.join("data", "handwritten", str(c))
    os.makedirs(folder, exist_ok=True)
    tag = os.path.basename(os.path.dirname(p))
    cv2.imwrite(os.path.join(folder, f"{tag}_{n}.png"), cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    count[c] += 1
print("images per class:", count)
