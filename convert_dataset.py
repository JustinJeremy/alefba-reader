"""Convert the Yousefnezhad Persian handwriting dataset to data/handwritten/<0..31>/.
Usage: python convert_dataset.py <folder containing Random1 ... Random10>
Image 'aa (N).bmp' has class (N-1)//100, in the same order as LETTERS in common.py
(1-100 = alef, 101-200 = be, ... 3101-3200 = ye). Verified visually."""
import glob, os, re, sys
import cv2


def main(src):
    files = glob.glob(os.path.join(src, "Random*", "*.bmp"))
    if not files:
        raise SystemExit("No Random*/ *.bmp files found in: " + src)
    count, skipped = [0] * 32, []
    for p in files:
        m = re.search(r"\((\d+)\)", os.path.basename(p))
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if m else None
        c = (int(m.group(1)) - 1) // 100 if m else -1
        if img is None or not 0 <= c < 32:
            skipped.append(p)
            continue
        folder = os.path.join("data", "handwritten", str(c))
        os.makedirs(folder, exist_ok=True)
        tag = os.path.basename(os.path.dirname(p))
        cv2.imwrite(os.path.join(folder, f"{tag}_{m.group(1)}.png"), img)
        count[c] += 1
    print("images per class:", count)
    if skipped:
        print(f"skipped {len(skipped)} files (no '(N)' in the name, N out of range, or unreadable)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
