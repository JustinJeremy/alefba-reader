"""Score the model on a test set (TEST ONLY, never train on it).
Usage:  python score_testset.py [folder]     e.g. rehearsal_set, hard_set or real_photos
Every sub-folder (q95, caption, JJ, Ethan, ...) is scored separately.
The right answer comes from <folder>/labels.csv if there is one, otherwise from the file name:
'01_be_desk.jpg' -> class 1 (the number is the class id, see real_photos/README.txt).
Photos of several letters (multi_set): '07-08-09_row1.jpg' -> letters 7, 8, 9 in reading order;
the photo counts as correct only when every letter is right and in the right order.
"""
import csv, glob, os, re, sys, time
from common import LETTERS, LOOKUP
from common import load_image
from predict import read_photo, utf8_console

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def labelled_files(root, folder):
    """-> ({file name: class id}, [files whose name gives no class])"""
    csv_path = os.path.join(root, "labels.csv")
    if os.path.isfile(csv_path):
        return {r["file"]: int(r["class_id"]) for r in csv.DictReader(open(csv_path, encoding="utf-8"))}, []
    labels, unnamed = {}, []
    for name in os.listdir(folder):
        if not name.lower().endswith(IMAGE_EXT):
            continue
        m = re.match(r"(\d{1,2}(?:-\d{1,2})*)_", name)
        nums = [int(v) for v in m.group(1).split("-")] if m else []
        if nums and all(0 <= v < 32 for v in nums):
            labels[name] = nums[0] if len(nums) == 1 else nums
        else:
            unnamed.append(name)
    return labels, unnamed


def main(root):
    total = correct = 0
    t0 = time.perf_counter()
    folders = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)) or [root]
    for q in folders:
        labels, unnamed = labelled_files(root, q)
        if not labels:
            print(f"{os.path.basename(q)}: (no labelled photos yet)")
            continue
        right, wrong = 0, []
        for name, truth in sorted(labels.items()):
            want = truth if isinstance(truth, list) else [truth]
            show = lambda cs: " ".join(LETTERS[c] for c in cs) + " (" + " ".join(LOOKUP[c] for c in cs) + ")"
            try:
                img = load_image(os.path.join(q, name))
                if img is None:
                    raise ValueError("unreadable image")
                letters, _ = read_photo(img)
            except ValueError as e:                      # no letter found / unreadable photo
                wrong.append(f"  {name}: {show(want)} -> ERROR {e}")
                continue
            got = [res[0][0] for res, _ in letters]          # a one-letter photo read as two letters is wrong
            ok = [LOOKUP[c] for c in got] == [LOOKUP[c] for c in want]   # the output text
            right += ok
            if not ok:
                conf = " ".join(f"{res[0][1]:.2f}" for res, _ in letters)
                wrong.append(f"  {name}: {show(want)} -> got {show(got)} conf={conf}")
        print(f"{os.path.basename(q)}: {right}/{len(labels)} correct")
        print("\n".join(wrong) if wrong else "  no mistakes")
        if unnamed:
            print(f"  skipped {len(unnamed)} file(s) without a 'NN_' class number in the name, e.g. {unnamed[0]}")
        total += len(labels); correct += right
    if total:
        ms = (time.perf_counter() - t0) * 1000 / total
        print(f"TOTAL: {correct}/{total} correct ({100 * correct / total:.1f}%), "
              f"{ms:.0f} ms per image incl. loading + preprocessing")


if __name__ == "__main__":
    utf8_console()
    main(sys.argv[1] if len(sys.argv) > 1 else "rehearsal_set")
