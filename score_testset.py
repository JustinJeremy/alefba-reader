"""Score the model on a test set (TEST ONLY, never train on it).
Usage:  python score_testset.py [folder]     e.g. rehearsal_set, hard_set or real_photos
Every sub-folder (q95, caption, JJ, Ethan, ...) is scored separately.
The right answer comes from <folder>/labels.csv if there is one, otherwise from the file name:
'01_be_desk.jpg' -> class 1 (the number is the class id, see real_photos/README.txt).
"""
import csv, glob, os, re, sys, time
from common import LETTERS, LOOKUP
from predict import predict_file, utf8_console

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
        m = re.match(r"(\d{1,2})_", name)
        if m and 0 <= int(m.group(1)) < 32:
            labels[name] = int(m.group(1))
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
            try:
                (pred, conf), *_ = predict_file(os.path.join(q, name))
            except ValueError as e:                      # no letter found / unreadable photo
                wrong.append(f"  {name}: {LETTERS[truth]} ({LOOKUP[truth]}) -> ERROR {e}")
                continue
            ok = LOOKUP[pred] == LOOKUP[truth]          # output text; every letter has its own spelling now
            right += ok
            if not ok:
                wrong.append(f"  {name}: {LETTERS[truth]} ({LOOKUP[truth]}) -> got {LETTERS[pred]} ({LOOKUP[pred]}) conf={conf:.2f}")
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
