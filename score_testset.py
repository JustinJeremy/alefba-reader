"""Score the model on a test set (TEST ONLY, never train on it).
Usage:  python score_testset.py [folder]     e.g. rehearsal_set  or  hard_set
Every sub-folder (q95, caption, border, ...) is scored separately.
"""
import csv, glob, os, sys, time
from common import LETTERS, LOOKUP
from predict import predict_file, utf8_console


def main(root):
    labels = {r["file"]: int(r["class_id"]) for r in csv.DictReader(open(os.path.join(root, "labels.csv")))}
    total = correct = 0
    t0 = time.perf_counter()
    for q in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
        right, wrong = 0, []
        for name, truth in sorted(labels.items()):
            (pred, conf), *_ = predict_file(os.path.join(q, name))
            ok = LOOKUP[pred] == LOOKUP[truth]          # output text; every letter has its own spelling now
            right += ok
            if not ok:
                wrong.append(f"  {LETTERS[truth]} ({LOOKUP[truth]}) -> got {LETTERS[pred]} ({LOOKUP[pred]}) conf={conf:.2f}")
        print(f"{os.path.basename(q)}: {right}/{len(labels)} correct")
        print("\n".join(wrong) if wrong else "  no mistakes")
        total += len(labels); correct += right
    if total:
        ms = (time.perf_counter() - t0) * 1000 / total
        print(f"TOTAL: {correct}/{total} correct ({100 * correct / total:.1f}%), "
              f"{ms:.0f} ms per image incl. loading + preprocessing")


if __name__ == "__main__":
    utf8_console()
    main(sys.argv[1] if len(sys.argv) > 1 else "rehearsal_set")
