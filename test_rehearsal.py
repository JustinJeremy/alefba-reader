"""Score the model on a test set (TEST ONLY, never train on it).
Usage:  python test_rehearsal.py [folder]     e.g. rehearsal_set  or  hard_set
Every sub-folder (q95, caption, border, ...) is scored separately.
"""
import csv, glob, os, sys
from common import LETTERS, LOOKUP
from predict import predict_file, utf8_console

utf8_console()

root = sys.argv[1] if len(sys.argv) > 1 else "rehearsal_set"
labels = {r["file"]: int(r["class_id"]) for r in csv.DictReader(open(os.path.join(root, "labels.csv")))}

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
