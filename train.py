"""Train. Expects data/<source>/<class 0..31>/*.png (handwritten from convert_dataset.py; printed, negative and
effects from prepare_data.py) and the team's own letters in member_handwriting/<member>/<NN>_<name>_*.jpg.
Writes persian_cnn.pt, letter_profile.npz, results/metrics.txt and results/confusion_matrix.csv."""
import glob, os, re, random, time
from multiprocessing import Pool
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torchvision.transforms.v2 as T
from common import PersianCNN, preprocess, load_image, NAMES, SIZE, NOT_A_LETTER, NUM_CLASSES

EPOCHS = 40        # (30 before batch 16: turned letters take longer to learn)
BATCH = 128
MODEL_PATH = "persian_cnn.pt"
RESULTS = "results"
# one whole font family is never trained on: its score shows how well the model reads an unseen font
HELDOUT_FONTS = ("calibri", "calibrib", "calibril")
# the team's handwriting (cut from the clear alphabet sheets): only ~125 photos, so each one is shown
# MEMBER_REPEAT times per epoch (with the usual random changes). Training only: the team's earlier sheets
# in real_photos/ are the test for these writers.
MEMBERS = "member_handwriting"
MEMBER_REPEAT = 15

# letters at any angle (batch 16): this share of the training letters is turned to a random angle (0-360),
# the rest stay upright. The best epoch is still chosen on upright letters (the validation set is not
# turned), so upright reading stays first. Note: some letters turned around ARE other letters
# (upside-down pe looks like se, be like nun), so those pairs stay uncertain when upside down.
TURN_SHARE = 0.5

aug = T.Compose([T.RandomAffine(8, translate=(.1, .1), scale=(.9, 1.1))])
blur = T.GaussianBlur(3, sigma=(0.1, 1.0))


def _prep(path):
    try:
        return preprocess(load_image(path)), None
    except (ValueError, cv2.error) as e:
        return None, f"{path}: {e}"


def font_of(path):
    """data/printed/<c>/<font>_<variant>_<n>.png -> font name ('' for handwritten)."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.rsplit("_", 2)[0] if os.sep + "printed" + os.sep in os.path.normpath(path) else ""


def load_workers():
    """How many processes prepare the images. Each one loads PyTorch; the GPU (CUDA) build is so big
    that one per core (16) runs out of memory on a 16 GB laptop, so it uses at most 6."""
    n = os.cpu_count() or 1
    return min(n, 6) if torch.version.cuda else n


def keep_awake():
    """Windows: don't go to sleep while training runs (until this program ends; closing the lid still can)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | SYSTEM_REQUIRED


def load_data():
    """Preprocess every training image (in parallel). Unreadable / blank files are skipped but reported."""
    items = [(p, c, source) for source in ("handwritten", "printed")
             for c in range(32) for p in sorted(glob.glob(f"data/{source}/{c}/*"))]
    items += [(p, NOT_A_LETTER, "negative") for p in sorted(glob.glob(f"data/negative/{NOT_A_LETTER}/*"))]
    items += [(p, c, "effects") for c in range(NUM_CLASSES) for p in sorted(glob.glob(f"data/effects/{c}/*"))]
    for p in sorted(glob.glob(os.path.join(MEMBERS, "*", "*.jpg"))):
        m = re.match(r"(\d{2})_", os.path.basename(p))
        if m and int(m.group(1)) < 32:
            items.append((p, int(m.group(1)), "members"))
    with Pool(load_workers()) as pool:
        out = pool.map(_prep, [p for p, _, _ in items], chunksize=64)
    X, y, src, font, skipped = [], [], [], [], []
    for (p, c, source), (arr, err) in zip(items, out):
        if arr is None:
            skipped.append(err)
            continue
        X.append(arr); y.append(c); src.append(source); font.append(font_of(p))
    if skipped:
        print(f"skipped {len(skipped)} unusable images, e.g.:", *skipped[:5], sep="\n  ")
    return np.array(X), np.array(y), np.array(src), np.array(font)


def _thicken(t):
    return F.max_pool2d(t, 3, 1, 1)


def _thin(t):
    return -F.max_pool2d(-t, 3, 1, 1)


def turn(view, angle):
    """A cleaned SIZE x SIZE letter turned by angle (degrees), then framed again the way preprocess()
    frames a letter (tight crop, 1/8 margin, square, SIZE x SIZE), so it matches a turned photo."""
    big = cv2.resize(view, (SIZE * 4, SIZE * 4), interpolation=cv2.INTER_LINEAR)   # finer turning
    h, w = big.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2                         # room for the corners
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    ink = cv2.warpAffine(big, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0) > 0.5
    pts = cv2.findNonZero(ink.astype(np.uint8))
    if pts is None:
        return view
    x, y, w, h = cv2.boundingRect(pts)
    side = max(w, h)
    pad = max(2, side // 8)
    sq = np.zeros((side + 2 * pad, side + 2 * pad), np.float32)
    oy, ox = (sq.shape[0] - h) // 2, (sq.shape[1] - w) // 2
    sq[oy:oy + h, ox:ox + w] = ink[y:y + h, x:x + w]
    return cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def batch(X, y, ids, train):
    xs = X[ids]
    if train and TURN_SHARE:
        xs = xs.copy()
        for k in np.where(np.random.rand(len(xs)) < TURN_SHARE)[0]:
            xs[k] = turn(xs[k], np.random.uniform(0, 360))
    t = torch.tensor(xs).unsqueeze(1)
    if train:
        t = torch.stack([aug(i) for i in t])
        r = torch.rand(len(t))
        t = torch.where((r < 0.25)[:, None, None, None], _thicken(t), t)             # bolder pen
        t = torch.where(((r >= 0.25) & (r < 0.4))[:, None, None, None], _thin(t), t)  # finer pen
        b = torch.rand(len(t)) < 0.2
        if b.any():
            t[b] = blur(t[b])                                                        # out of focus
        t = t * (0.8 + 0.4 * torch.rand(len(t), 1, 1, 1))       # contrast jitter
        t = (t + 0.03 * torch.randn_like(t)).clamp(0, 1)         # noise
    return t, torch.tensor(y[ids])


def evaluate(model, X, y, ids, dev):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            xb, _ = batch(X, y, ids[i:i + 512], False)
            preds += model(xb.to(dev)).argmax(1).cpu().tolist()
    return np.array(preds)


def split(y, src, font):
    """Held-out font -> its own test set; the team's letters -> training only (repeated);
    everything else stratified 80/10/10 per letter and source."""
    held = np.isin(font, HELDOUT_FONTS)
    members = src == "members"
    idx = np.where(~held & ~members)[0]
    kind = lambda i: (src[i] == "printed") * 1 + (src[i] == "effects") * 2
    tr, rest = train_test_split(idx, test_size=0.2, stratify=y[idx] * 3 + kind(idx), random_state=0)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest] * 3 + kind(rest), random_state=0)
    tr = np.concatenate([tr, np.repeat(np.where(members)[0], MEMBER_REPEAT)])
    return tr, va, te, np.where(held)[0]


def report(model, X, y, src, te, held, dev, seconds):
    p = evaluate(model, X, y, te, dev)
    lines = [f"input {SIZE}x{SIZE}, {sum(q.numel() for q in model.parameters()):,} parameters, "
             f"trained {EPOCHS} epochs in {seconds / 60:.1f} min",
             f"train/val/test images: see split in train.py; test = {len(te)}, held-out font test = {len(held)}",
             "",
             f"TEST accuracy (all):        {accuracy_score(y[te], p):.4f}"]
    for s in ("handwritten", "printed", "negative", "effects"):
        m = src[te] == s
        if m.any():
            lines.append(f"TEST accuracy ({s}):{' ' * (12 - len(s))}{accuracy_score(y[te][m], p[m]):.4f}  (n={m.sum()})")
    if len(held):
        ph = evaluate(model, X, y, held, dev)
        lines.append(f"UNSEEN FONT accuracy ({', '.join(HELDOUT_FONTS)}): {accuracy_score(y[held], ph):.4f}  (n={len(held)})")
    model.eval(); x1 = torch.tensor(X[te[:1]]).unsqueeze(1).to(dev)
    with torch.no_grad():
        for _ in range(5): model(x1)
        t = time.perf_counter()
        for _ in range(100): model(x1)
    lines.append(f"model inference time: {(time.perf_counter() - t) * 10:.2f} ms per image (batch of 1, {dev})")
    lines += ["", "Per-letter results on the test set:",
              classification_report(y[te], p, labels=range(NUM_CLASSES), target_names=NAMES + ["NOT A LETTER"], digits=3)]
    letters = y[te] != NOT_A_LETTER
    lines.insert(-2, f"real letters wrongly refused as 'not a letter': {(p[letters] == NOT_A_LETTER).mean():.4f}"
                     f"  | non-letters caught: {(p[~letters] == NOT_A_LETTER).mean():.4f}")
    os.makedirs(RESULTS, exist_ok=True)
    open(os.path.join(RESULTS, "metrics.txt"), "w", encoding="utf-8").write("\n".join(lines))
    np.savetxt(os.path.join(RESULTS, "confusion_matrix.csv"), confusion_matrix(y[te], p, labels=range(NUM_CLASSES)),
               fmt="%d", delimiter=",")
    print("\n".join(lines))


def main():
    keep_awake()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count())
    t0 = time.time()
    X, y, src, font = load_data()
    print(f"images: {len(X)} | classes found: {len(set(y))} | loaded in {time.time() - t0:.0f} s")
    if len(set(y)) != NUM_CLASSES:
        raise SystemExit(f"Need images for all {NUM_CLASSES} classes (32 letters + not-a-letter: run prepare_data.py).")
    tr, va, te, held = split(y, src, font)

    counts = np.bincount(y[tr], minlength=NUM_CLASSES)
    w = torch.tensor(len(tr) / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = PersianCNN().to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best, t0 = 0, time.time()
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), BATCH):
            xb, yb = batch(X, y, perm[i:i + BATCH], True)
            opt.zero_grad(); loss = loss_fn(model(xb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
        sched.step()
        acc = accuracy_score(y[va], evaluate(model, X, y, va, dev))
        print(f"epoch {ep+1}/{EPOCHS} val acc {acc:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if acc > best:
            best = acc; torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    report(model, X, y, src, te, held, dev, time.time() - t0)
    import build_profile                                  # what each letter looks like, for refusing non-letters
    build_profile.build(model.cpu(), X, y)
    import numpy_cnn                                      # the same weights as plain arrays, for the online copy
    numpy_cnn.export_weights(MODEL_PATH)


if __name__ == "__main__":
    main()
