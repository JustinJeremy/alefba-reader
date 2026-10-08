"""Train. Expects data/<source>/<class 0..31>/*.png (handwritten from IFHCDB, printed from prepare_data.py)."""
import glob, os, random
import cv2, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.transforms.v2 as T
from common import PersianCNN, preprocess

torch.manual_seed(0); random.seed(0); np.random.seed(0)
X, y, src = [], [], []
for source in ("handwritten", "printed"):
    for c in range(32):
        for p in glob.glob(f"data/{source}/{c}/*"):
            try:
                X.append(preprocess(cv2.imread(p, cv2.IMREAD_UNCHANGED)))
                y.append(c); src.append(source)
            except Exception:
                pass
X, y, src = np.array(X), np.array(y), np.array(src)
print("images:", len(X), "| classes found:", len(set(y)))
if len(set(y)) != 32:
    raise SystemExit("Need images for all 32 classes.")

idx = np.arange(len(X))
tr, rest = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=0)

aug = T.Compose([T.RandomAffine(8, translate=(.1, .1), scale=(.9, 1.1))])
def batch(ids, train):
    t = torch.tensor(X[ids]).unsqueeze(1)
    if train:
        t = torch.stack([aug(i) for i in t])
        t = t * (0.8 + 0.4 * torch.rand(len(t), 1, 1, 1))       # contrast jitter
        t = (t + 0.03 * torch.randn_like(t)).clamp(0, 1)         # noise
    return t, torch.tensor(y[ids])

counts = np.bincount(y[tr], minlength=32)
w = torch.tensor(len(tr) / (32 * np.maximum(counts, 1)), dtype=torch.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = PersianCNN().to(dev)
loss_fn = nn.CrossEntropyLoss(weight=w.to(dev))
EPOCHS = 30
opt = torch.optim.Adam(model.parameters(), 1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

def evaluate(ids):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 512):
            xb, _ = batch(ids[i:i + 512], False)
            preds += model(xb.to(dev)).argmax(1).cpu().tolist()
    return np.array(preds)

best = 0
for ep in range(EPOCHS):
    model.train(); perm = np.random.permutation(tr)
    for i in range(0, len(perm), 128):
        xb, yb = batch(perm[i:i + 128], True)
        opt.zero_grad(); loss = loss_fn(model(xb.to(dev)), yb.to(dev))
        loss.backward(); opt.step()
    sched.step()
    acc = accuracy_score(y[va], evaluate(va))
    print(f"epoch {ep+1}/{EPOCHS} val acc {acc:.4f}")
    if acc > best:
        best = acc; torch.save(model.state_dict(), "persian_cnn.pt")

model.load_state_dict(torch.load("persian_cnn.pt", map_location=dev))
p = evaluate(te)
print("TEST accuracy:", accuracy_score(y[te], p))
for s in ("handwritten", "printed"):
    m = src[te] == s
    if m.any(): print(f"  {s}:", accuracy_score(y[te][m], p[m]))
np.savetxt("confusion_matrix.csv", confusion_matrix(y[te], p), fmt="%d", delimiter=",")
