"""Store what each letter 'looks like' inside the model, so photos without a letter can be refused.

For every training image the model's 256-number feature vector (the layer before the answer) is
taken; the average per letter is saved to letter_profile.npz. predict.py measures how far a photo's
features are from the average of the letter it was read as: real letters are close, noise and
empty photos are far. Run after training (train.py does it automatically):  python build_profile.py
"""
import numpy as np
import torch
from predict import load_model, MODEL_PATH
import os

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letter_profile.npz")


def features(model, X, batch=512):
    """Cleaned SIZE x SIZE images -> unit-length feature vectors."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            f = model.head[:3](model.features(torch.tensor(X[i:i + batch]).unsqueeze(1)))
            out.append(torch.nn.functional.normalize(f, dim=1).numpy())
    return np.concatenate(out)


def build(model, X, y):
    X, y = X[y < 32], y[y < 32]                       # the 32 letters only, not the "not a letter" images
    F = features(model, X)
    centres = np.stack([F[y == c].mean(0) for c in range(32)])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    dist = 1 - (F * centres[y]).sum(1)
    np.savez(PROFILE_PATH, centres=centres.astype(np.float32))
    print(f"letter profile saved: distance of training letters to their own average "
          f"median {np.median(dist):.3f}, 99th percentile {np.percentile(dist, 99):.3f}, max {dist.max():.3f}")


def main():
    from train import load_data
    X, y, _, _ = load_data()
    build(load_model(MODEL_PATH), X, y)


if __name__ == "__main__":
    main()
