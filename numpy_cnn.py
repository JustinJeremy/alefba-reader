"""The letter model (PersianCNN in common.py) run with NumPy only, no PyTorch.

Used by the online copy (Render's free plan has 512 MB of memory, and PyTorch alone needs about half
of it), and on the laptop when ALEFBA_BACKEND=numpy. Same layers, same trained weights, same answers:
tests/test_numpy_cnn.py checks it against PyTorch on real photos.

The weights come from persian_cnn.pt, saved as plain arrays in persian_cnn.npz:
    python numpy_cnn.py          (needs PyTorch; train.py also does this after every training run)
"""
import os
import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persian_cnn.npz")
BN_EPS = 1e-5            # nn.BatchNorm2d's default
BLOCKS = 4               # conv 3x3 -> batch norm -> ReLU -> max pool 2x2, four times (64 -> 4 px)


def export_weights(pt_path=None, out_path=WEIGHTS_PATH):
    """Save the PyTorch model's weights as NumPy arrays (needs PyTorch)."""
    import torch
    from predict import MODEL_PATH
    state = torch.load(pt_path or MODEL_PATH, map_location="cpu")
    np.savez(out_path, **{k: v.numpy() for k, v in state.items() if not k.endswith("num_batches_tracked")})
    return out_path


class NumpyCNN:
    """forward(x): x = (N, 64, 64) float32 letters -> (256 features before the last layer, 33 scores)."""

    def __init__(self, path=WEIGHTS_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model weights not found: {path}. Run 'python numpy_cnn.py' first.")
        w = np.load(path)
        self.blocks = []
        for b in range(BLOCKS):
            conv, bn = f"features.{b}.0", f"features.{b}.1"
            scale = w[bn + ".weight"] / np.sqrt(w[bn + ".running_var"] + BN_EPS)       # batch norm, folded
            shift = w[bn + ".bias"] - w[bn + ".running_mean"] * scale
            kernel = w[conv + ".weight"].astype(np.float32)                           # (out, in, 3, 3)
            self.blocks.append((kernel.reshape(kernel.shape[0], -1).T.copy(),         # (in*9, out)
                                w[conv + ".bias"].astype(np.float32),
                                scale.astype(np.float32), shift.astype(np.float32)))
        self.fc1 = (w["head.1.weight"].T.astype(np.float32).copy(), w["head.1.bias"].astype(np.float32))
        self.fc2 = (w["head.4.weight"].T.astype(np.float32).copy(), w["head.4.bias"].astype(np.float32))

    @staticmethod
    def _conv3x3(x, kernel, bias):
        """x (N, C, H, W), padding 1 -> (N, out, H, W), as a matrix product over 3x3 patches."""
        n, c, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        patches = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # (N, C, H, W, 3, 3)
        cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * wd, c * 9)       # same order as the kernel
        out = cols @ kernel + bias
        return out.reshape(n, h, wd, -1).transpose(0, 3, 1, 2)

    def forward(self, x):
        x = np.asarray(x, np.float32)[:, None]                                       # (N, 1, 64, 64)
        for kernel, bias, scale, shift in self.blocks:
            x = self._conv3x3(x, kernel, bias)
            x = np.maximum(x * scale[None, :, None, None] + shift[None, :, None, None], 0)
            n, c, h, wd = x.shape
            x = x.reshape(n, c, h // 2, 2, wd // 2, 2).max(axis=(3, 5))              # 2x2 max pool
        f = np.maximum(x.reshape(x.shape[0], -1) @ self.fc1[0] + self.fc1[1], 0)     # flatten, dense 256, ReLU
        return f, f @ self.fc2[0] + self.fc2[1]                                      # (dropout: off when reading)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def normalize(f):
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


if __name__ == "__main__":
    print("saved", export_weights())
