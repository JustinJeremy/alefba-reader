"""The NumPy copy of the model (numpy_cnn.py, used online) must give PyTorch's answers."""
import glob, os
import numpy as np
import pytest
import torch
from conftest import ROOT
from common import load_image, preprocess
from numpy_cnn import NumpyCNN, WEIGHTS_PATH, softmax
import predict


def views(n=None):
    files = sorted(glob.glob(os.path.join(ROOT, "rehearsal_set", "q95", "*.jpg")))
    files += sorted(glob.glob(os.path.join(ROOT, "real_photos", "*", "*_sheet.jpg")))[:40]
    return np.array([preprocess(load_image(f)) for f in files[:n]])


def test_weights_file_matches_the_trained_model():
    # after a retrain, train.py rewrites persian_cnn.npz; a stale copy would make the online answers differ
    state = torch.load(predict.MODEL_PATH, map_location="cpu")
    saved = np.load(WEIGHTS_PATH)
    for k, v in state.items():
        if not k.endswith("num_batches_tracked"):
            assert np.array_equal(saved[k], v.numpy()), k


def test_tilt_weights_file_matches_the_tilt_model():
    state = torch.load(predict.TILT_MODEL_PATH, map_location="cpu")
    saved = np.load(predict.TILT_WEIGHTS_PATH)
    for k, v in state.items():
        if not k.endswith("num_batches_tracked"):
            assert np.array_equal(saved[k], v.numpy()), k


def test_numpy_and_pytorch_give_the_same_scores_and_answers():
    x = views()
    f_t, out_t = predict._TorchNet(predict.load_model()).forward(x)
    f_n, out_n = NumpyCNN().forward(x)
    assert np.allclose(f_n, f_t, atol=1e-3) and np.allclose(out_n, out_t, atol=1e-3)
    assert (out_n.argmax(1) == out_t.argmax(1)).all()
    assert np.abs(softmax(out_n) - softmax(out_t)).max() < 1e-4


def test_app_reads_the_same_with_the_numpy_model(monkeypatch):
    from common import LETTERS
    files = sorted(glob.glob(os.path.join(ROOT, "rehearsal_set", "q95", "*.jpg")))
    def read_all():
        return [predict.read_photo(load_image(f))[0][0][0][:2] for f in files]
    monkeypatch.setattr(predict, "_net", None)
    monkeypatch.setattr(predict, "BACKEND", "torch")
    with_torch = read_all()
    monkeypatch.setattr(predict, "_net", None)
    monkeypatch.setattr(predict, "BACKEND", "numpy")
    with_numpy = read_all()
    monkeypatch.setattr(predict, "_net", None)
    assert [r[0][0] for r in with_numpy] == [r[0][0] for r in with_torch]                     # same letters
    assert max(abs(a[0][1] - b[0][1]) for a, b in zip(with_numpy, with_torch)) < 1e-3           # same confidence
    assert len(with_numpy) == len(LETTERS)


def test_missing_weights_file_gives_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="numpy_cnn.py"):
        NumpyCNN(str(tmp_path / "nope.npz"))
