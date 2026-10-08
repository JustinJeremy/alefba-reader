"""The helper scripts import cleanly (no work at import time) and their pieces behave."""
import os
import numpy as np
import torch
from conftest import ROOT


def test_scripts_import_without_running():
    import train, prepare_data, convert_dataset, score_testset   # would train / write files if unguarded
    assert callable(train.main) and callable(prepare_data.main)
    assert callable(convert_dataset.main) and callable(score_testset.main)


def test_training_batch_shapes_and_range():
    import train
    X = np.random.RandomState(0).rand(10, 32, 32).astype(np.float32)
    y = np.arange(10)
    for is_train in (True, False):
        xb, yb = train.batch(X, y, np.arange(10), is_train)
        assert xb.shape == (10, 1, 32, 32) and yb.tolist() == list(range(10))
        assert 0 <= xb.min() and xb.max() <= 1


def test_score_testset_runs(capsys):
    import score_testset
    score_testset.main(os.path.join(ROOT, "rehearsal_set"))
    out = capsys.readouterr().out
    assert "q95: 32/32 correct" in out
