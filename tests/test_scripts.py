"""The helper scripts import cleanly (no work at import time) and their pieces behave."""
import os
import numpy as np
import torch
from conftest import ROOT
from common import SIZE


def test_scripts_import_without_running():
    import train, prepare_data, convert_dataset, score_testset   # would train / write files if unguarded
    assert callable(train.main) and callable(prepare_data.main)
    assert callable(convert_dataset.main) and callable(score_testset.main)


def test_training_batch_shapes_and_range():
    import train
    X = np.random.RandomState(0).rand(10, SIZE, SIZE).astype(np.float32)
    y = np.arange(10)
    for is_train in (True, False):
        xb, yb = train.batch(X, y, np.arange(10), is_train)
        assert xb.shape == (10, 1, SIZE, SIZE) and yb.tolist() == list(range(10))
        assert 0 <= xb.min() and xb.max() <= 1


def test_font_name_is_read_from_printed_file_names():
    import train
    assert train.font_of(os.path.join("data", "printed", "3", "DUBAI-BOLD_0_7.png")) == "DUBAI-BOLD"
    assert train.font_of(os.path.join("data", "printed", "3", "Amiri-Regular_1_0.png")) == "Amiri-Regular"
    assert train.font_of(os.path.join("data", "handwritten", "3", "Random1_301.png")) == ""


def test_heldout_font_never_lands_in_training():
    import train
    y = np.repeat(np.arange(32), 80)
    src = np.array(["handwritten", "printed"] * (len(y) // 2))
    font = np.where(src == "printed", np.where(np.arange(len(y)) % 4 == 1, "calibri", "arial"), "")
    tr, va, te, held = train.split(y, src, font)
    assert set(font[held]) == {"calibri"}
    assert "calibri" not in set(font[np.concatenate([tr, va, te])])
    assert len(set(tr) | set(va) | set(te) | set(held)) == len(y)


def test_score_testset_runs(capsys):
    import score_testset
    score_testset.main(os.path.join(ROOT, "rehearsal_set"))
    out = capsys.readouterr().out
    assert "q95: 32/32 correct" in out
