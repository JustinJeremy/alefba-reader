"""The helper scripts import cleanly (no work at import time) and their pieces behave."""
import os
import numpy as np
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


def test_score_testset_reads_labels_from_file_names(tmp_path, capsys):
    import shutil, score_testset
    chart = os.path.join(ROOT, "rehearsal_set", "q95")
    (tmp_path / "JJ").mkdir(); (tmp_path / "Ethan").mkdir()
    shutil.copy(os.path.join(chart, "01_be.jpg"), tmp_path / "JJ" / "01_be_desk.jpg")
    shutil.copy(os.path.join(chart, "23_qaf.jpg"), tmp_path / "JJ" / "23_qaf_1.jpg")
    shutil.copy(os.path.join(chart, "23_qaf.jpg"), tmp_path / "JJ" / "05_jim_mislabelled.jpg")
    shutil.copy(os.path.join(chart, "23_qaf.jpg"), tmp_path / "JJ" / "IMG_0001.jpg")
    (tmp_path / "JJ" / "00_alef_blank.png").write_bytes(open(os.path.join(chart, "00_alef.jpg"), "rb").read()[:50])
    score_testset.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "Ethan: (no labelled photos yet)" in out
    assert "JJ: 2/4 correct" in out
    assert "05_jim_mislabelled.jpg" in out and "00_alef_blank.png" in out and "ERROR" in out
    assert "skipped 1 file(s)" in out and "IMG_0001.jpg" in out
    assert "TOTAL: 2/4" in out


def test_phone_photo_keeps_the_letter_and_is_repeatable():
    import cv2
    import make_phone_set
    from common import load_image
    img = load_image(os.path.join(ROOT, "rehearsal_set", "q95", "01_be.jpg"))
    for level in ("mild", "strong"):
        a = make_phone_set.phone_photo(img, np.random.default_rng(1), level)
        b = make_phone_set.phone_photo(img, np.random.default_rng(1), level)
        assert a.ndim == 3 and min(a.shape[:2]) >= 100 and np.array_equal(a, b)    # same seed, same photo
        ink = (cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) < 100).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(ink))
        assert x > 0 and y > 0 and x + w < a.shape[1] and y + h < a.shape[0]       # the letter is not cut off
