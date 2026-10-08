"""The trained model and the command-line tool."""
import csv, os, subprocess, sys
import pytest
from conftest import ROOT
from common import PersianCNN
from predict import predict_file, load_model

CHART = os.path.join(ROOT, "rehearsal_set", "q95")
LABELS = {r["file"]: int(r["class_id"])
          for r in csv.DictReader(open(os.path.join(ROOT, "rehearsal_set", "labels.csv")))}


def chart(name):
    return os.path.join(CHART, name)


def test_model_has_expected_size():
    n = sum(p.numel() for p in PersianCNN().parameters())
    assert 700_000 < n < 850_000                # ~0.77 M parameters (64 px input, four conv blocks)


def test_model_accepts_preprocessed_size():
    import torch
    from common import SIZE
    assert PersianCNN()(torch.zeros(1, 1, SIZE, SIZE)).shape == (1, 33)      # 32 letters + not a letter


def test_missing_model_file_gives_clear_error(tmp_path):
    import predict
    saved, predict._model = predict._model, None
    try:
        with pytest.raises(FileNotFoundError, match="train.py"):
            load_model(str(tmp_path / "nope.pt"))
    finally:
        predict._model = saved


@pytest.mark.parametrize("name", sorted(LABELS))
def test_every_chart_letter_is_recognised(name):
    (pred, conf), *_ = predict_file(chart(name))
    assert pred == LABELS[name]


def test_top3_is_sorted_probabilities():
    res = predict_file(chart("00_alef.jpg"))
    confs = [c for _, c in res]
    assert len(res) == 3 and confs == sorted(confs, reverse=True) and sum(confs) <= 1.0001


def test_unreadable_file_raises_valueerror(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        predict_file(str(bad))


def run_cli(*args):
    # stdout is a pipe here, which is exactly the case that used to crash on Windows (cp1252)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    return subprocess.run([sys.executable, os.path.join(ROOT, "predict.py"), *args],
                          capture_output=True, encoding="utf-8", env=env, cwd=os.path.dirname(ROOT))


def test_cli_three_images():
    r = run_cli(chart("00_alef.jpg"), chart("15_shin.jpg"), chart("21_ghain.jpg"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "Output: - š gh"
    assert "ERROR" not in r.stdout + r.stderr


def test_cli_ascii_flag():
    r = run_cli("--ascii", chart("15_shin.jpg"))
    assert r.stdout.strip().splitlines()[-1] == "Output: sh"


def test_cli_keeps_going_after_a_bad_file():
    r = run_cli(chart("23_qaf.jpg"), "does_not_exist.jpg", chart("29_vav.jpg"))
    assert r.returncode == 0
    assert r.stdout.strip().splitlines()[-1] == "Output: q ? w"
    assert "does_not_exist.jpg: ERROR" in r.stderr
