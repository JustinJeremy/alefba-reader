"""Photos with several letters: each letter found and read, in Persian reading order (right to left)."""
import glob, io, os
import numpy as np
import pytest
from common import load_image
from predict import read_photo
from segment import find_letters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Mark writes qaf with a long open tail: read as zad, alone too (real_photos/Mark/23_qaf_sheet.jpg).
# A model miss, not a splitting one: the row is still split into its two letters correctly.
KNOWN = {"22-23_mark_row9.jpg": "Mark's qaf read as zad (also alone); needs more qaf in this style"}
MULTI = [pytest.param(f, marks=pytest.mark.xfail(strict=True, reason=KNOWN[os.path.basename(f)]))
         if os.path.basename(f) in KNOWN else f
         for f in sorted(glob.glob(os.path.join(ROOT, "multi_set", "*.jpg")))]


def truth(path):
    """'07-08-09-10_mark_row1.jpg' -> [7, 8, 9, 10] (reading order)."""
    return [int(c) for c in os.path.basename(path).split("_")[0].split("-")]


@pytest.mark.parametrize("path", MULTI, ids=lambda p: os.path.basename(p) if isinstance(p, str) else None)
def test_letters_in_a_row_are_read_in_order(path):
    letters, how = read_photo(load_image(path))
    assert [res[0][0] for res, _ in letters] == truth(path)
    assert "right to left" in how


def test_single_letter_photos_stay_single():
    # a photo of one letter must never be split into several (dots, captions, neighbours at the edge)
    files = [f for d in ("rehearsal_set/q95", "hard_set/caption", "hard_set/neighbor", "hard_set/border",
                         "hard_set/ruled_lines", "real_photos/random_test_photos")
             for f in glob.glob(os.path.join(ROOT, d, "*.jpg"))]
    split = [os.path.relpath(f, ROOT) for f in files if len(read_photo(load_image(f))[0]) > 1]
    assert not split, split


def test_finder_order_is_right_to_left_then_down():
    img = np.full((300, 600), 255, np.uint8)
    import cv2
    for x, y in ((450, 60), (250, 60), (450, 200)):              # two letters on top, one below
        cv2.line(img, (x, y), (x, y + 70), 0, 6)
    xs = [(l.box[0], l.row) for l in find_letters(img)]
    assert [r for _, r in xs] == [0, 0, 1] and xs[0][0] > xs[1][0]


def test_app_returns_every_letter():
    from app import app
    c = app.test_client()
    path = os.path.join(ROOT, "multi_set", "07-08_mark_pair1.jpg")
    r = c.post("/predict", data={"image": (io.BytesIO(open(path, "rb").read()), "p.jpg")},
               content_type="multipart/form-data")
    d = r.get_json()
    assert r.status_code == 200 and [l["results"][0]["id"] for l in d["letters"]] == [7, 8]
    assert d["results"] == d["letters"][0]["results"]          # first letter, as before


def test_member_letters_are_training_only():
    import train
    y = np.array([0] * 40 + [1] * 40 + [0, 1])
    src = np.array(["handwritten"] * 80 + ["members"] * 2)
    font = np.array([""] * 82)
    tr, va, te, held = train.split(y, src, font)
    assert set(np.where(src == "members")[0]) <= set(tr)
    assert not set(np.where(src == "members")[0]) & (set(va) | set(te))


@pytest.mark.parametrize("name", ["30-18-10-21_1.jpg", "30-18-10-21_2.jpg"])
def test_small_letters_next_to_a_shadow_and_form_box(name):
    # notepad photo: four small letters, a ceiling shadow and the printed "NO./DATE" box. The shadow and
    # box made he and zal look like dots (too small, pen too thick) and gheyn's dot too far away;
    # the refused shapes are now left out and the letters looked for again
    path = os.path.join(ROOT, "real_photos", "multi_letter_photos", name)
    letters, how = read_photo(load_image(path))
    assert [res[0][0] for res, _ in letters] == truth(path)
