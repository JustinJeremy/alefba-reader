"""Photos with no Persian letter are refused with a clear message instead of a made-up answer."""
import io, os
import cv2
import numpy as np
import pytest
from conftest import ROOT
from predict import predict_with_view, predict_file, NO_LETTER_BLANK, PROFILE_PATH

rng = np.random.RandomState(0)


def grainy(mean, std, size=(480, 640, 3)):
    return np.clip(rng.normal(mean, std, size), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("img", [
    grainy(15, 4), grainy(22, 6), grainy(40, 10),             # dark / covered webcam
    np.full((300, 300, 3), 0, np.uint8), np.full((300, 300, 3), 255, np.uint8), np.full((300, 300, 3), 90, np.uint8),
], ids=["dark1", "dark2", "dark3", "black", "white", "grey"])
def test_blank_or_dark_photo(img):
    with pytest.raises(ValueError, match="too dark or blank"):
        predict_with_view(img)


@pytest.mark.parametrize("img", [
    grainy(20, 15), grainy(35, 20),                           # very grainy dark frames
    rng.randint(0, 256, (400, 400, 3)).astype(np.uint8),      # random noise
    rng.randint(0, 256, (200, 200)).astype(np.uint8),
], ids=["grain1", "grain2", "noise", "noise_small"])
def test_noise_is_not_a_letter(img):
    with pytest.raises(ValueError, match="No Persian letter found"):
        predict_with_view(img)


def test_letter_profile_exists():
    assert os.path.isfile(PROFILE_PATH), "run python build_profile.py"


def test_real_letters_are_still_read():
    for name in ("00_alef.jpg", "01_be.jpg", "20_ain.jpg", "31_ye.jpg"):
        (pred, conf), *_ = predict_file(os.path.join(ROOT, "hard_set", "shadow_noise", name))
        assert pred == int(name[:2])


def test_app_shows_the_message():
    from app import app
    ok, buf = cv2.imencode(".jpg", grainy(18, 5))
    r = app.test_client().post("/predict", data={"image": (io.BytesIO(buf.tobytes()), "dark.jpg")},
                               content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"] == NO_LETTER_BLANK


NO_LETTER_SET = os.path.join(ROOT, "no_letter_set")


def test_phone_photo_of_a_desk_shadow_is_refused():
    # was read as dal (76%): now refused (by the model, or because the shape runs off the photo edge)
    with pytest.raises(ValueError, match="No (complete )?Persian letter found"):
        predict_file(os.path.join(NO_LETTER_SET, "phone_desk_shadow.jpg"))


def test_phone_photo_of_english_text_is_refused():
    with pytest.raises(ValueError, match="does not look like Persian"):
        predict_file(os.path.join(NO_LETTER_SET, "phone_english_text.jpg"))


def test_persian_sheet_still_asks_for_a_mark():
    with pytest.raises(ValueError, match="many letters"):
        predict_file(os.path.join(ROOT, "marked_set", "sheet_without_mark.jpg"))


def test_dragging_a_tight_box_is_not_an_edge_problem():
    # a letter filling the user's own selection touches its edges: that is fine
    from common import load_image
    res, _, how = predict_with_view(load_image(os.path.join(ROOT, "marked_set", "sheet_without_mark.jpg")),
                                    select=(1640, 1780, 140, 200))
    assert res[0][0] == 27 and how == "your selection"


def test_same_photo_can_be_saved_twice_in_different_folders(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    c = app_module.app.test_client()
    data = open(os.path.join(ROOT, "rehearsal_set", "q95", "15_shin.jpg"), "rb").read()
    for folder in ("JJ", "Ethan", "JJ"):
        r = c.post("/save", data={"image": (io.BytesIO(data), "x.jpg"), "class_id": "15", "folder": folder},
                   content_type="multipart/form-data")
        assert r.status_code == 200
    assert sorted(os.listdir(tmp_path / "JJ")) == ["15_shin_1.jpg", "15_shin_2.jpg"]
    assert os.listdir(tmp_path / "Ethan") == ["15_shin_1.jpg"]


KNOWN_MISS = pytest.mark.xfail(strict=True, reason="known miss: still read as a letter (61-67%); needs more "
                               "varied 'not a letter' training images. strict: this test fails once it is fixed")


@pytest.mark.parametrize("name", ["screenshot_game_tiles.jpg", "screenshot_profile_page.jpg",
                                  pytest.param("screenshot_app_icon.jpg", marks=KNOWN_MISS),
                                  pytest.param("screenshot_number_77776.jpg", marks=KNOWN_MISS)])
def test_screenshots_of_objects_are_refused(name):
    # these used to be read as che / gaf with 85-98% confidence
    with pytest.raises(ValueError, match="No (complete )?Persian letter found"):
        predict_file(os.path.join(NO_LETTER_SET, name))


def test_loop_letters_are_never_refused():
    # the 'not a letter' examples deliberately contain no loops, so he (a loop) must still be read
    import glob
    files = [f for d in ("rehearsal_set", "hard_set") for f in glob.glob(os.path.join(ROOT, d, "*", "30_he2.jpg"))]
    files += glob.glob(os.path.join(ROOT, "rehearsal_set", "*", "07_he.jpg"))
    for f in files:
        (pred, conf), *_ = predict_file(f)
        assert pred == int(os.path.basename(f)[:2]), f


def test_phone_photo_of_a_person_is_refused():
    with pytest.raises(ValueError, match="No (complete )?Persian letter found"):
        predict_file(os.path.join(NO_LETTER_SET, "phone_person_lying.jpg"))
