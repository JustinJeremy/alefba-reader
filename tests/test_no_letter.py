"""Photos with no Persian letter are refused with a clear message instead of a made-up answer."""
import io, os
import cv2
import numpy as np
import pytest
from conftest import ROOT
from predict import predict_with_view, predict_file, NO_LETTER, NO_LETTER_BLANK, PROFILE_PATH

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
