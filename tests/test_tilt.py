"""Ruled paper photographed at a tilt: found, straightened, and never at the cost of upright photos."""
import os
import pytest
from conftest import ROOT
from common import load_image, ruled_tilt, TILT_MIN
from make_turned_set import turn_photo
import predict

SHEET = os.path.join(ROOT, "real_photos", "Mark", "01_be_sheet.jpg")      # be on ruled notebook paper


@pytest.mark.parametrize("angle", [20, 30, 90, -30, -90])
def test_tilt_of_ruled_paper_is_found(angle):
    t = ruled_tilt(turn_photo(load_image(SHEET), angle))
    assert t is not None
    assert abs((t - angle + 90) % 180 - 90) < 3                # same angle (up to upside down)


def test_level_paper_and_upright_strokes_are_not_straightened():
    assert ruled_tilt(load_image(SHEET)) is None                # the sheet as photographed: level
    lam = load_image(os.path.join(ROOT, "real_photos", "Mark", "26_lam_sheet.jpg"))
    assert ruled_tilt(lam) is None                              # a long upright stroke is not ruling
    assert ruled_tilt(turn_photo(load_image(SHEET), TILT_MIN / 2)) is None   # small tilts: read as they are


@pytest.mark.parametrize("sheet, angle", [("Kaye/06_che_sheet.jpg", 90), ("Ethan/09_dal_sheet.jpg", -90),
                                          ("Ethan/11_re_sheet.jpg", -90)])
def test_sideways_ruled_photo_is_read_after_straightening(sheet, angle):
    # the main model alone misreads these sideways sheets; straightened along the ruling it reads them
    img = turn_photo(load_image(os.path.join(ROOT, "real_photos", sheet)), angle)
    want = int(os.path.basename(sheet)[:2])
    with predict._using("main"):
        letters, how = predict._read_level(img)
    assert letters[0][0][0][0] == want and how.endswith(predict.LEVEL_HOW)
    letters, how = predict.read_photo(img)                       # and the whole reader
    assert letters[0][0][0][0] == want
    x, y, w, h = letters[0][1][1]                               # the box is in the photo as given
    assert 0 <= x and 0 <= y and x + w <= img.shape[1] and y + h <= img.shape[0]


def test_box_back_maps_corners_into_the_photo():
    import numpy as np
    img = np.full((200, 300, 3), 255, np.uint8)
    level, back = predict._level(img, 30)
    x, y, w, h = predict._box_back((0, 0, level.shape[1], level.shape[0]), back, img.shape)
    assert (x, y) == (0, 0) and x + w <= 300 and y + h <= 200


@pytest.mark.parametrize("name, angle", [("01_be.jpg", 90), ("23_qaf.jpg", -90), ("14_sin.jpg", 45), ("27_mim.jpg", 30),
                                         ("22_fe.jpg", -45)])
def test_turned_letter_on_plain_paper_is_read_by_the_tilt_model(name, angle):
    # no ruling to straighten by: the tilt model, asked when the main model is unsure, reads it
    assert predict.has_tilt_model()
    img = turn_photo(load_image(os.path.join(ROOT, "rehearsal_set", "q95", name)), angle)
    assert predict.read_photo(img)[0][0][0][0][0] == int(name[:2])


def test_sure_upright_readings_never_go_to_the_tilt_model():
    import glob
    for f in sorted(glob.glob(os.path.join(ROOT, "rehearsal_set", "q95", "*.jpg"))):
        img = load_image(f)
        with predict._using("main"):
            alone = predict._read_level(img)
        both = predict.read_photo(img)
        if alone[0][0][0][0][1] >= predict.BACKUP_BELOW:                # sure: the same answer, untouched
            assert both[0][0][0] == alone[0][0][0] and both[1] == alone[1], f
