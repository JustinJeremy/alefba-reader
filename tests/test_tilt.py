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


def test_turned_whole_photo_keeps_all_of_the_letter_and_its_paper():
    import numpy as np
    from make_turned_set import turn_whole
    img = load_image(SHEET)
    for angle in (30, 135):
        t = turn_whole(img, angle)
        assert min(t.shape[:2]) >= 0.9 * min(img.shape[:2])               # not cropped small
        corners = np.array([t[0, 0], t[0, -1], t[-1, 0], t[-1, -1]], np.float32)
        assert np.abs(corners - np.median(img.reshape(-1, 3), 0)).max() < 40   # paper, not empty corners


def test_levelled_corners_carry_the_ruled_lines_on_level():
    # turned back, the empty corners are filled along each row: no slanted streaks are left for the
    # line removal to miss, so the levelled photo shows no tilt any more
    from make_turned_set import turn_whole
    level, _ = predict._level(turn_whole(load_image(SHEET), 30), 30)
    assert ruled_tilt(level) is None


@pytest.mark.parametrize("sheet, angle", [("Kaye/00_alef_sheet.jpg", 30), ("Kaye/01_be_sheet.jpg", 30),
                                          ("Mark/00_alef_sheet.jpg", 330), ("Kaye/06_che_sheet.jpg", 330),
                                          ("Kaye/01_be_sheet.jpg", 90), ("Ethan/03_te_sheet.jpg", 90),
                                          ("Kaye/03_te_sheet.jpg", 90), ("Kaye/00_alef_sheet.jpg", 135)])
def test_turned_ruled_photo_is_read_turned_back_either_way(sheet, angle):
    # whole turned photos (not cropped); sideways and 135 degrees: the ruling cannot tell which way is up,
    # so both ways are read. Each of these was read wrongly before batch 20.
    from make_turned_set import turn_whole
    img = turn_whole(load_image(os.path.join(ROOT, "real_photos", sheet)), angle)
    letters, how = predict.read_photo(img)
    assert letters[0][0][0][0] == int(os.path.basename(sheet)[:2]), how


@pytest.mark.parametrize("sheet", ["Mark/03_te_sheet.jpg", "Mark/05_jim_sheet.jpg", "Mark/07_he_sheet.jpg"])
def test_one_ruled_line_across_the_photo_is_enough(sheet):
    # these show a single ruled line: running edge to edge it is ruling (a letter's stroke has paper round it)
    from make_turned_set import turn_whole
    img = turn_whole(load_image(os.path.join(ROOT, "real_photos", sheet)), 30)
    assert abs(ruled_tilt(img) - 30) < 3
    assert predict.read_photo(img)[0][0][0][0][0] == int(os.path.basename(sheet)[:2])
