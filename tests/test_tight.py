"""A letter cropped tightly (no white space round it) is read, with its box inside the photo."""
import os
import pytest
from conftest import ROOT
from common import load_image, preprocess
import predict


def tight(path, margin=0):
    img = load_image(path)
    info = {}
    preprocess(img, info=info)
    x, y, w, h = info["box"]
    m = int(margin * max(w, h) / 100)
    return img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]


@pytest.mark.parametrize("name", ["05_jim.jpg", "15_shin.jpg", "23_qaf.jpg", "27_mim.jpg", "31_ye.jpg"])
def test_tightly_cropped_chart_letter_is_read(name):
    crop = tight(os.path.join(ROOT, "rehearsal_set", "q95", name))
    letters, how = predict.read_photo(crop)
    assert letters[0][0][0][0] == int(name[:2])
    x, y, w, h = letters[0][1][1]
    assert 0 <= x and 0 <= y and x + w <= crop.shape[1] and y + h <= crop.shape[0]


@pytest.mark.parametrize("path", ["Ethan/05_jim_sheet.jpg", "Kaye/11_re_sheet.jpg", "Mark/23_qaf_sheet.jpg"])
def test_tightly_cropped_handwriting_is_read(path):
    crop = tight(os.path.join(ROOT, "real_photos", path))
    assert predict.read_photo(crop)[0][0][0][0][0] == int(os.path.basename(path)[:2])


def test_border_has_the_paper_colour_and_box_maps_back():
    import numpy as np
    img = np.full((40, 30, 3), (200, 210, 220), np.uint8)
    img[10:30, 12:18] = 0                                       # a dark stroke
    big = predict._with_border(img, 10)
    assert big.shape == (60, 50, 3) and tuple(big[0, 0]) == (200, 210, 220)
    assert predict._box_unpad((10, 10, 30, 40), 10, img.shape) == (0, 0, 30, 40)
    assert predict._box_unpad((0, 0, 60, 60), 10, img.shape) == (0, 0, 30, 40)    # kept inside the photo
