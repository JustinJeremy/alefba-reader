"""Alphabet sheets with one letter marked (circle / box, highlighter, arrow), and drag-selection."""
import glob, io, os
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from conftest import ROOT
import marks
from predict import predict_file, predict_with_view, MANY_LETTERS
from common import load_image

MARKED = sorted(glob.glob(os.path.join(ROOT, "marked_set", "*", "*.jpg")))
SHEET = os.path.join(ROOT, "marked_set", "sheet_without_mark.jpg")
MIM_BOX = (1600, 1740, 220, 260)            # where mim is on the unmarked sheet (photo pixels)


@pytest.mark.parametrize("path", MARKED, ids=os.path.basename)
def test_marked_letter_is_read(path):
    truth = int(os.path.basename(path)[:2])
    (pred, conf), *_ = predict_file(path)
    assert pred == truth


@pytest.mark.parametrize("name,kind", [("27_mim_arrow_blue.jpg", "arrow"), ("31_ye_box_blue.jpg", "circle"),
                                       ("20_eyn_highlight_yellow.jpg", "highlight")])
def test_mark_kind_is_recognised(name, kind):
    t = marks.find_marked_letter(load_image(os.path.join(ROOT, "marked_set", "real", name)))
    assert t is not None and t.kind == kind


def test_unmarked_sheet_asks_for_a_mark():
    with pytest.raises(ValueError, match="many letters"):
        predict_file(SHEET)


def test_dragged_box_reads_only_that_letter():
    res, (view, box), how = predict_with_view(load_image(SHEET), select=MIM_BOX)
    assert res[0][0] == 27 and how == "your selection"
    x, y, w, h = box                                        # reported in whole-photo pixels
    assert MIM_BOX[0] <= x and x + w <= MIM_BOX[0] + MIM_BOX[2] and MIM_BOX[1] <= y


@pytest.mark.parametrize("sel", [(10, 10, 3, 200), (5000, 5000, 100, 100)])
def test_tiny_or_outside_selection_is_refused(sel):
    with pytest.raises(ValueError, match="too small"):
        predict_with_view(load_image(SHEET), select=sel)


def test_single_letter_photos_have_no_mark():
    for path in glob.glob(os.path.join(ROOT, "real_photos", "*", "*.jpg")):   # tinted webcam paper
        assert marks.find_marked_letter(load_image(path)) is None, path
    assert marks.find_marked_letter(load_image(os.path.join(ROOT, "rehearsal_set", "q95", "01_be.jpg"))) is None


def test_letter_written_in_coloured_pen_is_not_a_mark():
    font = ImageFont.truetype(os.path.join(ROOT, "fonts", "Vazirmatn-Bold.ttf"), 260)
    for colour in [(220, 30, 30), (30, 160, 60), (230, 120, 20)]:          # red, green, orange pen
        img = Image.new("RGB", (600, 600), (245, 245, 240))
        ImageDraw.Draw(img).text((300, 300), "ب", font=font, fill=colour, anchor="mm")   # be
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        res, _, how = predict_with_view(bgr)
        assert res[0][0] == 1 and how == "whole photo", colour


def test_small_images_are_never_sheets():
    assert not marks.many_letters(np.full((300, 300, 3), 255, np.uint8))
    assert not marks.many_letters(None)


def test_app_reports_how_and_uses_selection():
    from app import app
    c = app.test_client()
    data = open(SHEET, "rb").read()
    r = c.post("/predict", data={"image": (io.BytesIO(data), "s.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 400 and r.get_json()["error"] == MANY_LETTERS
    sel = dict(zip(("sel_x", "sel_y", "sel_w", "sel_h"), map(str, MIM_BOX)))
    r = c.post("/predict", data={"image": (io.BytesIO(data), "s.jpg"), **sel}, content_type="multipart/form-data")
    d = r.get_json()
    assert r.status_code == 200 and d["results"][0]["latin"] == "m" and d["how"] == "your selection"
    r = c.post("/predict", data={"image": (io.BytesIO(data), "s.jpg"), "sel_x": "abc", "sel_y": "1",
                                 "sel_w": "9", "sel_h": "9"}, content_type="multipart/form-data")
    assert r.status_code == 400 and "not valid" in r.get_json()["error"]
