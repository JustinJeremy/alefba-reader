"""The BUS project test images (bus_test_set/): coloured, pastel and cartoon letters, letters on a disc,
and cartoon cards with a picture next to the letter."""
import glob, os
import cv2
import numpy as np
import pytest
from conftest import ROOT
import marks
from common import load_image, _remove_rings, _fill_dots
from predict import read_photo

BUS = os.path.join(ROOT, "bus_test_set")


def _files(*folders):
    return sorted(f for d in folders for f in glob.glob(os.path.join(BUS, d, "*.jpg")))


@pytest.mark.parametrize("path", _files("pastel_tight", "pastel_tiles", "cartoon_white", "cartoon_background"),
                         ids=lambda p: os.path.relpath(p, BUS))
def test_a_letter_in_colour_is_not_a_highlighter_mark(path):
    # before batch 11 a pastel or cartoon letter was taken for a highlighter stroke over a letter:
    # the letter itself was painted out and whatever was left was read (16/32 pastel letters right)
    t = marks.find_marked_letter(load_image(path))
    assert t is None or t.kind != "highlight"


def test_ring_around_the_letter_is_removed():
    bw = np.zeros((200, 200), np.uint8)
    cv2.circle(bw, (100, 100), 90, 255, 4)                 # edge of the disc the letter is printed on
    cv2.line(bw, (100, 60), (100, 140), 255, 8)            # the letter
    out = _remove_rings(bw)
    assert out[100, 10] == 0 and out[100, 100] == 255


@pytest.mark.parametrize("draw", ["loop", "bowl"])
def test_a_letters_own_loop_stays(draw):
    bw = np.zeros((200, 200), np.uint8)
    if draw == "loop":                                     # he: a closed loop with nothing inside
        cv2.circle(bw, (100, 100), 70, 255, 10)
    else:                                                  # nun: an open bowl with its dot inside the box
        cv2.ellipse(bw, (100, 90), (80, 70), 0, 0, 180, 255, 10)
        cv2.circle(bw, (100, 70), 8, 255, -1)
    assert np.array_equal(_remove_rings(bw), bw)


def test_hollow_dots_are_filled():
    bw = np.zeros((200, 200), np.uint8)
    cv2.rectangle(bw, (30, 120), (170, 135), 255, -1)      # letter body
    cv2.circle(bw, (100, 60), 12, 255, 4)                  # shiny dot: a ring
    out = _fill_dots(bw)
    assert out[60, 100] == 255 and out[128, 100] == 255
    big = np.zeros((200, 200), np.uint8)
    cv2.circle(big, (100, 100), 70, 255, 10)               # a letter's loop is not a dot: stays open
    assert np.array_equal(_fill_dots(big), big)


# fewest right answers per folder (batch 13 results; before batch 11: 28, 26, 11, 15, -, -, 16, 19, 32, 32).
# cartoon_with_pictures (whole cards): 4/28 until batch 13, when each coloured shape began to be read on its own
MIN_RIGHT = {"arabic_print": 28, "arabic_on_disc": 28, "cartoon_white": 25, "cartoon_background": 25,
             "cartoon_with_pictures": 17, "cartoon_with_pictures_boxed": 22, "pastel_tight": 32, "pastel_tiles": 32,
             "coloured_tiles_tight": 32, "coloured_tiles": 32}


@pytest.mark.parametrize("folder", MIN_RIGHT)
def test_folder_score(folder):
    right, wrong = 0, []
    for path in _files(folder):
        truth = int(os.path.basename(path)[:2])
        try:
            letters, _ = read_photo(load_image(path))
            ok = len(letters) == 1 and letters[0][0][0][0] == truth
        except ValueError:
            ok = False
        right += ok
        if not ok:
            wrong.append(os.path.basename(path))
    assert right >= MIN_RIGHT[folder], wrong


@pytest.mark.parametrize("path", _files("cartoon_with_pictures"), ids=os.path.basename)
def test_dragged_boxes_on_picture_cards_never_crash(path):
    # a box dragged over part of a card once crashed the ruled-line clean-up (empty line fit)
    img = load_image(path)
    h, w = img.shape[:2]
    rng = np.random.default_rng(int(os.path.basename(path)[:2]))
    for _ in range(4):
        x, y = int(rng.integers(0, w // 2)), int(rng.integers(0, h // 2))
        box = (x, y, int(rng.integers(w // 4, w - x + 1)), int(rng.integers(h // 4, h - y + 1)))
        try:
            read_photo(img, select=box)
        except ValueError:
            pass                                           # "no letter found" is a fine answer
