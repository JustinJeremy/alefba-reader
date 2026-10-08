"""Image clean-up: shapes, polarity, and bad inputs."""
import cv2
import numpy as np
import pytest
from common import preprocess, load_image, SIZE


def letter_image(dark_on_light=True, channels=3):
    """A simple synthetic 'letter': a thick bar plus a dot above it."""
    bg, ink = (255, 0) if dark_on_light else (0, 255)
    img = np.full((200, 200), bg, np.uint8)
    cv2.rectangle(img, (60, 110), (140, 130), ink, -1)
    cv2.circle(img, (100, 70), 10, ink, -1)
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        out[:, :, 3] = 255
        return out
    return img


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_output_is_32x32_in_unit_range(channels):
    out = preprocess(letter_image(channels=channels))
    assert out.shape == (SIZE, SIZE) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.max() > 0.5                      # the letter survived


def test_letter_is_white_on_black():
    out = preprocess(letter_image())
    border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    assert border.max() < 0.1


def test_inverted_image_gives_same_result():
    a = preprocess(letter_image(dark_on_light=True))
    b = preprocess(letter_image(dark_on_light=False))
    assert np.abs(a - b).mean() < 0.02


def test_dot_is_kept():
    out = (preprocess(letter_image()) > 0.5).astype(np.uint8)
    n, _ = cv2.connectedComponents(out)
    assert n - 1 == 2                           # bar + dot


@pytest.mark.parametrize("img", [
    np.full((100, 100, 3), 255, np.uint8),      # blank white page
    np.zeros((100, 100, 3), np.uint8),          # blank black page
    np.full((100, 100), 128, np.uint8),         # flat grey
    np.zeros((100, 100, 4), np.uint8),          # fully transparent PNG
    np.zeros((1, 1, 3), np.uint8),              # 1 pixel
    np.zeros((0, 0, 3), np.uint8),              # empty array
])
def test_blank_or_empty_images_raise_valueerror(img):
    with pytest.raises(ValueError):
        preprocess(img)


def test_none_raises_valueerror():
    with pytest.raises(ValueError):
        preprocess(None)


def test_huge_photo_is_handled():
    big = cv2.resize(letter_image(), (4000, 4000), interpolation=cv2.INTER_NEAREST)
    assert preprocess(big).shape == (SIZE, SIZE)


def test_load_image_rejects_non_images():
    assert load_image(b"definitely not an image") is None


def test_load_image_reads_bytes_and_paths(tmp_path):
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), letter_image())
    assert load_image(str(p)).shape[:2] == (200, 200)
    assert load_image(p.read_bytes()).shape[:2] == (200, 200)
