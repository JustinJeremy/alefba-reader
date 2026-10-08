"""The Flask web app."""
import io, os
import cv2
import numpy as np
import pytest
from conftest import ROOT
from app import app


@pytest.fixture
def client():
    return app.test_client()


def post(client, data, name="x.jpg"):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Persian letter reader" in r.data


def test_predict_returns_letter_name_and_spellings(client):
    data = open(os.path.join(ROOT, "rehearsal_set", "q95", "21_ghain.jpg"), "rb").read()
    r = post(client, data)
    assert r.status_code == 200
    top = r.get_json()["results"][0]
    assert top == {**top, "letter": "غ", "latin": "gh", "ascii": "gh", "name": "gheyn"}
    assert 0 < top["confidence"] <= 1
    assert len(r.get_json()["results"]) == 3


def test_no_file_is_400(client):
    r = client.post("/predict")
    assert r.status_code == 400 and "No image" in r.get_json()["error"]


def test_not_an_image_is_400(client):
    r = post(client, b"hello")
    assert r.status_code == 400 and "not a readable image" in r.get_json()["error"]


def test_blank_photo_is_400(client):
    ok, buf = cv2.imencode(".jpg", np.full((300, 300, 3), 255, np.uint8))
    r = post(client, buf.tobytes())
    assert r.status_code == 400 and "blank" in r.get_json()["error"]


def test_too_large_upload_is_rejected(client):
    r = post(client, b"0" * (21 * 1024 * 1024))
    assert r.status_code == 413
