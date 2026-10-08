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


def chart_bytes(name="01_be.jpg"):
    return open(os.path.join(ROOT, "rehearsal_set", "q95", name), "rb").read()


def test_predict_includes_what_the_model_saw(client):
    d = post(client, chart_bytes()).get_json()
    assert d["view"].startswith("data:image/png;base64,")
    assert d["found"].startswith("data:image/jpg;base64,")
    assert d["results"][0]["id"] == 1


def test_home_page_lists_all_32_letters_for_saving(client):
    html = client.get("/").data.decode("utf-8")
    assert html.count("<option value=") == 32 and "Save as test photo" in html


def test_save_writes_labelled_photo(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "SAVE_DIR", str(tmp_path))
    for _ in range(2):                              # two saves in the same second get different names
        r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": "1"},
                        content_type="multipart/form-data")
        assert r.status_code == 200
    names = sorted(os.listdir(tmp_path))
    assert len(names) == 2 and all(n.startswith("01_be_") and n.endswith(".jpg") for n in names)
    assert cv2.imread(str(tmp_path / names[0])) is not None


@pytest.mark.parametrize("cls", ["", "abc", "-1", "32"])
def test_save_rejects_bad_letter(client, tmp_path, monkeypatch, cls):
    import app as app_module
    monkeypatch.setattr(app_module, "SAVE_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": cls},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and os.listdir(tmp_path) == []


def test_save_rejects_non_image(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "SAVE_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(b"nope"), "x.jpg"), "class_id": "3"},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and os.listdir(tmp_path) == []


def test_file_labels_match_readme_names():
    from app import file_label
    assert [file_label(i) for i in (0, 1, 7, 20, 30, 31)] == ["00_alef", "01_be", "07_he", "20_eyn", "30_he2", "31_ye"]
    assert len({file_label(i) for i in range(32)}) == 32
