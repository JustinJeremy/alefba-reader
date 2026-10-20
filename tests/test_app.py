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
    assert r.status_code == 200 and b"Alefba Reader" in r.data


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
    import re
    assert len(re.findall(r'<option value="\d+"', html)) == 32 and "Save as test photo" in html


def test_home_page_has_paste_dark_mode_mascot_and_refusal_card(client):
    html = client.get("/").data.decode("utf-8")
    for part in ("Ctrl + V", 'id="themeBtn"', 'id="noghte-art"', 'id="refused"', 'id="plain"'):
        assert part in html, part


def test_page_works_offline_fonts_and_icon_come_from_the_app(client):
    css = client.get("/static/style.css").data.decode("utf-8")
    assert "http" not in css                                    # no fonts or images from the internet
    for f in ("Lora-Variable.ttf", "Vazirmatn-Regular.ttf", "Vazirmatn-Medium.ttf", "Vazirmatn-Bold.ttf"):
        assert f in css and client.get("/static/fonts/" + f).status_code == 200, f
    assert client.get("/static/icon.svg").status_code == 200


def test_save_writes_labelled_photo(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    saved = []
    for _ in range(2):                              # second photo of the same letter gets number 2
        r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": "1"},
                        content_type="multipart/form-data")
        assert r.status_code == 200
        saved.append(r.get_json()["saved"])
    assert saved == ["01_be_1.jpg", "01_be_2.jpg"]
    assert sorted(os.listdir(tmp_path / "random_test_photos")) == saved      # the default folder
    assert cv2.imread(str(tmp_path / "random_test_photos" / saved[0])) is not None


def test_save_into_a_chosen_folder(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": "1", "folder": "Kaye"},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["folder"] == "real_photos/Kaye"
    assert os.listdir(tmp_path / "Kaye") == ["01_be_1.jpg"]


def test_save_photo_of_several_letters(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    names = []
    for _ in range(2):
        r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_ids": "6,5",
                                       "folder": "multi_letter_photos"}, content_type="multipart/form-data")
        assert r.status_code == 200
        names.append(r.get_json()["saved"])
    assert names == ["06-05_1.jpg", "06-05_2.jpg"]            # letters in reading order, then the number
    import score_testset                                       # and the scorer reads the answer from the name
    labels, _ = score_testset.labelled_files(str(tmp_path), str(tmp_path / "multi_letter_photos"))
    assert labels == {"06-05_1.jpg": [6, 5], "06-05_2.jpg": [6, 5]}


@pytest.mark.parametrize("ids", ["6", "6,32", "6,x", ",".join(["1"] * 21)])
def test_save_rejects_bad_letter_lists(client, tmp_path, monkeypatch, ids):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_ids": ids},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and not os.listdir(tmp_path)


@pytest.mark.parametrize("folder", ["../outside", "..", "C:/Windows", "nope"])
def test_save_refuses_unknown_folders(client, tmp_path, monkeypatch, folder):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path / "real_photos"))
    r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": "1", "folder": folder},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "Unknown folder" in r.get_json()["error"]
    assert not os.path.exists(tmp_path / "outside") and not os.path.exists(tmp_path / "real_photos")


def test_folder_list_has_team_folders_then_others(tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    (tmp_path / "Mark").mkdir(); (tmp_path / "phone_tests").mkdir(); (tmp_path / "notes.txt").write_text("x")
    assert app_module.save_folders() == ["random_test_photos", "multi_letter_photos", "JJ", "Ethan", "Mark", "Kaye",
                                         "phone_tests"]


def test_home_page_offers_the_folders(client):
    html = client.get("/").data.decode("utf-8")
    for f in ("random_test_photos", "multi_letter_photos", "JJ", "Ethan", "Mark", "Kaye"):
        assert f'<option value="{f}"' in html


def test_numbering_continues_after_highest_and_is_per_letter(tmp_path):
    from app import next_free_name
    for n in ("01_be_1.jpg", "01_be_5.jpg", "01_be_desk.jpg", "02_pe_3.jpg", "01_be_7.png"):
        (tmp_path / n).write_bytes(b"x")
    assert next_free_name(str(tmp_path), 1) == "01_be_6.jpg"      # after 5; gaps and other names ignored
    assert next_free_name(str(tmp_path), 2) == "02_pe_4.jpg"
    assert next_free_name(str(tmp_path), 31) == "31_ye_1.jpg"     # first photo of a letter


@pytest.mark.parametrize("cls", ["", "abc", "-1", "32"])
def test_save_rejects_bad_letter(client, tmp_path, monkeypatch, cls):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(chart_bytes()), "x.jpg"), "class_id": cls},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and not any(os.scandir(tmp_path))


def test_save_rejects_non_image(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "PHOTOS_DIR", str(tmp_path))
    r = client.post("/save", data={"image": (io.BytesIO(b"nope"), "x.jpg"), "class_id": "3"},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and not any(os.scandir(tmp_path))


def test_file_labels_match_readme_names():
    from app import file_label
    assert [file_label(i) for i in (0, 1, 7, 20, 30, 31)] == ["00_alef", "01_be", "07_he", "20_eyn", "30_he2", "31_ye"]
    assert len({file_label(i) for i in range(32)}) == 32
