"""convert_dataset.py: file numbering -> class folders."""
import os
import cv2
import numpy as np
import pytest
import convert_dataset


def test_numbers_map_to_classes_and_bad_files_are_skipped(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src" / "Random1"
    src.mkdir(parents=True)
    img = np.full((20, 20), 255, np.uint8)
    for n in (1, 100, 101, 3200):               # alef, alef, be, ye
        cv2.imwrite(str(src / f"aa ({n}).bmp"), img)
    cv2.imwrite(str(src / "no_number.bmp"), img)   # name without (N)
    cv2.imwrite(str(src / "aa (3201).bmp"), img)   # beyond the 32 letters
    (src / "aa (5).bmp").write_bytes(b"broken")     # unreadable
    monkeypatch.chdir(tmp_path)

    convert_dataset.main(str(tmp_path / "src"))

    out = tmp_path / "data" / "handwritten"
    assert sorted(os.listdir(out / "0")) == ["Random1_1.png", "Random1_100.png"]
    assert os.listdir(out / "1") == ["Random1_101.png"]
    assert os.listdir(out / "31") == ["Random1_3200.png"]
    assert "skipped 3 files" in capsys.readouterr().out


def test_empty_folder_exits_with_message(tmp_path):
    with pytest.raises(SystemExit, match="No Random"):
        convert_dataset.main(str(tmp_path))
