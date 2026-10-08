"""The class list and transliteration tables."""
import unicodedata
from common import LETTERS, LOOKUP, LOOKUP_ASCII, NAMES


def test_tables_have_32_entries():
    assert len(LETTERS) == len(LOOKUP) == len(LOOKUP_ASCII) == len(NAMES) == 32


def test_letters_are_unique_persian_code_points():
    assert len(set(LETTERS)) == 32
    assert "ک" in LETTERS and "ی" in LETTERS        # Persian kaf / ye, not Arabic
    assert "ك" not in LETTERS and "ي" not in LETTERS


def test_every_letter_has_its_own_spelling():
    # matches the instructor chart: homophones (z, ẕ, ż, ẓ ...) stay distinguishable
    assert len(set(LOOKUP)) == 32


def test_chart_spellings():
    spell = dict(zip(LETTERS, LOOKUP))
    assert spell["غ"] == "gh"     # غ (the example in the project slides)
    assert spell["ق"] == "q"      # ق
    assert spell["و"] == "w"      # و
    assert spell["ا"] == "-"      # ا


def test_ascii_table_is_plain_keyboard_text():
    for s in LOOKUP_ASCII:
        assert s and all(ord(ch) < 128 for ch in s), s
        assert s == s.lower()


def test_ascii_is_chart_spelling_without_accent_marks():
    special = {"-": "a", "š": "sh", "ʿ": "'"}      # alef, shin, eyn are spelled differently
    for chart, plain in zip(LOOKUP, LOOKUP_ASCII):
        stripped = "".join(c for c in unicodedata.normalize("NFD", chart) if not unicodedata.combining(c))
        assert special.get(chart, stripped) == plain, (chart, plain)
