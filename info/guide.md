# Alefba Reader (Persian OCR): guide

Everything you need to run, test and change the project without help.
Run every command in the VS Code terminal, inside the `persian_ocr` folder.

---

## 1. What the project is

A web app and command-line tool that reads *Persian letters from a photo* (one letter, a marked
letter on a sheet, or a few separate letters) and gives their *English spelling*, using the
romanisation from the instructor's alphabet chart (e.g. غ → `gh`, ق → `q`).

*How it works*

1. *Clean-up* (OpenCV, `common.py`): fixes shadows and lighting, smooths film grain, removes
   ruled lines (also slanted, curved, faint or dashed ones, and where they cross a letter), frames,
   desk edges, fingers and captions, finds the letter and its dots, and turns it into a
   black-and-white 64×64 image. Small photos are enlarged first so dots survive. It also removes a
   ring around the letter (the edge of a coloured disc) and fills hollow (shiny) dots. When the
   photo has colour and the usual reading is unsure, it reads it again counting *colour* as ink
   too, so pastel and cartoon letters and dots in another colour are found.
2. *Marks* (`marks.py`): on a whole alphabet sheet, finds the letter that is *circled,
   highlighted or pointed at with an arrow*, in any marker colour. A letter that is itself
   coloured (pastel, cartoon) is not taken for a highlighter mark.
3. *Several letters* (`segment.py`): a photo of 2–20 separate letters is split into letters
   (body + dots), and each is read on its own, in Persian reading order (right to left).
4. *Model* (PyTorch CNN, ~774k parameters, `common.py`): 33 possible answers, the 32 letters
   plus *"not a letter"*.
5. *Refusals* (`predict.py`): blank or dark photos, noise, shapes running off the photo edge,
   non-Persian writing and objects get "No Persian letter found" instead of a guess.
6. *Lookup table*: the letter is turned into its spelling, e.g. `š`, or `sh` with `--ascii`.

*Training data*

- *Handwriting:* 32,000 images (Yousefnezhad dataset)
- *Printed:* 14,874 photo-like images in 37 fonts
- *"Not a letter":* 5,384 images (Latin letters, digits, shapes)
- *Hard copies:* 7,650 letters (and non-letters) with ruled lines, film grain or "blackout"
  (light letter fading into black), made by `effects.py`
- *Team handwriting:* 125 letters from Mark's, Kaye's and Ethan's clear sheets
  (`member_handwriting/`), each shown 15 times per round
- *Held out:* the Calibri family is never trained on, to measure accuracy on an unseen font

*Results* (full numbers in `results/metrics.txt`)

| Measure | Result |
|---|---|
| Handwritten | 98.8% |
| Printed | 97.9% |
| Unseen font | 96.9% |
| Letters with lines / grain / blackout | 99.0% |
| Real letters wrongly refused | 0.11% |
| Speed | ~3 ms per image (model only) |
| Team handwriting (members' earlier sheets, never trained on) | 104/107 (97.2%) |
| Photos of several letters (`multi_set`) | 24/25 |
| Rehearsal, marked sheets | 100% |
| Hard set (incl. ruled lines, film grain, blackout) | 383/384 |
| BUS testing dataset (`bus_test_set`: Arabic, pastel, coloured, cartoon letters) | 259/296 (87.5%); without the whole picture cards 255/268 (95.1%) |
| Automated tests | 347 (+1 known miss) |

The letters must be upright: upside-down, sideways (90°) or mirrored letters are not read
(small tilts up to about 10° are fine). Turned around, many letters become other letters
(پ and ث, ب and ن), so the app cannot guess which way is up.

---

## 2. Commands

### Setup (once per computer)

```
pip install -r requirements.txt
```

### Run the web app

| What | Command |
|---|---|
| Laptop only | `python app.py`, then open `http://127.0.0.1:5000` |
| Laptop *and phone* (same Wi-Fi) | `python app.py --host 0.0.0.0` (it prints both addresses) |
| Different port | `python app.py --port 5001` |
| Stop it | *Ctrl+C* in that terminal |

- The live camera works only at `127.0.0.1` on the laptop. On a phone, *Take photo* opens the
  phone's camera app instead.
- After changing any file, restart the app and press *Ctrl+F5* in the browser.
- If the phone can't open the page: allow Python through the Windows Firewall (private networks),
  or connect the laptop to the phone's hotspot and restart the app.
- *"Could not reach the app"* (older versions said "Failed to fetch"): the page is open but
  `python app.py` is no longer running (terminal closed, Ctrl+C, laptop slept). Start it again and
  press the button again; there is no need to reload the photo.

### Read letters from the command line

```
python predict.py photo1.jpg photo2.jpg photo3.jpg
python predict.py photo.jpg --ascii          (sh instead of š)
python predict.py photo.jpg --min-conf 0.8   (warn below 80% instead of 60%)
```

A photo with several letters prints one line per letter (`letter 1/3`, `letter 2/3`…), in Persian
reading order.

### Check everything still works

| What | Command | Time |
|---|---|---|
| All automated tests | `python -m pytest` | ~4 min |
| One test file | `python -m pytest tests/test_marks.py` | |
| Tests whose name contains a word | `python -m pytest -k save` | |
| Code scan (unused or undefined names) | `python -m pyflakes .` | seconds |

Expected result: `347 passed, 1 xfailed`. The "xfailed" is a known miss (Mark's ق is read as ض
in a row of two letters), which is fine.

### Score the test sets

*(This replaces the old `test_rehearsal.py`.)*

```
python score_testset.py rehearsal_set
python score_testset.py hard_set
python score_testset.py real_photos
python score_testset.py marked_set
python score_testset.py multi_set          (photos of several letters)
python score_testset.py bus_test_set       (BUS testing dataset: Arabic, pastel, coloured, cartoon)
```

### Rebuild the training data and retrain

Only needed if you change the model or the data.

| Step | Command | Time |
|---|---|---|
| 1. Handwriting (only if `data/` is missing) | `python convert_dataset.py <folder with Random1…Random10>` | ~1 min |
| 2. Printed letters, "not a letter" and hard copies (lines, grain, blackout) | `python prepare_data.py` | ~6 min |
| Only the hard copies (after changing `effects.py`) | `python prepare_data.py --effects-only` | ~3 min |
| 3. Train (also writes the letter profile and results; uses `member_handwriting/` too) | `python train.py` | *~2–3 h* |
| Only rebuild the letter profile | `python build_profile.py` | ~3 min |

After retraining, run `python -m pytest` and all five score commands before using the new model.
If the results get worse, git still has the old model (`git checkout persian_cnn.pt letter_profile.npz`).

### Git (when working without help)

```
git status                         what changed
git checkout -b my-change          new branch for a change
git add -A
git commit -m "what I changed"
git push -u origin my-change       then open a pull request on GitHub and merge it
git checkout main
git pull                           get the latest main after merging
```

---

## 3. Adding your own photos to the test sets

The test folders are ordinary folders: you *can* copy photos into them directly. What matters
is *which folder* and *the file name*, because that is how the scripts know the right answer.

*Rule for all of them: test folders are for testing only.* Never copy these photos into `data/`
(training). The project rules say the instructor's images must not be trained on, and a photo that
was trained on no longer tells you anything as a test.

### `real_photos/`: photos of one letter

- *Easiest:* read the photo in the web app, choose the right letter and the folder, click *Save photo*.
  The name is done for you. A photo of *several letters* gets one choice per letter (right to left)
  and goes to `real_photos/multi_letter_photos/` by default, named e.g. `06-05_1.jpg` (چ ج).
- *By hand:* copy the photo into one of the subfolders (`random_test_photos/`, `JJ/`, `Ethan/`,
  `Mark/`, `Kaye/`, or a new folder you make) and name it
  *`<number>_<name>_<anything>.jpg`*, e.g. `01_be_3.jpg`, `21_gheyn_phone.jpg`.
  Only the number decides the right answer; the full list of 32 numbers is in `real_photos/README.txt`.
- *Check:* `python score_testset.py real_photos`. Files without a number are skipped and listed.
- Automatic: every correctly named photo is scored.

### `multi_set/`: photos of several letters

- Copy the photo into `multi_set/` and name it with *all* its letters' numbers in Persian reading
  order (right to left, then the next row), joined by `-`: e.g. `07-08-09-10_row1.jpg` for ح خ د ذ.
- *Check:* `python score_testset.py multi_set`.
- Automatic: every photo here is also an automated test (`tests/test_multi.py`).
- Keep the letters well apart and fully inside the photo (a letter cut off by the photo edge is skipped).

### `member_handwriting/`: a member's letters for *training*

This is the one photo folder that *is* trained on. Add a member's clear alphabet sheet cut into one
photo per letter, named like the test folders (`05_jim_top.jpg`), in `member_handwriting/<Name>/`,
then retrain. Never put the same photos in `real_photos/` too: the members' *earlier* sheets in
`real_photos/<Member>/*_sheet.jpg` are the test for their handwriting.

### `marked_set/`: a whole alphabet sheet with one letter marked

- Copy the photo into *`marked_set/real/`* (or a new subfolder such as `marked_set/team/`).
- Name it after the *marked* letter: *`<number>_<name>_<mark>_<colour>.jpg`*,
  e.g. `05_jim_circle_red.jpg`, `27_mim_arrow_blue.jpg`, `20_eyn_highlight_yellow.jpg`.
- *Check:* `python score_testset.py marked_set`.
- Automatic: *every `.jpg` in a `marked_set/` subfolder also becomes an automated test* that must
  pass (`tests/test_marks.py`). If the app misreads your new photo, `python -m pytest` will fail.
  That is useful, because it shows a real problem to fix. If you just want to try a photo first,
  run `python predict.py yourphoto.jpg` before copying it in.
- A sheet *without* a mark: put it directly in `marked_set/` (not in a subfolder) with *no
  number* in the name. Check it with `python predict.py marked_set/yourphoto.jpg`: it should say
  "This photo shows many letters…".

### `no_letter_set/`: photos that must be refused (objects, people, English text…)

- Copy the photo into `no_letter_set/` with *any name without a number at the start*,
  e.g. `phone_keyboard.jpg`.
- *Check:* `python predict.py no_letter_set/phone_keyboard.jpg`. It should print `ERROR -`
  followed by a "No Persian letter found…" message. If it prints a letter instead, the app was fooled.
- Not automatic: only the photos listed by name in `tests/test_no_letter.py` are part of
  `python -m pytest`, and `score_testset.py` skips this folder (its names have no number).
  To check every photo in the folder at once:
  `python predict.py (Get-ChildItem no_letter_set\*.jpg)`
  Every line should say `ERROR` (the app icon and the number 77776 are refused too since batch 11).
  (The VS Code terminal is PowerShell, which does not expand `no_letter_set/*.jpg` by itself;
  `Get-ChildItem` does the expanding.)
- To make a new photo a permanent test, add its file name to the list in
  `test_screenshots_of_objects_are_refused` in `tests/test_no_letter.py`.
- List each new photo in `no_letter_set/README.txt` so the team knows what it is.

### After adding photos

Commit them like any other change (section 2, Git), so the whole team has the same test photos.

---

## 4. Quick changes: where to edit what

After any change: *restart the app*, press *Ctrl+F5*, and run *`python -m pytest`*.

| I want to change… | File | What to edit |
|---|---|---|
| A letter's English spelling | `common.py` | `LOOKUP` (chart spelling), `LOOKUP_ASCII` (plain), `NAMES` |
| Save folders in the dropdown | `app.py` | `DEFAULT_FOLDERS` (`MULTI_FOLDER` for photos of several letters), or just create a folder inside `real_photos/` |
| Maximum upload size (20 MB) | `app.py` | `MAX_CONTENT_LENGTH` |
| "Low confidence" warning (60%) | `static/app.js` | `t.confidence < 0.6` (2 places) |
| "Not sure" note (50%) | `static/app.js` | `t.confidence < 0.5` |
| Colours | `static/style.css` | the variables at the top of the file: `--primary` (banner, buttons), `--accent` (edges, bar), `--page`… |
| When the two-column layout starts (960 px) | `static/style.css` | `@media (min-width: 960px)` |
| Page text (headings, tips) | `templates/index.html` | the text itself |
| Error messages | `predict.py` | `NO_LETTER`, `NO_LETTER_BLANK`, `NO_LETTER_EDGE`, `NOT_PERSIAN`, `MANY_LETTERS` |
| How dark/blank a photo must be to be refused | `predict.py` | `MIN_CONTRAST = 12` (lower = refuses less) |
| Noise / grain refusal | `predict.py` | `MAX_INK = 0.32` (higher = refuses less), `MAX_GRAIN = 25` |
| When grainy photos get smoothed | `common.py` | `GRAIN = 2.5` (lower = smooths more photos) |
| Photos enlarged before clean-up (under 256 px) | `common.py` | `SMALL = 256` |
| "Unlike every letter" refusal | `predict.py` | `MAX_DISTANCE = 0.65` (higher = refuses less); `MULTI_MAX_DISTANCE = 0.85` for photos of several letters |
| Most letters read from one photo (20) | `predict.py` | `MAX_LETTERS` |
| Splitting a photo into letters | `segment.py` | `DOT_REACH` (how far a dot may be), `PIECE_GAP` (pieces of one letter), `MIN_SIZE` (smallest letter vs. the biggest) |
| English-text refusal on busy photos | `predict.py` | `MIN_PERSIAN = 0.22` (lower = refuses less) |
| When the stricter crop is also tried | `predict.py` | `STRICT_BELOW = 0.99` |
| How many shapes count as "many letters" | `marks.py` | `many_letters(img, limit=35)` |
| How colourful a mark must be vs. the paper | `marks.py` | `STANDOUT = 50` (lower = finds paler marks, but more false ones) |
| Highlighter vs. pen line | `marks.py` | `SOLID = 0.13` |
| Highlighter vs. a letter printed in colour | `marks.py` | `HIGHLIGHT_MAX = 0.1` (share of the photo), `HIGHLIGHT_INK = 0.08` (writing under it) |
| When the colour clean-up is tried / trusted | `predict.py` | `COLOUR_BELOW = 0.9`, `COLOUR_MIN = 0.5`; how strongly colour counts as ink: `COLOUR_INK = 2.5` in `common.py` |
| Hollow dots filled up to this size | `common.py` | `DOT_SIZE = 0.25` (x the letter body) |
| *Needs retraining:* training length | `train.py` | `EPOCHS = 30` |
| *Needs retraining:* fonts used | `prepare_data.py` | `WINDOWS_FONTS`, or add `.ttf` files to `fonts/` |
| *Needs retraining:* "not a letter" characters | `prepare_data.py` | `NOT_PERSIAN_CHARS` |
| *Needs retraining:* image size (64) | `common.py` | `SIZE`, plus the model's layers |
| *Needs retraining:* lines / grain / blackout copies | `effects.py`, `prepare_data.py` | the effect functions; `FX_HANDWRITTEN`, `FX_PRINTED`, `FX_NEGATIVE` (how many) |
| *Needs retraining:* how often team letters are shown | `train.py` | `MEMBER_REPEAT = 15` |

Every threshold has a comment next to it with the numbers it was measured on. Read that before
changing it, because loosening a refusal rule can start refusing real letters, or the other way round.

---

## 5. Folders

| Folder | What's in it |
|---|---|
| `rehearsal_set/`, `hard_set/` | letters from the slide-3 chart, clean and distorted, incl. ruled lines, film grain and blackout (test only) |
| `marked_set/` | alphabet sheets with one letter marked (test only) |
| `multi_set/` | photos of 2–4 letters, read letter by letter (test only) |
| `bus_test_set/` | the BUS testing dataset: Arabic, pastel, coloured-tile and cartoon letters (test only, see its README) |
| `real_photos/` | your saved photos: `random_test_photos/`, `JJ/`, `Ethan/`, `Mark/`, `Kaye/`; members' sheets |
| `member_handwriting/` | members' letters cut from their clear sheets (*training*) |
| `no_letter_set/` | photos the app must refuse |
| `results/` | `metrics.txt` (per-letter scores) and the confusion matrix, useful for slides |
| `data/` | training images (not on GitHub; rebuilt with steps 1–2 in section 2) |
| `static/`, `templates/` | the web page: style, behaviour, layout |
| `tests/` | the automated tests |
| `info/` | this guide |
