# Alefba Reader (Persian OCR): guide

Everything you need to run, test and change the project without help.
Run every command in the VS Code terminal, inside the `alefba_reader` folder.
Slides-ready summary for the presentation (technique, algorithm, software): `info/presentation.md`.

---

## 0. Getting the project from GitHub (step by step)

The repository is private: ask JJ (JustinJeremy) to add you as a collaborator first.

*Once per computer*

1. Install *Python 3.14* (python.org; tick "Add python.exe to PATH"), *Git* (git-scm.com) and
   *VS Code*.
2. Open VS Code, then *Terminal → New Terminal*, and go to where the project should live:
   `cd Downloads`
3. Copy the project from GitHub (sign in when asked):
   `git clone https://github.com/JustinJeremy/alefba-reader.git alefba_reader`
4. Open the folder: *File → Open Folder… → Downloads\alefba_reader*, then open a new terminal.
5. Install the libraries (a few minutes, torch is big): `pip install -r requirements.txt`
6. Check it works: `python -m pytest` (about 4 min, all should pass).
7. Start the app: `python app.py`, open `http://127.0.0.1:5000`, upload a photo, press *Read letter*.

The trained model (`persian_cnn.pt`) comes with the download, so nothing has to be trained.
Only the training images (`data/`, 1.4 GB) are not on GitHub; they are needed only to retrain
(section 2, "Rebuild the training data and retrain").

*Every time you start working*

1. `git checkout main` then `git pull` (gets everything merged since last time)
2. `pip install -r requirements.txt` only if `requirements.txt` changed
3. `python app.py`

## How do I...?

| To do this | Do these steps |
|---|---|
| Show the demo | `python app.py` → open `http://127.0.0.1:5000` → *Choose file*, *Take photo* or *Ctrl + V* (paste a copied image) → *Read letter* |
| Use a phone as the camera | Same Wi-Fi (or the phone's hotspot) → `python app.py --host 0.0.0.0` → open the second address it prints on the phone |
| Read a whole alphabet sheet | Circle, highlight or draw an arrow at one letter (any colour), or drag a box around it on the photo |
| Read a picture card / busy photo | Upload it; if the answer is wrong, drag a box around the letter and read again |
| Keep a photo for testing | After reading: pick the right letter under *Save as test photo* → choose the folder → *Save photo* |
| Check the app still works | `python -m pytest`, then the `python score_testset.py ...` commands (section 2) |
| Add test photos by hand | Name them `NN_name_anything.jpg` and copy them into a test folder (section 3) |
| Change a setting | Find it in section 4, edit, restart the app, *Ctrl+F5* in the browser |
| Make a change for the team | `git checkout -b my-change` → edit → test → `git add -A` → `git commit -m "..."` → `git push -u origin my-change` → open a pull request on GitHub → merge → `git checkout main` → `git pull` |
| Retrain the model | Only after changing training data or `common.py`'s clean-up: section 2, ~2–3 h; compare old and new scores before keeping it |
| Undo a bad model | `git checkout persian_cnn.pt letter_profile.npz` |

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
- *Team handwriting:* 125 letters (dim photos now cleaned correctly) from Mark's, Kaye's and Ethan's clear sheets
  (`member_handwriting/`), each shown 15 times per round
- *Held out:* the Calibri family is never trained on, to measure accuracy on an unseen font

*Results* (full numbers in `results/metrics.txt`)

| Measure | Result |
|---|---|
| Handwritten | 98.8% |
| Printed | 97.8% |
| Unseen font | 97.4% |
| Letters with lines / grain / blackout | 98.7% |
| Real letters wrongly refused | 0.13% |
| Speed | ~3 ms per image (model only); whole app: one letter ~0.1-0.3 s, marked sheet ~1 s (was ~2 s) |
| Team handwriting (members' earlier sheets, never trained on) | 98/99 (99.0%); only Ethan's pe missed |
| Photos of several letters (`multi_set`) | 25/25 |
| All team photos (`real_photos`, incl. notepad photos of several letters) | 109/110 (99.1%) |
| Rehearsal, marked sheets | 100% |
| Hard set (incl. ruled lines, film grain, blackout) | 384/384 |
| BUS testing dataset (`bus_test_set`: Arabic, pastel, coloured, cartoon letters) | 277/296 (93.6%); whole picture cards 17/28 (was 4/28) |
| Automated tests | 387, no known misses |

*Turned letters* (batch 16): tilted and sideways letters are read too. On `turned_set` (the test
photos turned to 15-345°, made with `python make_turned_set.py`): 440 -> 811 of 1,390 right, with
every upright score unchanged. How: ruled paper photographed at a tilt is first straightened along its
lines (`ruled_tilt` in `common.py`); and when the main model is unsure (under 97%) or refuses a photo
of one letter, a second model trained on turned letters (`persian_cnn_tilt.pt`) reads it too, and the
surer answer wins. Upside down stays hardest, because turned around some letters *are* other letters
(پ and ث, ب and ن). Mirrored letters are not read.

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
- *Ctrl + V* anywhere on the page pastes a copied image or a screenshot as the photo; a photo can also
  be dropped anywhere on the page. *Enter* reads it, and *✕ New photo* (on the photo) starts over.
- The moon / sun button switches light and dark mode; the browser remembers the choice.
- *This session* (top of the answer side) lists the letters read since the page was opened; click one
  to show its photo and answer again. It empties when the page is reloaded.
- *Alphabet chart* (under the buttons) shows all 32 letters; clicking one shows it like an answer,
  without a photo. *Copy* next to the spelling copies it.
- *How the app reads your photo* and *Save as test photo* start folded: click their titles to open them.
- On a phone the page scrolls down to the answer by itself; the *New photo* button at the bottom goes back up.
- *✕ Clear* (top right of the answer) clears the photo and its answer; for a letter opened from the chart
  it says *✕ Close* and goes back to the answer that was there before.
- Small photos and screenshots are drawn large, so a box can be dragged on them.
- *Tight crops* (batch 17): a letter cropped with no white space round it (a screenshot cut to the letter)
  touches the photo's edges, where the clean-up removes desk edges and shadows. When the first reading is
  unsure (under 90%) or refused, the photo is read again with a border of its own paper colour, and that
  answer is used when it is at least 90% sure and surer. `tight_set/00`: 128 -> 133 of 139 right; every
  upright score unchanged. Settings: `PAD_BELOW`, `PAD_SURE`, `PAD_EDGE`, `PAD_SIDES` in `predict.py`.
- The page uses short animations (fades, Noghte hopping); they switch off when the computer or phone is
  set to reduce motion.
- "No Persian letter found" and "many letters" answers appear on the answer side with the mascot;
  problems with the camera, the file or the app appear in red under the buttons.
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

Expected result: `387 passed`.

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
| 3b. Train the tilt model (turned letters too; writes `persian_cnn_tilt.pt`, `letter_profile_tilt.npz`, `persian_cnn_tilt.npz`, `results_tilt/`) | `python train.py --tilt` | *~1 h on the GPU* |
| Score turned letters | `python make_turned_set.py` then `python score_testset.py turned_set` | ~10 min |
| Score tightly cropped letters | `python make_tight_set.py` then `python score_testset.py tight_set` | ~5 min |
| Only rebuild the letter profile | `python build_profile.py` | ~3 min |

After retraining, run `python -m pytest` and all five score commands before using the new model.

*Training on the GPU (much faster):* a separate environment with the GPU (CUDA) version of PyTorch is at
`VS Code Project Files\alefba_gpu_env` (outside the project). From the project folder:
`..\alefba_gpu_env\Scripts\python train.py --tilt`. Plug the laptop in and keep the lid open;
Windows is kept awake while it trains.
During training `persian_cnn.pt` is overwritten after every better round: do not use the app,
run tests or commit until it has finished (copy the old model somewhere first to compare).
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
- At most 20 letters per photo (`MAX_LETTERS`); a whole sheet asks you to mark one letter instead.
- Shadows, printed form boxes ("NO./DATE") and other shapes the model refuses are left out and the
  photo is looked at again (up to `SEVERAL_TRIES = 3` times), so they no longer hide small letters.

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
| Colours | `static/style.css` | the variables at the top of the file: `:root` for light mode (`--side` panel, `--page` answer stage, `--gold` buttons, `--saffron` bar and rings…), `:root[data-theme=dark]` for dark mode |
| The mascot (Noghte) | `templates/index.html` | the `<template id="noghte-art">` drawing; its moods are picked in `static/app.js` (`mood(...)`) |
| How many answers *This session* keeps (10) | `static/app.js` | `HISTORY_MAX` |
| Letters with only 2 shapes inside a word | `static/app.js` | `NON_JOINING` (ا آ د ذ ر ز ژ و) |
| How long "Reading your photo…" stays at least (0.9 s) | `static/app.js` | `READING_MIN_MS = 900` |
| Tallest the chosen photo is drawn (420 px) | `static/app.js` | `PHOTO_MAX_H = 420` |
| Animations | `static/style.css` | the "animations" section at the end of the file |
| When the side-panel layout starts (960 px) | `static/style.css` | `@media (min-width: 960px)` (narrower: one column) |
| Page text (headings, tips) | `templates/index.html` | the text itself |
| Error messages | `predict.py` | `NO_LETTER`, `NO_LETTER_BLANK`, `NO_LETTER_EDGE`, `NOT_PERSIAN`, `MANY_LETTERS` |
| How dark/blank a photo must be to be refused | `predict.py` | `MIN_CONTRAST = 12` (lower = refuses less) |
| Noise / grain refusal | `predict.py` | `MAX_INK = 0.32` (higher = refuses less), `MAX_GRAIN = 25` |
| When grainy photos get smoothed | `common.py` | `GRAIN = 2.5` (lower = smooths more photos) |
| Photos enlarged before clean-up (under 256 px) | `common.py` | `SMALL = 256` |
| Speed of the lighting clean-up on big photos | `common.py` | `BG_SHRINK = 4` (paper brightness measured on a 4x smaller copy; 1 = old, slower way) |
| Picture cards (coloured letter next to a picture) | `predict.py` | `CARD_UNSURE = 0.97` (used only when the plain reading is less sure), `CARD_CONF = 0.7`, `CARD_SAT = 90` |
| Phone-sharpening halo, notebook ticks | `common.py` | `RIM = 5` (1 = off), `TICK_ROW = 3`, `TICK_GAP = 5` |
| Refuse weak guesses | `predict.py` | `MIN_SHOW = 0.4` (a single letter read less surely than this gets "No Persian letter found") |
| "Unlike every letter" refusal | `predict.py` | `MAX_DISTANCE = 0.65` (higher = refuses less); `MULTI_MAX_DISTANCE = 0.42` for photos of several letters |
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

## 6. Putting it online (Render, free plan)

*Not done yet: the files are ready, nothing is online.* The laptop app stays the one for the demo
(no internet needed, never asleep).

*What is prepared*

| File | What it does |
|---|---|
| `render.yaml` | Tells Render how to build and start the app (free plan, public mode on) |
| `requirements-server.txt` | The packages the online copy needs: no PyTorch, no training tools |
| `.python-version` | Python 3.14 |
| `numpy_cnn.py` | Runs the model with NumPy only (PyTorch needs about half of the free plan's 512 MB) |
| `persian_cnn.npz` | The trained weights as plain arrays, made from `persian_cnn.pt` |

*Public mode* (`ALEFBA_PUBLIC=1`, set by `render.yaml`): "Save as test photo" is hidden and the server
refuses to save, so visitors cannot write files onto it. On the laptop nothing changes.

*Steps (once)*

1. Merge everything on GitHub (the online copy is built from `main`).
2. On render.com: *New → Blueprint*, connect GitHub, pick `JustinJeremy/alefba-reader`, then *Apply*.
   (Or *New → Web Service* by hand: runtime *Python*, build `pip install -r requirements-server.txt`,
   start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`,
   plan *Free*, and an environment variable `ALEFBA_PUBLIC` = `1`.)
3. Wait for *Live* (about 3-5 min), then open the `https://alefba-reader....onrender.com` address it shows.
4. Put that address in *Website* in the GitHub repo's About.

After that, every merge on GitHub updates the website by itself.

*Good to know*
- The free plan sleeps after 15 min without visitors; the first visit after that takes about a minute.
- It has a small share of one processor (0.1 CPU): expect a few seconds per photo, more for whole sheets.
- The live camera works on phones there too (the site is https).

*After retraining*, `train.py` also rewrites `persian_cnn.npz`; commit it together with `persian_cnn.pt`
(`tests/test_numpy_cnn.py` fails if the two do not match). By hand: `python numpy_cnn.py`.
To check on the laptop that the NumPy copy reads the same: `$env:ALEFBA_BACKEND="numpy"; python -m pytest`
(PowerShell; takes about twice as long).

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
