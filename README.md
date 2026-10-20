# Alefba Reader: Persian Letter Recognition (CNN)

A web app and command-line tool that reads *Persian letters from a photo* and gives their
*English spelling* (e.g. غ → `gh`, ق → `q`), using the romanisation from the instructor's
alphabet chart (slide 3).

It can read:

- *one letter*: handwritten or printed, on plain, ruled or coloured paper
- *a marked letter on a whole alphabet sheet*: circled, boxed, highlighted or pointed at with an arrow, in any marker colour
- *several separate letters* in one photo (2–20), read right to left
- and it *refuses* blank photos, noise, objects and non-Persian writing with "No Persian letter found"

Full guide (step-by-step setup, every command, adding test photos, where to change what):
[`info/guide.md`](info/guide.md) · Presentation summary: [`info/presentation.md`](info/presentation.md)

---

## Quick start

```
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>, upload, paste or take a photo, and press *Read letter*.

The trained model (`persian_cnn.pt`) is included, so nothing has to be trained first.
Everything runs offline: the fonts are stored in `static/fonts/`.

## Usage

### Web app

| What | Command |
|---|---|
| Laptop only | `python app.py`, then open `http://127.0.0.1:5000` |
| Laptop and phone (same Wi-Fi) | `python app.py --host 0.0.0.0` (it prints the phone address) |
| Different port | `python app.py --port 5001` |

The page has a brown side panel with everything you do (photo, buttons, saving) and an answer
stage where the letter is shown large, next to its spelling and how sure the app is.

- Choose a photo with *Choose file*, *Take photo*, *Ctrl + V* (paste a copied image or screenshot),
  or drop it anywhere on the page. *Enter* reads it; *✕ New photo* on the photo, or *✕ Clear* beside
  the answer, starts over. Small photos are drawn large so a box can be dragged on them.
- Drag a box on the photo to read only that part.
- On a whole alphabet sheet with no mark, the app asks for one, or you can drag a box around the letter.
- The answer shows the letter, its spelling (with a *Copy* button), how sure the app is, other possible
  letters, and how the letter looks inside a word (at the start, in the middle, at the end).
- *This session* lists every letter read since the page was opened; click one to see it again.
- *Alphabet chart* shows all 32 letters with their spellings; click one to see it large.
- *How the app reads your photo* and *Save as test photo* are folded; click to open them.
  Saving puts the photo into the chosen `real_photos/` folder on the laptop (also from a phone).
- On a phone the page scrolls to the answer by itself; *New photo* at the bottom goes back up.
- The moon / sun button switches between light and dark mode; the browser remembers the choice.
- Noghte (نقطه, "dot"), the ink-dot mascot, shows how the app feels about its answer: happy, thinking or puzzled.

### Command line

```
python predict.py img1.jpg img2.jpg img3.jpg
python predict.py photo.jpg --ascii          (sh instead of š)
python predict.py photo.jpg --min-conf 0.8   (warn below 80% instead of 60%)
```

## How it works

1. *Clean-up* (`common.py`, OpenCV): fixes shadows and lighting, removes ruled lines, frames
   and grain, finds the letter and its dots, and turns it into a black-and-white 64×64 image.
2. *Marks* (`marks.py`): finds the circled / highlighted / arrowed letter on a whole sheet.
3. *Several letters* (`segment.py`): splits a photo into separate letters (body + dots).
4. *Model* (`common.py`, PyTorch CNN, ~774k parameters): 33 answers, the 32 letters plus "not a letter".
5. *Refusals* (`predict.py`): photos that are not a Persian letter get "No Persian letter found".
6. *Lookup*: the letter is turned into its spelling, following the instructor's chart.

## Results

Full numbers in [`results/metrics.txt`](results/metrics.txt).

| Measure | Result |
|---|---|
| Handwritten | 98.8% |
| Printed | 97.8% |
| Unseen font (Calibri, never trained on) | 97.4% |
| Letters with ruled lines / grain / blackout | 98.7% |
| Real letters wrongly refused | 0.13% |
| Team photos (`real_photos`) | 109/110 |
| Photos of several letters (`multi_set`) | 25/25 |
| Hard set (`hard_set`) | 384/384 |
| Rehearsal and marked sheets | 100% |
| BUS testing dataset (`bus_test_set`) | 277/296 (93.6%) |
| Speed | ~3 ms per image (model only); ~0.1–0.3 s per photo in the app |

Letters must be upright (small tilts up to about 10° are fine). Turned around, many letters
become other letters (پ and ث, ب and ن), so the app can't tell which way is up.

## Testing

```
python -m pytest                        (all automated tests, ~4 min, expect 355 passed)
python score_testset.py rehearsal_set
python score_testset.py hard_set
python score_testset.py real_photos
python score_testset.py marked_set
python score_testset.py multi_set
python score_testset.py bus_test_set
```

Test sets are *test only*: never train on them.

| Folder | What it holds |
|---|---|
| `real_photos/` | Team photos of one letter, named like `01_be_anything.jpg` (see `real_photos/README.txt`) |
| `multi_set/` | Photos of several letters |
| `marked_set/` | Whole alphabet sheets with one letter marked |
| `hard_set/`, `rehearsal_set/` | Harder and demo-style photos |
| `bus_test_set/` | Arabic, pastel, coloured and cartoon letters |
| `no_letter_set/` | Photos that must be refused (checked by `tests/test_no_letter.py`) |

## Retraining (optional)

Only needed after changing the training data or the model. The training images (`data/`,
1.4 GB) are not on GitHub.

| Step | Command |
|---|---|
| 1. Handwriting data | `python convert_dataset.py <folder with Random1..Random10>` → `data/handwritten/<0..31>/` |
| 2. Printed + "not a letter" data | `python prepare_data.py` (fonts in `fonts/` plus the Windows Arabic-script fonts) |
| 3. Train (~2 h on CPU) | `python train.py` → `persian_cnn.pt`, `letter_profile.npz`, `results/` |
| 4. Rebuild letter profile only | `python build_profile.py` |

Compare the old and new scores before keeping a new model. To undo:
`git checkout persian_cnn.pt letter_profile.npz`.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask web app |
| `templates/index.html`, `static/` | Web page, styles, script, fonts and the tab icon |
| `predict.py` | Command-line reader and refusal rules |
| `common.py` | Image clean-up, model, letter list (`LETTERS`) and romanisation |
| `marks.py` | Finding the marked letter on a whole sheet |
| `segment.py` | Splitting a photo into several letters |
| `effects.py` | Ruled lines, film grain and blackout effects for training |
| `train.py`, `prepare_data.py`, `convert_dataset.py`, `build_profile.py` | Training pipeline |
| `score_testset.py` | Scoring a test-set folder |
| `member_handwriting/` | Team handwriting used for training |
| `tests/` | Automated tests (`pytest`) |
| `results/` | Metrics and confusion matrix from the last training |
| `info/` | Guide and presentation notes |

## Team

TIP Manila, CIT 401A: Asilo, Enriquez, Guinto, Tania.
