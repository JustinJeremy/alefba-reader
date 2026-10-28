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
- *Turn* on the photo gives it a quarter turn (and reads it again if it was read): for a sideways or
  upside-down letter the app is unsure of.
- On a whole alphabet sheet with no mark, the app asks for one, or you can drag a box around the letter.
- The answer shows the letter, its spelling (with a *Copy* button), how sure the app is, the other possible
  letters (folded until opened), and how the letter looks inside a word (at the start, in the middle, at the end).
- *This session* lists every letter read since the page was opened; click one to see it again, or ✕ to clear it.
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
5. *Turned letters* (`predict.py`): ruled paper photographed at a tilt is straightened along its lines
   (read both ways up, since the lines do not show which way is up); a second model trained on turned
   letters (`persian_cnn_tilt.pt`) also reads one-letter photos the main model is unsure of, and the
   surer answer wins. A letter on plain paper that is still read unsurely is also read turned to other
   angles, and a clearly surer reading wins.
6. *Tight crops* (`predict.py`): a letter cropped with no white space round it (it touches the photo's
   edges) is also read with a border of its own paper colour, when the first reading is unsure.
7. *Refusals* (`predict.py`): photos that are not a Persian letter get "No Persian letter found".
8. *Lookup*: the letter is turned into its spelling, following the instructor's chart.

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
| BUS testing dataset (`bus_test_set`) | 282/296 (95.3%); whole picture cards 22/28 |
| Turned letters, 15–345° (`turned_set`) | 1066/1390 (77%), was 832 before batch 20 |
| Tightly cropped letters, no white space (`tight_set/00`) | 133/139 (95.7%), was 128 before batch 17 |
| Phone-style photos (`phone_set`: slanted, blurred, uneven light, compressed) | mild 122/139, strong 89/139; marked sheets 11/21, several letters 19/25 |
| Speed | ~3 ms per image (model only); ~0.1–0.3 s per photo in the app |

Tilted and sideways letters are read too, always with every upright score unchanged. Ruled paper
photographed at a tilt is straightened along its lines first (batch 16), and a second model, trained on
turned letters, is asked when the main model is unsure. Batch 20 fixed the straightening: turning a
photo back smeared its edges into slanted streaks, a page is now read both ways up, and one ruled line
running edge to edge is enough. It also made `turned_set` fair: each turned photo keeps all of its
letter (before, cropping the corners cut parts of letters off at a slant). On the old cropped set the
same photos went from 825 to 965 right; on the fair set from 832 to 1,043. Batch 21 added two more
steps: a letter on plain paper is also read turned to other angles, and ruling is found past a dark
desk edge (1,066).

| Angle | 15° | 30° | 45° | 90° | 135° | 180° | 270° | 315° | 330° | 345° |
|---|---|---|---|---|---|---|---|---|---|---|
| Right (of 139) | 118 | 112 | 106 | 109 | 95 | 88 | 105 | 109 | 110 | 114 |

Upside down stays hardest: turned around, some letters *are* other letters (پ and ث, ب and ن, ل and م),
so a level page is not read upside down as well (it misread upright lam as mim); *Turn* on the photo
is the way round that. Mirrored letters are not read.

## Online copy

Try it at <https://alefba-reader.onrender.com> (Render's free plan: the first visit after 15 minutes
without visitors takes about a minute, and each photo takes a few seconds).

`render.yaml`, `requirements-server.txt` and `numpy_cnn.py` run the app there without PyTorch and with
saving turned off. Every merge to `main` updates the website by itself. Setup: [`info/guide.md`](info/guide.md), section 6.

## Testing

```
python -m pytest                        (all automated tests, ~4 min, expect 419 passed)
python score_testset.py rehearsal_set
python score_testset.py hard_set
python score_testset.py real_photos
python score_testset.py marked_set
python score_testset.py multi_set
python score_testset.py bus_test_set
python make_turned_set.py               (then: python score_testset.py turned_set)
python make_tight_set.py                (then: python score_testset.py tight_set)
python make_phone_set.py                (then: python score_testset.py phone_set)
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
| `turned_set/` | The one-letter test photos turned to 10 angles, the whole photo kept (made by `make_turned_set.py`, not in git) |
| `tight_set/` | The same photos cropped tightly round the letter: 0, 2 and 5% margin (`make_tight_set.py`, not in git) |
| `phone_set/` | The test photos as a phone might take them: slanted, blurred, unevenly lit, compressed (`make_phone_set.py`, not in git). A stand-in: real phone photos are the real test |

## Retraining (optional)

Only needed after changing the training data or the model. The training images (`data/`,
1.4 GB) are not on GitHub.

| Step | Command |
|---|---|
| 1. Handwriting data | `python convert_dataset.py <folder with Random1..Random10>` → `data/handwritten/<0..31>/` |
| 2. Printed + "not a letter" data | `python prepare_data.py` (fonts in `fonts/` plus the Windows Arabic-script fonts) |
| 3. Train (~2 h on CPU) | `python train.py` → `persian_cnn.pt`, `letter_profile.npz`, `results/` |
| 3b. Train the tilt model (~1 h on the GPU) | `python train.py --tilt` → `persian_cnn_tilt.pt`, `letter_profile_tilt.npz`, `results_tilt/` |
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
| `numpy_cnn.py`, `persian_cnn.npz` | The model run with NumPy only, for the online copy |
| `persian_cnn_tilt.pt`, `letter_profile_tilt.npz`, `persian_cnn_tilt.npz`, `results_tilt/` | The tilt model (turned letters) |
| `make_turned_set.py`, `make_tight_set.py`, `make_phone_set.py` | Make `turned_set/`, `tight_set/` and `phone_set/`: turned, tightly cropped and phone-style test photos |
| `render.yaml`, `requirements-server.txt`, `.python-version` | Settings for putting it online on Render |
| `score_testset.py` | Scoring a test-set folder |
| `member_handwriting/` | Team handwriting used for training |
| `tests/` | Automated tests (`pytest`) |
| `results/` | Metrics and confusion matrix from the last training |
| `info/` | Guide and presentation notes |

## Team

TIP Manila, CIT 401A: Asilo, Enriquez, Guinto, Tania.
