# Alefba Reader: prototype presentation notes

Reads a Persian letter from a photo (printed, handwritten, coloured, or marked on an alphabet sheet)
and gives its English spelling from the instructor's alphabet chart (e.g. غ → `gh`, ق → `q`).

---

## Technique

- **Image processing** (computer vision) cleans the photo so that only the letter is left.
- **Deep learning**: a **Convolutional Neural Network (CNN)** classifies the cleaned letter.
  It is *supervised learning*: the network learns from about 60,000 labelled letter images.
- **Data augmentation**: during training, every image is randomly turned, moved and resized a little,
  and blurred. We also made extra training copies with notebook lines, film grain and fading, so the
  model copes with real photos.
- **Rejecting non-letters**: the network has a 33rd answer, *"not a letter"*, and each answer is compared
  with that letter's average look, so blank photos, objects and Latin text are refused instead of guessed.
- **Lookup table**: the predicted letter is turned into its romanised spelling.

## Algorithm (what happens to a photo)

1. **Load and resize**: very large photos are shrunk, very small ones enlarged (so dots survive).
2. **Even out the lighting**: the paper brightness is estimated and divided out (removes shadows).
3. **Black and white**: Otsu thresholding separates ink from paper.
4. **Remove clutter**: ruled notebook lines (straight, slanted or dashed), frames, desk edges, fingers,
   captions and the notebook's printed ticks.
5. **Find the letter(s)**: connected-component analysis groups each letter body with its dots.
   - a circled, highlighted or arrowed letter on a sheet is found first;
   - a photo of 2–20 separate letters is split and read right to left;
   - on a cartoon card, each strongly coloured shape is tried on its own.
6. **Normalise**: the letter is cropped, centred and scaled to **64 × 64 pixels**.
7. **Classify with the CNN** (below). The answer with the highest probability wins.
8. **Check and answer**: the app refuses the photo if it's unsure or the shape doesn't look like any
   letter. Otherwise it shows the letter, its spelling, how sure it is, and the next best guesses.

**CNN architecture** (about 774,000 parameters)

| Layer | Output |
|---|---|
| Input: cleaned letter | 1 × 64 × 64 |
| 4 blocks of *Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2* (32, 64, 128, 128 filters) | 128 × 4 × 4 |
| Fully connected 256 + ReLU + Dropout 0.4 | 256 |
| Fully connected + Softmax | 33 (32 letters + "not a letter") |

**Training**: Adam optimiser (learning rate 0.001, cosine schedule), cross-entropy loss, 30 epochs,
batches of 128, 80/10/10 train/validation/test split. One font family (Calibri) is never trained on,
to test on an unseen font.

**Training data**: 32,000 handwritten letters (Yousefnezhad dataset), 14,874 printed letters in 37
fonts, 5,384 "not a letter" images, 7,650 copies with lines/grain/fading, and the team's own handwriting.

## Software

| Software | Used for |
|---|---|
| **Python 3.14** | programming language |
| **PyTorch** + torchvision | building and training the CNN, data augmentation |
| **OpenCV** | all the image processing (lighting, thresholding, line removal, finding letters) |
| **NumPy** | image arrays and calculations |
| **scikit-learn** | splitting the data, accuracy and confusion matrix |
| **Flask** | the web app (works on a laptop and on a phone over Wi-Fi) |
| HTML, CSS, JavaScript | the web page (upload, camera, drag a box around a letter) |
| **pytest** | the automated tests |
| Git + GitHub, VS Code | version control and editing |

## Results (short)

| Test | Score |
|---|---|
| Handwritten / printed / unseen font (held-out test images) | 98.8% / 97.9% / 96.9% |
| Team members' real handwriting photos (never trained on) | 108/110 (98.2%) |
| Marked alphabet sheets (rehearsal) | 100% |
| BUS testing dataset (coloured, pastel, cartoon, Arabic) | 273/296 (92.2%) |
| Speed | about 0.1–0.3 s per photo on a laptop (about 1 s for a whole marked sheet) |

**Limits**: letters must be upright (turning a letter around can make it a different letter, such as
پ and ث). The app reads separate letters, not joined-up words.
