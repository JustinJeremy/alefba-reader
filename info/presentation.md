# Alefba Reader: prototype presentation notes

**What it does:** you give it a photo of a Persian letter, and it tells you which letter it is and how
to spell it in English (for example غ → `gh`). The spellings follow the instructor's alphabet chart.
It works with printed and handwritten letters, coloured letters, and letters circled on a full
alphabet sheet.

---

## Technique: how it works, in two parts

**1. Image processing: cleaning the photo** (with OpenCV)
A real photo has shadows, notebook lines, fingers and background. The app removes all of that, so
only the letter is left, in black and white.

**2. A CNN: recognising the letter** (with PyTorch)
A **CNN (Convolutional Neural Network)** is a type of AI that is good at recognising pictures.
We **trained** it by showing it about **60,000 example letters** with the right answer for each one,
until it learned what each letter looks like. This is called **supervised learning**: learning from
examples that have the answers.

Three extra tricks:
- **Data augmentation:** while it learns, we slightly tilt, move, resize and blur each example, so it
  still recognises letters in messy real photos.
- **A "not a letter" answer:** it can say "this isn't a Persian letter" (an empty page, an object, English
  text) instead of guessing.
- **A lookup table:** once it knows the letter, a simple table gives the English spelling.

## Algorithm: what happens to a photo, step by step

1. **Resize:** very big photos are made smaller (faster) and very small ones bigger (so the dots don't
   disappear).
2. **Fix the lighting:** shadows and dark corners are evened out.
3. **Make it black and white:** every pixel becomes either "ink" or "paper" (Otsu's method picks the
   best dividing point automatically).
4. **Remove clutter:** notebook lines, frames, the edge of the desk, fingers and small printed text.
5. **Find the letter:** the ink shapes are grouped, and each letter body is joined with its dots.
   - On a full alphabet sheet, it looks for the letter you circled, highlighted or pointed at.
   - If there are several letters, it reads each one, right to left.
6. **Make it standard size:** the letter is cut out, centred and shrunk to a 64 × 64 pixel picture.
7. **Ask the CNN:** the CNN gives a score for each of the 33 answers (32 letters plus "not a letter"),
   and the highest score wins.
8. **Double-check:** if it isn't sure, it says so or asks for a better photo. Otherwise it shows the
   letter, the spelling and how sure it is (for example 97%).

**Inside the CNN (if asked):**
- It looks at the 64 × 64 picture through **4 layers**. Each layer finds patterns: first simple ones
  (edges, curves), then bigger ones (loops, dots, tails).
- The last part combines those patterns into one of 33 answers.
- It has about 774,000 numbers (weights) that it adjusted during training.

**How it was trained (if asked):**
- It went through all the examples 30 times.
- 80% of the images were used to learn, and 10% to check progress. The last 10% were kept hidden
  until the end for the final score.
- One font (Calibri) was never shown to it, to prove it can read a font it has never seen.

**Training examples:** 32,000 handwritten letters (a public dataset), 14,874 printed letters in 37
fonts, 5,384 "not a letter" pictures, 7,650 letters made harder on purpose (lines, grain, fading),
and our team's own handwriting.

## Software

| Software | What we used it for |
|---|---|
| **Python** | the programming language everything is written in |
| **PyTorch** | building and training the CNN (the AI part) |
| **OpenCV** | cleaning the photo (lighting, lines, finding the letter) |
| **NumPy** | handling the image as numbers |
| **scikit-learn** | splitting the examples and measuring accuracy |
| **Flask** | the web app, which opens in a browser on a laptop or a phone |
| **HTML, CSS, JavaScript** | the web page itself (upload, camera, buttons) |
| **pytest** | automatic tests that check nothing broke after each change |
| **Git, GitHub, VS Code** | saving versions, sharing the code with the team, and writing the code |

## Results

| Test | Score |
|---|---|
| Handwritten / printed / a font it never saw | 98.8% / 97.8% / 97.4% |
| Our team's real handwriting photos (never used for training) | 109 of 110 (99.1%) |
| Full alphabet sheets with one letter marked | 100% |
| Extra test set (coloured, pastel, cartoon and Arabic letters) | 277 of 296 (93.6%) |
| Speed | under half a second per photo (about 1 second for a full sheet) |

**Limits:**
- The letter must be the right way up. Turned around, some letters look like other letters (پ and ث).
- It reads separate letters, not joined-up words.
