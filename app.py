"""Frontend server:  python app.py  ->  http://127.0.0.1:5000
To open it from a phone on the same Wi-Fi:  python app.py --host 0.0.0.0  ->  http://<laptop-ip>:5000"""
import argparse, base64, os, re, socket
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from common import LETTERS, LOOKUP, LOOKUP_ASCII, NAMES, load_image
from predict import read_photo, load_model

# "Save as test photo" puts photos in a folder inside real_photos/, named <class>_<name>_<n>.jpg
# (TEST ONLY, never train on them). Any folder made inside real_photos/ also shows up in the list.
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_photos")
DEFAULT_FOLDERS = ["random_test_photos", "multi_letter_photos", "JJ", "Ethan", "Mark", "Kaye"]
MULTI_FOLDER = "multi_letter_photos"      # chosen automatically for photos of several letters

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
load_model()


@app.errorhandler(413)
def too_big(_):
    """A photo over the upload limit: answer with a message the page can show (not an HTML page)."""
    mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify(error=f"This photo is too big (over {mb} MB). Take it again at a lower resolution "
                         "or send a smaller copy."), 413


def file_label(i):
    """Class index -> the start of a test-photo file name, e.g. 1 -> '01_be'."""
    return f"{i:02d}_" + NAMES[i].split(" ")[0] + ("2" if i == 30 else "")   # he (jimi) vs he2 (do-cheshm)


def next_free_name(folder, cls):
    """Next numbered file name: 01_be_1.jpg, 01_be_2.jpg, ... for one letter (cls = 1), or
    06-05_1.jpg, 06-05_2.jpg, ... for several letters in reading order (cls = [6, 5]).
    Continues after the highest number already in the folder, so nothing is overwritten."""
    label = file_label(cls) if isinstance(cls, int) else "-".join(f"{c:02d}" for c in cls)
    pattern = re.compile(re.escape(label) + r"_(\d+)\.jpg$", re.IGNORECASE)
    taken = [int(m.group(1)) for f in os.listdir(folder) if (m := pattern.match(f))]
    return f"{label}_{max(taken, default=0) + 1}.jpg"


def save_folders():
    """Folders the app may save into: the team's folders first, then any others in real_photos/."""
    found = sorted(d for d in os.listdir(PHOTOS_DIR) if os.path.isdir(os.path.join(PHOTOS_DIR, d))
                   and not d.startswith((".", "_"))) if os.path.isdir(PHOTOS_DIR) else []
    return DEFAULT_FOLDERS + [d for d in found if d not in DEFAULT_FOLDERS]


def _data_url(img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    return f"data:image/{ext[1:]};base64," + base64.b64encode(buf.tobytes()).decode()


def _found_here(img, boxes, side=360):
    """The user's photo, shrunk, with a green box around each part that was read (numbered when
    there are several letters, in reading order)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    k = min(1.0, side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img.copy()
    t = max(2, round(max(small.shape[:2]) / 150))
    for n, box in enumerate(boxes, 1):
        x, y, w, h = (int(round(v * k)) for v in box)
        pad = max(3, int(0.06 * max(w, h)))
        cv2.rectangle(small, (x - pad, y - pad), (x + w + pad, y + h + pad), (63, 95, 31), t, cv2.LINE_AA)    # forest green (BGR)
        if len(boxes) > 1:
            org = (x - pad, max(14, y - pad - 4))
            cv2.putText(small, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(small, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (63, 95, 31), 1, cv2.LINE_AA)
    return _data_url(small, ".jpg")


def _cleaned(view, side=192):
    """The model's input, enlarged smoothly so it reads as a letter rather than blocky pixels."""
    return _data_url(cv2.resize((view * 255).astype(np.uint8), (side, side), interpolation=cv2.INTER_CUBIC))


@app.errorhandler(500)
def crashed(e):
    """Any unexpected error: the details go to the terminal, the page gets a short message."""
    app.logger.exception(getattr(e, "original_exception", e))
    return jsonify(error="The app hit an unexpected problem with this photo (details are in the terminal "
                         "running app.py). Try another photo, or drag a box around one letter."), 500


@app.get("/")
def home():
    letters = [{"id": i, "letter": LETTERS[i], "latin": LOOKUP[i], "name": NAMES[i]} for i in range(32)]
    return render_template("index.html", letters=letters, folders=save_folders())


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded."), 400
    img = load_image(f.read())
    if img is None:
        return jsonify(error="That file is not a readable image."), 400
    try:
        select = _selection(request.form)
        letters, how = read_photo(img, select=select)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    out = [{"results": _results(res), "view": _cleaned(view)} for res, (view, _) in letters]
    # results / view: the (first) letter, as before; letters: every letter, in reading order (right to left)
    return jsonify(results=out[0]["results"], view=out[0]["view"], letters=out,
                   found=_found_here(img, [box for _, (_, box) in letters]), how=how)


def _results(res):
    return [{"id": i, "letter": LETTERS[i], "latin": LOOKUP[i], "ascii": LOOKUP_ASCII[i],
             "name": NAMES[i], "confidence": round(c, 3)} for i, c in res]


def _selection(form):
    """Optional box the user dragged on the photo: sel_x, sel_y, sel_w, sel_h in image pixels."""
    keys = ("sel_x", "sel_y", "sel_w", "sel_h")
    if not any(form.get(k) for k in keys):
        return None
    try:
        return tuple(int(float(form.get(k, ""))) for k in keys)
    except ValueError:
        raise ValueError("The selected box is not valid. Clear the selection and drag again.")


@app.post("/save")
def save():
    """Keep a photo as a labelled test image: the person says which letter it really is
    (class_id), or, for a photo of several letters, which letters they are in reading order
    (class_ids, e.g. "6,5")."""
    f = request.files.get("image")
    try:
        if request.form.get("class_ids"):
            cls = [int(v) for v in request.form["class_ids"].split(",")]
            ok = 2 <= len(cls) <= 20 and all(0 <= c < 32 for c in cls)
        else:
            cls = int(request.form.get("class_id", ""))
            ok = 0 <= cls < 32
    except ValueError:
        ok = False
    if not f or not ok:
        return jsonify(error="Need an image and its letter(s) (class ids 0-31)."), 400
    folder = request.form.get("folder", DEFAULT_FOLDERS[0])
    if folder not in save_folders():                           # only listed folders: no writing elsewhere
        return jsonify(error="Unknown folder. Choose one from the list."), 400
    img = load_image(f.read())
    if img is None:
        return jsonify(error="That file is not a readable image."), 400
    target = os.path.join(PHOTOS_DIR, folder)
    os.makedirs(target, exist_ok=True)
    name = next_free_name(target, cls)
    if img.ndim == 3 and img.shape[2] == 4:                    # transparent PNG: put it on white paper
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    cv2.imwrite(os.path.join(target, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])   # phone rotation already applied
    return jsonify(saved=name, folder=f"real_photos/{folder}")


def laptop_address():
    """This computer's address on the Wi-Fi, for opening the app from a phone (no data is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    if a.host == "0.0.0.0" and laptop_address():
        print(f"\n  On this laptop, open:              http://127.0.0.1:{a.port}   (live camera works here)"
              f"\n  On your phone (same Wi-Fi), open:  http://{laptop_address()}:{a.port}   (Take photo opens the phone camera)\n")
    app.run(host=a.host, port=a.port, debug=False)
