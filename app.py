"""Frontend server:  python app.py  ->  http://127.0.0.1:5000
To open it from a phone on the same Wi-Fi:  python app.py --host 0.0.0.0  ->  http://<laptop-ip>:5000"""
import argparse, base64, os, time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from common import LETTERS, LOOKUP, LOOKUP_ASCII, NAMES, load_image
from predict import predict_with_view, load_model

# "Save as test photo" puts photos here, named <class>_<name>_<time>.jpg (TEST ONLY, never train on them)
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_photos", "random_test_photos")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
load_model()


def file_label(i):
    """Class index -> the start of a test-photo file name, e.g. 1 -> '01_be'."""
    return f"{i:02d}_" + NAMES[i].split(" ")[0] + ("2" if i == 30 else "")   # he (jimi) vs he2 (do-cheshm)


def _png_data_url(view):
    big = cv2.resize((view * 255).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", big)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def home():
    letters = [{"id": i, "letter": LETTERS[i], "latin": LOOKUP[i], "name": NAMES[i]} for i in range(32)]
    return render_template("index.html", letters=letters)


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded."), 400
    img = load_image(f.read())
    if img is None:
        return jsonify(error="That file is not a readable image."), 400
    try:
        res, view = predict_with_view(img)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(results=[{"id": i, "letter": LETTERS[i], "latin": LOOKUP[i], "ascii": LOOKUP_ASCII[i],
                             "name": NAMES[i], "confidence": round(c, 3)} for i, c in res],
                   view=_png_data_url(view))


@app.post("/save")
def save():
    """Keep a photo as a labelled test image: the person says which letter it really is."""
    f = request.files.get("image")
    try:
        cls = int(request.form.get("class_id", ""))
    except ValueError:
        cls = -1
    if not f or not 0 <= cls < 32:
        return jsonify(error="Need an image and a letter (class_id 0-31)."), 400
    img = load_image(f.read())
    if img is None:
        return jsonify(error="That file is not a readable image."), 400
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{file_label(cls)}_{stamp}.jpg"
    n = 1
    while os.path.exists(os.path.join(SAVE_DIR, name)):        # two saves in the same second
        n += 1; name = f"{file_label(cls)}_{stamp}-{n}.jpg"
    if img.ndim == 3 and img.shape[2] == 4:                    # transparent PNG: put it on white paper
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    cv2.imwrite(os.path.join(SAVE_DIR, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])   # phone rotation already applied
    return jsonify(saved=name, folder="real_photos/random_test_photos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=False)
