"""Frontend server:  python app.py  ->  http://127.0.0.1:5000
To open it from a phone on the same Wi-Fi:  python app.py --host 0.0.0.0  ->  http://<laptop-ip>:5000"""
import argparse
from flask import Flask, jsonify, render_template, request
from common import LETTERS, LOOKUP, LOOKUP_ASCII, NAMES, load_image
from predict import predict_array, load_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
load_model()

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded."), 400
    img = load_image(f.read())
    if img is None:
        return jsonify(error="That file is not a readable image."), 400
    try:
        res = predict_array(img)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(results=[{"letter": LETTERS[i], "latin": LOOKUP[i], "ascii": LOOKUP_ASCII[i],
                             "name": NAMES[i], "confidence": round(c, 3)} for i, c in res])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, debug=False)