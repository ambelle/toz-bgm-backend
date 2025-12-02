import os
import numpy as np
import librosa

from flask import Flask, request, jsonify
from flask_cors import CORS

# ==== Audio / fingerprint settings ====
TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# ==== Load fingerprints.npz ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FP_PATH = os.path.join(BASE_DIR, "fingerprints.npz")

if not os.path.exists(FP_PATH):
    raise FileNotFoundError(f"fingerprints.npz not found at {FP_PATH}")

data = np.load(FP_PATH, allow_pickle=True)

feature_key = None
for k in ("feats", "fingerprints", "features", "fps"):
    if k in data:
        feature_key = k
        break

if feature_key is None:
    raise KeyError("fingerprints.npz missing feature array (expected one of: feats, fingerprints, features, fps)")

FEATS = data[feature_key]           # shape: (num_tracks, feature_dim)
NAMES = data["names"].tolist()      # list of track names

print(f"Loaded fingerprints from key '{feature_key}'")
print(f"Loaded {len(NAMES)} fingerprints from fingerprints.npz")

# Precompute norms for cosine similarity
FEATS = np.asarray(FEATS, dtype=np.float32)
NORMS = np.linalg.norm(FEATS, axis=1) + 1e-9

# ==== Helper to make fingerprint from an audio file ====
def make_fingerprint(path: str) -> np.ndarray:
    """
    Load audio from 'path' and compute a simple chroma-based fingerprint.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio")

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    fp = chroma.mean(axis=1)  # (12,)
    fp = fp.astype(np.float32)
    n = np.linalg.norm(fp) + 1e-9
    fp /= n
    return fp

# ==== Flask app ====
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/tracks", methods=["GET"])
def tracks():
    """Optional: list known track names."""
    return jsonify({"tracks": NAMES}), 200

# =============== TEMP DEBUG /match ===============
@app.route("/match", methods=["POST"])
def match():
    """
    TEMP DEBUG VERSION:

    Instead of trying to decode audio and match it, we just report what
    the server actually receives from the client.
    """

    content_type = request.content_type or ""
    raw = request.get_data(cache=False)

    info = {
        "content_type": content_type,
        "num_bytes": len(raw),
        "has_files": bool(request.files),
        "file_keys": list(request.files.keys()),
    }

    # You will see this JSON in the scanner "Result" box
    return jsonify(info), 200
# =================================================


if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
