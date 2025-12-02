import os
import base64
import tempfile

import numpy as np
import librosa

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =====================
# Fingerprint parameters
# =====================
TARGET_SR = 22050
HOP_LENGTH = 512
N_MELS = 64

# =====================
# Load fingerprints.npz
# =====================
print("Loading fingerprints.npz ...")
data = np.load("fingerprints.npz", allow_pickle=True)

FEATURE_KEYS = ["feats", "fingerprints", "features", "fps"]
FEATS = None
USED_KEY = None

for k in FEATURE_KEYS:
    if k in data:
        FEATS = data[k]
        USED_KEY = k
        break

if FEATS is None:
    raise KeyError(
        "fingerprints.npz missing feature array "
        "(expected one of: feats, fingerprints, features, fps)"
    )

if "names" in data:
    NAMES = data["names"].tolist()
else:
    NAMES = [f"Track {i}" for i in range(FEATS.shape[0])]

print(f"Loaded fingerprints from key '{USED_KEY}'")
print(f"Loaded {FEATS.shape[0]} fingerprints from fingerprints.npz")

# Make sure FEATS is 2D: (num_tracks, feature_dim)
FEATS = np.asarray(FEATS)
if FEATS.ndim != 2:
    raise ValueError(f"FEATS must be 2D, got shape {FEATS.shape}")

# =====================
# Helper: fingerprint for an audio file
# =====================

def make_fingerprint(path: str) -> np.ndarray:
    """
    Load audio from 'path', compute a simple log-mel mean feature vector.
    Returns: np.ndarray of shape (N_MELS,)
    """
    # librosa will try SoundFile first, then audioread (which can use ffmpeg, etc.)
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio after loading")

    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
    )
    logmel = librosa.power_to_db(mels + 1e-10)
    feat = logmel.mean(axis=1)  # (N_MELS,)
    return feat


def best_match(query_feat: np.ndarray):
    """
    Compare query_feat to all FEATS using cosine similarity.
    Returns: (best_index, similarity_score)
    """
    # Normalize
    q = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    f = FEATS / (np.linalg.norm(FEATS, axis=1, keepdims=True) + 1e-8)

    sims = f @ q  # dot products -> cosine similarities
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return idx, score


# =====================
# Health check
# =====================

@app.route("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "tracks": len(NAMES),
        }
    )


# =====================
# /match endpoint
# =====================

@app.route("/match", methods=["POST"])
def match():
    """
    Accepts audio either as:
      1) JSON:  {"audio": "data:audio/webm;base64,AAA..."}
      2) multipart/form-data:  file field named "audio"

    Returns JSON:
      { "match": "<track name>", "score": 0.92 }
    """
    raw_bytes = None

    # --------- Try JSON ----------
    if request.is_json:
        data = request.get_json(silent=True) or {}
        audio_b64 = data.get("audio")

        if not audio_b64:
            return jsonify({"error": "Missing 'audio' field in JSON"}), 400

        try:
            # If it's a data URL, strip the header: "data:...;base64,XXXX"
            if "," in audio_b64:
                audio_b64 = audio_b64.split(",", 1)[1]
            raw_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 audio: {e}"}), 400

    # --------- Try multipart/form-data ----------
    else:
        # We expect a file input named "audio"
        if "audio" not in request.files:
            return jsonify(
                {"error": "No 'audio' file found in multipart/form-data"}
            ), 400
        file = request.files["audio"]
        raw_bytes = file.read()

    if not raw_bytes:
        return jsonify({"error": "Empty audio payload"}), 400

    # --------- Save to temp .webm and fingerprint ----------
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        query_feat = make_fingerprint(tmp_path)
        idx, score = best_match(query_feat)

        return jsonify(
            {
                "match": NAMES[idx],
                "score": score,
            }
        )

    except Exception as e:
        # Any processing error -> 500
        return jsonify({"error": f"Failed to process audio: {e}"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # For local testing only; Render uses gunicorn with: gunicorn app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
