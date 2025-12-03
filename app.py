import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # 🔴 turn off numba JIT globally

import io
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa


# === Audio / fingerprint settings ===
TARGET_SR = 22050
N_MELS = 64
SIM_THRESHOLD = 0.75  # tweak if needed

app = Flask(__name__)
CORS(app)

# === Load fingerprints once on startup ===

F = None           # feature matrix (num_tracks, feat_dim)
TRACK_NAMES = []   # list of track names

try:
    data = np.load("fingerprints.npz", allow_pickle=True)

    feat_key = None
    for k in ("feats", "fingerprints", "features", "fps"):
        if k in data.files:
            feat_key = k
            break

    if feat_key is None:
        raise KeyError(
            "fingerprints.npz missing feature array "
            "(expected one of: feats, fingerprints, features, fps)"
        )

    F = data[feat_key].astype(np.float32)

    name_key = None
    for k in ("names", "track_names", "labels"):
        if k in data.files:
            name_key = k
            break

    if name_key is not None:
        TRACK_NAMES = [str(x) for x in data[name_key].tolist()]
    else:
        TRACK_NAMES = [f"Track {i}" for i in range(F.shape[0])]

    print(f"Loaded fingerprints from key '{feat_key}'")
    print(f"Loaded {len(TRACK_NAMES)} fingerprints from fingerprints.npz")

except Exception as e:
    print("Failed to load fingerprints:", e)
    F = None
    TRACK_NAMES = []


# === Helpers ===

def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode WAV (or other supported format) from bytes,
    compute a log-mel spectrogram, average over time,
    and L2-normalize to get a vector.
    """
    bio = io.BytesIO(audio_bytes)
    # librosa can load from file-like objects
    y, sr = librosa.load(bio, sr=TARGET_SR, mono=True)

    if y.size == 0:
        raise ValueError("Empty decoded audio")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    feat = log_mel.mean(axis=1).astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm

    return feat


def find_best_match(query_fp: np.ndarray):
    """
    Cosine similarity between query and all stored fingerprints.
    F is assumed L2-normalized per-row.
    """
    if F is None or len(TRACK_NAMES) == 0:
        return None, 0.0

    # ensure shape (1, D)
    q = query_fp.reshape(1, -1)

    # cosine similarity = dot product because all vectors are normalized
    sims = (F @ q.T).ravel()  # (num_tracks,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = TRACK_NAMES[best_idx]

    return best_name, best_score


# === Routes ===

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "loaded": F is not None,
        "tracks": len(TRACK_NAMES),
    })


@app.route("/match", methods=["POST"])
def match():
    """
    Accepts:
      - Raw audio bytes with Content-Type: audio/wav (what we'll use from scanner.html)
      - OR legacy multipart/form-data with a file field named 'audio'
    """
    try:
        ct = request.content_type or ""
        audio_bytes = None

        # New path: raw bytes (mobile/desktop both)
        if ct.startswith("audio/"):
            audio_bytes = request.data
        else:
            # Legacy path: multipart upload with <input name="audio">
            file = request.files.get("audio")
            if file:
                audio_bytes = file.read()

        if not audio_bytes:
            print(f"/match 400: no audio bytes, content_type={ct}")
            return jsonify({"error": "no_audio"}), 400

        print(f"/match received {len(audio_bytes)} bytes, content_type={ct}")

        try:
            query_fp = make_fingerprint_from_bytes(audio_bytes)
        except Exception as e:
            print("Error decoding audio:", e)
            return jsonify({"error": "decode_failed"}), 400

        name, score = find_best_match(query_fp)

        if name is None:
            return jsonify({
                "ok": True,
                "match": None,
                "confidence": 0.0,
                "threshold": SIM_THRESHOLD,
                "is_confident": False,
            })

        resp = {
            "ok": True,
            "match": name,
            "confidence": score,
            "threshold": SIM_THRESHOLD,
            "is_confident": bool(score >= SIM_THRESHOLD),
        }
        return jsonify(resp)

    except Exception:
        print("Error in /match:", traceback.format_exc())
        return jsonify({"error": "server_error"}), 500


if __name__ == "__main__":
    # For local testing
    app.run(debug=True, host="0.0.0.0", port=5000)
