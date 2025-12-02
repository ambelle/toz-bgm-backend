import os
import io
import numpy as np
import soundfile as sf
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
    raise KeyError(
        "fingerprints.npz missing feature array "
        "(expected one of: feats, fingerprints, features, fps)"
    )

FEATS = data[feature_key]           # shape: (num_tracks, feature_dim)
NAMES = data["names"].tolist()      # list of track names

print(f"Loaded fingerprints from key '{feature_key}'")
print(f"Loaded {len(NAMES)} fingerprints from fingerprints.npz")

# Normalize stored fingerprints & precompute norms for cosine similarity
FEATS = np.asarray(FEATS, dtype=np.float32)
FEATS_NORMS = np.linalg.norm(FEATS, axis=1) + 1e-9


# ==== Fingerprint helper ====

def fingerprint_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Given a mono PCM array y and sampling rate sr,
    convert to the same fingerprint representation as in build_fingerprints.py.
    """
    if y.size == 0:
        raise ValueError("Empty audio")

    # Resample if needed to TARGET_SR
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Same style as build_fingerprints.py: chroma_cqt -> mean over time
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    fp = chroma.mean(axis=1).astype(np.float32)  # shape (12,)
    # L2-normalize
    fp_norm = np.linalg.norm(fp) + 1e-9
    fp /= fp_norm
    return fp


def match_fingerprint(query_fp: np.ndarray):
    """
    Compare query_fp with all stored FEATS using cosine similarity.
    Return (best_name, best_score).
    """
    # FEATS are NOT normalized yet, so do dot / (norms * norm_q)
    q_norm = np.linalg.norm(query_fp) + 1e-9
    # But query_fp should already be unit norm; this is just safety
    sims = FEATS @ query_fp / (FEATS_NORMS * q_norm)

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = NAMES[best_idx]
    return best_name, best_score


# ==== Flask app ====

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/tracks", methods=["GET"])
def tracks():
    """List known track names."""
    return jsonify({"tracks": NAMES}), 200


@app.route("/match", methods=["POST"])
def match():
    """
    Expect raw audio/wav in the request body.

    Frontend:
      fetch('/match', {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wavBlob
      })

    Response JSON:
      {
        "status": "match",
        "name": "...",
        "score": 0.87
      }
      or
      {
        "status": "no_match",
        "best_name": "...",
        "best_score": 0.21
      }
    """

    raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"error": "no audio data"}), 400

    # Check content-type for debug
    ctype = request.content_type or ""
    print(f"/match received {len(raw)} bytes, content_type={ctype}")

    # Decode WAV from raw bytes using soundfile
    try:
        buf = io.BytesIO(raw)
        y, sr = sf.read(buf, dtype="float32")
    except Exception as e:
        print("Error decoding audio:", e)
        return jsonify({"error": "cannot decode audio"}), 400

    # If stereo, make mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Very short chunks are not useful
    if y.size < TARGET_SR * 0.5:  # < 0.5 sec
        return jsonify({"status": "too_short"}), 200

    # Build fingerprint
    try:
        q_fp = fingerprint_from_array(y, sr)
    except Exception as e:
        print("Error building fingerprint:", e)
        return jsonify({"error": "fingerprint_failed"}), 500

    # Compare with stored fingerprints
    best_name, best_score = match_fingerprint(q_fp)

    # Threshold: tweak as you like (0.35–0.5)
    THRESH = 0.4
    if best_score >= THRESH:
        return jsonify({
            "status": "match",
            "name": best_name,
            "score": round(best_score, 4)
        }), 200
    else:
        return jsonify({
            "status": "no_match",
            "best_name": best_name,
            "best_score": round(best_score, 4)
        }), 200


if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)
