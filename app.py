import os
# Disable numba JIT on Render (prevents heavy compilation + timeouts)
os.environ["NUMBA_DISABLE_JIT"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa

from bgm_list import TRACKS  # list of {"name": ..., "file": ...}

# -------------------------------------------------
# Config
# -------------------------------------------------
TARGET_SR = 22050          # sample rate used for both DB and query
NPZ_PATH = "fingerprints.npz"

# -------------------------------------------------
# Load fingerprints at startup
# -------------------------------------------------
if not os.path.exists(NPZ_PATH):
    raise RuntimeError(f"{NPZ_PATH} not found. Run build_fingerprints.py first.")

data = np.load(NPZ_PATH, allow_pickle=True)

# Names
if "names" not in data.files:
    raise KeyError("fingerprints.npz missing 'names' array")
NAMES = data["names"]

# Feature matrix: try several key names to be safe
_feat_key = None
for k in ("feats", "fingerprints", "features"):
    if k in data.files:
        _feat_key = k
        break
if _feat_key is None:
    raise KeyError("fingerprints.npz missing feature array (expected one of: feats, fingerprints, features)")

FINGERPRINTS = data[_feat_key]  # shape (N_tracks, feature_dim)

# FPS (feature dimension) – mainly for sanity check
if "fps" in data.files:
    FPS = int(data["fps"])
else:
    FPS = FINGERPRINTS.shape[1]

if FINGERPRINTS.shape[0] != len(NAMES):
    raise ValueError(
        f"Mismatch: FINGERPRINTS has {FINGERPRINTS.shape[0]} rows "
        f"but NAMES has {len(NAMES)} entries."
    )

# -------------------------------------------------
# Fingerprint function (must match build_fingerprints.py logic)
# -------------------------------------------------
def make_fingerprint(path: str) -> np.ndarray:
    """
    Load a short audio file, compute MFCC-based fingerprint.
    This MUST be the same logic used in build_fingerprints.py.
    """
    # Limit duration so we don't do too much work per request (good for Render)
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True, duration=5.0)

    if y.size == 0:
        raise ValueError("Empty audio after loading")

    # Example MFCC-based embedding: (same as in build_fingerprints.py)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    fp = np.mean(mfcc, axis=1)  # (20,)

    # Normalize to unit length to use cosine similarity
    norm = np.linalg.norm(fp)
    if norm > 0:
        fp = fp / norm

    return fp  # shape (feature_dim,)


def match_fingerprint(query_fp: np.ndarray):
    """
    Compare query fingerprint against all known FINGERPRINTS using cosine similarity.
    Returns (best_name, best_score).
    """
    if query_fp.ndim != 1:
        raise ValueError("query_fp must be 1D")

    # Ensure FINGERPRINTS rows are unit-normalized (if not already)
    # (We can pre-normalize here once)
    global FINGERPRINTS
    norms = np.linalg.norm(FINGERPRINTS, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    FINGERPRINTS = FINGERPRINTS / norms

    # Cosine similarity = dot product (since all are normalized)
    scores = FINGERPRINTS @ query_fp  # shape (N_tracks,)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_name = str(NAMES[best_idx])

    return best_name, best_score

# -------------------------------------------------
# Flask app
# -------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/match", methods=["POST"])
def match():
    """
    Expects multipart/form-data with file field name 'audio'.
    Audio is a short recording (WAV/WEBM->WAV on frontend), we match it to the DB.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    f = request.files["audio"]

    # Save to temp file
    tmp_path = "/tmp/upload_audio.wav"
    f.save(tmp_path)

    try:
        query_fp = make_fingerprint(tmp_path)
        best_name, best_score = match_fingerprint(query_fp)

        # You can tune this threshold; e.g. require score > 0.6 to be "confident"
        CONF_THRESHOLD = 0.55
        if best_score < CONF_THRESHOLD:
            return jsonify({
                "match": None,
                "score": best_score,
                "message": "No confident match"
            })

        return jsonify({
            "match": best_name,
            "score": best_score
        })

    except Exception as e:
        # For debugging – feel free to print(e) as well
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # Local dev only; on Render, gunicorn runs app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
