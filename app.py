# app.py
import io
import os
import logging

import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------- CONFIG -----------------
TARGET_SR = 22050
N_MFCC = 20
THRESHOLD_CONFIDENT = 0.75  # on [0, 1] confidence scale

FINGERPRINTS_PATH = "fingerprints.npz"

# Globals
F = None              # (num_tracks, N_MFCC)
TRACK_NAMES = None    # list[str]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Flask app
app = Flask(__name__)
CORS(app)


# ----------------- FINGERPRINT UTILS -----------------
def load_fingerprints():
    global F, TRACK_NAMES

    if F is not None and TRACK_NAMES is not None:
        return

    logger.info(f"Loading fingerprints from {FINGERPRINTS_PATH}...")
    data = np.load(FINGERPRINTS_PATH, allow_pickle=True)

    feats = data["feats"].astype(np.float32)   # shape: (N, D)
    names = [str(x) for x in data["names"].tolist()]

    # L2-normalise rows (so dot product = cosine similarity)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    feats = feats / norms

    F = feats
    TRACK_NAMES = names

    logger.info(f"Loaded {F.shape[0]} fingerprints, dim={F.shape[1]}")


def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray | None:
    """
    Decode the incoming WAV bytes and make a normalised MFCC fingerprint.
    """
    # Load audio using librosa directly from bytes
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    except Exception as e:
        raise RuntimeError(f"librosa.load failed: {e}")

    if y is None or y.size == 0:
        raise RuntimeError("No audio data after decoding")

    # Resample to TARGET_SR
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Use only a small center window (~3s) from what the browser sends
    max_seconds = 3.0
    max_len = int(max_seconds * TARGET_SR)
    if y.shape[0] > max_len:
        start = (y.shape[0] - max_len) // 2
        y = y[start:start + max_len]

    # MFCC extraction (must match generator settings)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
        n_mels=64,
    )  # (N_MFCC, T)

    if mfcc.size == 0:
        raise RuntimeError("Empty MFCC matrix")

    v = mfcc.mean(axis=1).astype(np.float32)

    # L2-normalise
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise RuntimeError("Zero-norm MFCC vector")

    v /= norm
    return v


def cosine_best_match(F: np.ndarray, q_vec: np.ndarray):
    """
    Given:
      F: (N, D) L2-normalised
      q_vec: (D,) L2-normalised
    Returns:
      best_idx, best_sim (cosine similarity in [-1, 1])
    """
    sims = F @ q_vec  # (N,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    return best_idx, best_sim


def sim_to_confidence(sim: float) -> float:
    """
    Map cosine similarity [-1, 1] -> [0, 1] for a friendlier confidence.
    """
    conf = (sim + 1.0) / 2.0
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return conf


# ----------------- ROUTES -----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/match", methods=["POST", "OPTIONS"])
def match():
    if request.method == "OPTIONS":
        # CORS preflight handled by Flask-CORS; just respond 200
        return ("", 200)

    global F, TRACK_NAMES
    load_fingerprints()

    ct = request.content_type or ""
    if "audio/wav" not in ct:
        return jsonify({"ok": False, "error": "Expected Content-Type audio/wav"}), 400

    audio_bytes = request.get_data()
    logger.info(f"[MATCH] received {len(audio_bytes)} bytes, content_type={ct}")

    try:
        q_vec = make_fingerprint_from_bytes(audio_bytes)
    except Exception as e:
        logger.exception("[MATCH] Error decoding / fingerprinting audio")
        return jsonify({"ok": False, "error": f"decode_error: {e}"}), 400

    best_idx, best_sim = cosine_best_match(F, q_vec)
    conf = sim_to_confidence(best_sim)
    is_confident = conf >= THRESHOLD_CONFIDENT

    match_name = TRACK_NAMES[best_idx] if 0 <= best_idx < len(TRACK_NAMES) else "Unknown"

    logger.info(
        f"[MATCH] best='{match_name}' sim={best_sim:.3f} conf={conf:.3f} "
        f"threshold={THRESHOLD_CONFIDENT:.2f} confident={is_confident}"
    )

    return jsonify(
        {
            "ok": True,
            "match": match_name,
            "similarity": best_sim,
            "confidence": conf,
            "threshold": THRESHOLD_CONFIDENT,
            "is_confident": is_confident,
        }
    )


# ----------------- ENTRY -----------------
if __name__ == "__main__":
    load_fingerprints()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
