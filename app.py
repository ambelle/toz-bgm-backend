import io
import os
import wave
import logging

import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# Config
# -----------------------------
TARGET_SR = 22050
N_MELS = 64
FINGERPRINTS_PATH = "fingerprints.npz"
FEAT_KEY = "feats"
NAME_KEY = "names"
MATCH_THRESHOLD = 0.75  # cosine similarity threshold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load fingerprints at startup
# -----------------------------
def load_fingerprints(path=FINGERPRINTS_PATH,
                      feat_key=FEAT_KEY,
                      name_key=NAME_KEY):
    if not os.path.exists(path):
        logger.error("fingerprints.npz not found at %s", path)
        raise FileNotFoundError(f"{path} not found; run generate_fingerprints_simple.py first.")

    data = np.load(path, allow_pickle=True)

    F = data[feat_key].astype(np.float32)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    F = F / norms  # L2-normalized rows

    names_raw = data[name_key].tolist()
    names = [str(x) for x in names_raw]

    logger.info(
        "Loaded fingerprints from %s: feats.shape=%s, names=%d",
        path, F.shape, len(names)
    )
    return F, names


F_MATRIX, TRACK_NAMES = load_fingerprints()


# -----------------------------
# Audio → fingerprint
# -----------------------------
def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode a PCM WAV from bytes, resample to TARGET_SR,
    build mel-spectrogram, mean-pool, and L2-normalize.
    """
    bio = io.BytesIO(audio_bytes)

    try:
        with wave.open(bio, "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # int16 PCM → float32 -1..1
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        # If multi-channel, take first channel
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

    except Exception as e:
        logger.exception("[make_fingerprint_from_bytes] WAV decode failed")
        raise ValueError(f"WAV decode failed: {e}")

    if audio.size == 0:
        raise ValueError("Empty audio after decode.")

    # Resample to TARGET_SR if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=N_MELS,
        power=2.0,
        fmin=20,
        fmax=8000,
    )

    # Mean pool over time axis → (N_MELS,)
    feat = np.mean(mel, axis=1).astype(np.float32)

    nrm = np.linalg.norm(feat)
    if nrm == 0.0:
        raise ValueError("Zero-norm feature vector.")

    feat = feat / nrm
    return feat


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "up"})


@app.route("/match", methods=["POST", "OPTIONS"])
def match():
    # CORS preflight
    if request.method == "OPTIONS":
        return ("", 200)

    content_type = request.headers.get("Content-Type", "")
    logger.info(
        "[MATCH] received %s bytes, content_type=%s",
        request.content_length,
        content_type,
    )

    if "audio/wav" not in content_type:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "bad_content_type",
                    "detail": "Expected Content-Type: audio/wav",
                }
            ),
            400,
        )

    audio_bytes = request.get_data()
    if not audio_bytes:
        return (
            jsonify({"ok": False, "error": "empty_body", "detail": "No audio data."}),
            400,
        )

    try:
        q_vec = make_fingerprint_from_bytes(audio_bytes)
    except Exception as e:
        logger.exception("[MATCH] make_fingerprint_from_bytes failed")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "decode_failed",
                    "detail": str(e),
                }
            ),
            400,
        )

    # Cosine similarity with fingerprint matrix (already L2-normalized)
    sims = F_MATRIX @ q_vec  # (num_tracks,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    is_confident = best_sim >= MATCH_THRESHOLD
    best_name = TRACK_NAMES[best_idx] if TRACK_NAMES else "Unknown"

    logger.info(
        "[MATCH] best=%s score=%.3f threshold=%.2f",
        best_name,
        best_sim,
        MATCH_THRESHOLD,
    )

    return (
        jsonify(
            {
                "ok": True,
                "match": best_name,
                "confidence": best_sim,
                "threshold": MATCH_THRESHOLD,
                "is_confident": is_confident,
            }
        ),
        200,
    )


if __name__ == "__main__":
    # For local testing; Render will use gunicorn
    app.run(host="0.0.0.0", port=10000, debug=False)
