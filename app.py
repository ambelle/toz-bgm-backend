import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # Render + librosa safety

import io
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa

# === Config ===
TARGET_SR = 22050
N_MELS = 64

# This is how strict we are when deciding "yes, this is the track"
# You can tweak 0.90 → 0.85 or 0.93 later.
SIM_THRESHOLD = 0.90

app = Flask(__name__)
CORS(app)

# === Load fingerprints.npz on startup ===
F = None           # (num_tracks, feat_dim)
TRACK_NAMES = []   # ["Henesys", "Perion", ...]


def _load_fingerprints():
    global F, TRACK_NAMES

    try:
        data = np.load("fingerprints.npz", allow_pickle=True)

        # feature key
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

        # L2-normalize each row just in case
        norms = np.linalg.norm(F, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        F = F / norms

        name_key = None
        for k in ("names", "track_names", "labels"):
            if k in data.files:
                name_key = k
                break

        if name_key is not None:
            TRACK_NAMES = [str(x) for x in data[name_key].tolist()]
        else:
            TRACK_NAMES = [f"Track {i}" for i in range(F.shape[0])]

        print(f"[INIT] Loaded {F.shape[0]} fingerprints from key '{feat_key}'")
        print(f"[INIT] Example names: {TRACK_NAMES[:5]}")

    except Exception as e:
        print("[INIT] Failed to load fingerprints.npz:", e)
        F = None
        TRACK_NAMES = []


_load_fingerprints()

# === Helpers ===


def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode audio from bytes, compute log-mel spectrogram,
    average over time, L2-normalize → (N_MELS,) vector.
    """
    bio = io.BytesIO(audio_bytes)
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
    F is assumed L2-normalized row-wise.
    Returns (best_name or None, best_score).
    """
    if F is None or len(TRACK_NAMES) == 0:
        return None, 0.0

    q = query_fp.reshape(1, -1)

    sims = (F @ q.T).ravel()  # (num_tracks,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # NaN / inf safety
    if not np.isfinite(best_score):
        best_score = 0.0

    best_name = TRACK_NAMES[best_idx]

    # Debug log top-3 to Render logs (optional but very useful)
    try:
        top3_idx = np.argsort(sims)[-3:][::-1]
        top3 = [(TRACK_NAMES[i], float(sims[i])) for i in top3_idx]
        print(f"[MATCH] Top3: {top3}")
    except Exception:
        pass

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
      - Raw audio bytes with Content-Type: audio/wav
      - OR multipart/form-data with field 'audio'
    Returns JSON with:
      ok, match, confidence, threshold, is_confident
    """
    try:
        ct = request.content_type or ""
        audio_bytes = None

        # Raw wav from scanner.html
        if ct.startswith("audio/"):
            audio_bytes = request.data
        else:
            # Fallback: multipart upload
            file = request.files.get("audio")
            if file:
                audio_bytes = file.read()

        if not audio_bytes:
            print(f"[MATCH] 400: no audio bytes, content_type={ct}")
            return jsonify({
                "ok": False,
                "error": "no_audio",
                "match": None,
                "confidence": 0.0,
                "threshold": float(SIM_THRESHOLD),
                "is_confident": False,
            }), 400

        print(f"[MATCH] received {len(audio_bytes)} bytes, content_type={ct}")

        try:
            query_fp = make_fingerprint_from_bytes(audio_bytes)
        except Exception as e:
            print("[MATCH] Error decoding audio:", e)
            return jsonify({
                "ok": False,
                "error": "decode_failed",
                "match": None,
                "confidence": 0.0,
                "threshold": float(SIM_THRESHOLD),
                "is_confident": False,
            }), 400

        name, score = find_best_match(query_fp)

        # If similarity is too low, treat as "no match"
        if name is None or score < 0.1:
            resp = {
                "ok": True,
                "match": None,
                "confidence": float(score),
                "threshold": float(SIM_THRESHOLD),
                "is_confident": False,
            }
            print(f"[MATCH] No match (score={score:.3f})")
            return jsonify(resp)

        is_confident = bool(score >= SIM_THRESHOLD)

        resp = {
            "ok": True,
            "match": name,
            "confidence": float(score),
            "threshold": float(SIM_THRESHOLD),
            "is_confident": is_confident,
        }

        print(f"[MATCH] best='{name}' score={score:.3f} "
              f"confident={is_confident} (threshold={SIM_THRESHOLD})")

        return jsonify(resp)

    except Exception:
        print("[MATCH] Error in /match:", traceback.format_exc())
        return jsonify({
            "ok": False,
            "error": "server_error",
            "match": None,
            "confidence": 0.0,
            "threshold": float(SIM_THRESHOLD),
            "is_confident": False,
        }), 500


if __name__ == "__main__":
    # Local testing
    app.run(debug=True, host="0.0.0.0", port=5000)
