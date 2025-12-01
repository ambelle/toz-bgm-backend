import os
import tempfile
import numpy as np
import librosa

from flask import Flask, request, jsonify
from flask_cors import CORS

from bgm_list import TRACKS  # we use this for the names

# ------------------------------------------------
# Basic config
# ------------------------------------------------
TARGET_SR = 22050
N_MELS = 64
HOP_LENGTH = 512

app = Flask(__name__)
CORS(app)

# ------------------------------------------------
# Load fingerprints.npz (be tolerant about keys)
# ------------------------------------------------
npz_path = "fingerprints.npz"
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"{npz_path} not found in working directory")

data = np.load(npz_path, allow_pickle=True)

# Try some likely keys; otherwise just take the first array in the file
possible_keys = ["feats", "fingerprints", "features", "fps"]
fp_array = None

for k in possible_keys:
    if k in data.files:
        fp_array = data[k]
        print(f"Loaded fingerprints from key '{k}'")
        break

if fp_array is None:
    # fall back to "whatever is there"
    first_key = data.files[0]
    fp_array = data[first_key]
    print(f"Loaded fingerprints from first key '{first_key}'")

FINGERPRINTS = np.array(fp_array)

if FINGERPRINTS.ndim != 2:
    raise ValueError(
        f"FINGERPRINTS should be 2D (tracks x features), "
        f"but got shape {FINGERPRINTS.shape}"
    )

N_TRACKS = FINGERPRINTS.shape[0]
print(f"Loaded {N_TRACKS} fingerprints from {npz_path}")

# sanity check with TRACKS length (not fatal, just warn)
if len(TRACKS) != N_TRACKS:
    print(
        f"WARNING: TRACKS length ({len(TRACKS)}) != "
        f"fingerprints rows ({N_TRACKS}). "
        "Matching will still run but names may be misaligned."
    )

# ------------------------------------------------
# Fingerprint helpers
# ------------------------------------------------
def make_fingerprint(path: str) -> np.ndarray:
    """
    Load an audio file and compute a simple log-mel mean fingerprint.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio signal")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    feat = log_mel.mean(axis=1)  # (N_MELS,)
    return feat.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between 1D vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_best_match(query_feat: np.ndarray):
    """
    Compare query_feat against all rows in FINGERPRINTS.
    Returns (best_index, best_score).
    """
    if query_feat.ndim != 1:
        raise ValueError("query_feat must be 1D")

    scores = []
    for i in range(N_TRACKS):
        ref = FINGERPRINTS[i]
        scores.append(cosine_sim(query_feat, ref))

    scores = np.array(scores)
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    return best_idx, best_score

# ------------------------------------------------
# Routes
# ------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tracks": N_TRACKS})


@app.route("/match", methods=["POST"])
def match():
    """
    Expect: multipart/form-data with field name 'audio'
    Frontend sends small WEBM/OGG/WAV chunks.
    """
    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "Missing 'audio' file field"}), 400

    f = request.files["audio"]

    # Save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        # librosa can usually read webm/ogg via audioread+ffmpeg
        q_feat = make_fingerprint(tmp_path)
        best_idx, best_score = find_best_match(q_feat)

        # pick name from TRACKS if lengths match, else fallback
        if best_idx < len(TRACKS):
            best_name = TRACKS[best_idx]["name"]
        else:
            best_name = f"Track #{best_idx+1}"

        # simple threshold so we don’t shout random answers
        THRESH = 0.7
        if best_score < THRESH:
            return jsonify({
                "status": "no_match",
                "score": best_score,
                "message": "No confident match"
            })

        return jsonify({
            "status": "ok",
            "name": best_name,
            "score": best_score
        })

    except Exception as e:
        print("Error in /match:", repr(e))
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ------------------------------------------------
# Local dev entry
# ------------------------------------------------
if __name__ == "__main__":
    # For local testing only; Render uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
