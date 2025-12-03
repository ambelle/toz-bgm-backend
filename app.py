import os
import io
import wave
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# ==========================
# Audio / fingerprint config
# ==========================

TARGET_SR = 22050       # resample everything here
N_FFT = 1024            # so fingerprint dim = 1024//2 + 1 = 513
SIM_THRESHOLD = 0.75    # similarity threshold for "confident" matches

app = Flask(__name__)
CORS(app)

# ==========================
# WAV loading (no librosa)
# ==========================

def load_wav_from_bytes(audio_bytes: bytes):
    """
    Load a PCM WAV (like the one sent from scanner.html) using stdlib wave.
    Returns: y (float32, mono, -1..1), sr
    """
    bio = io.BytesIO(audio_bytes)
    with wave.open(bio, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        # We only expect 16-bit PCM from our scanner.
        raise ValueError(f"Unsupported sample width: {sample_width * 8} bits")

    # int16 -> float32 in [-1, 1]
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        # reshape (frames, channels) then average to mono
        y = y.reshape(-1, n_channels).mean(axis=1)

    return y, sr


def resample_to_target(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Very simple linear resampler using NumPy only.
    Enough for this matching use-case.
    """
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float32)

    # duration in seconds
    duration = y.size / float(orig_sr)
    new_length = int(round(duration * target_sr))

    if new_length <= 1:
        return y.astype(np.float32)

    old_idx = np.linspace(0.0, 1.0, num=y.size, endpoint=False)
    new_idx = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    y_resampled = np.interp(new_idx, old_idx, y).astype(np.float32)
    return y_resampled


def compute_fingerprint(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Simple, robust FFT-based fingerprint:

    1) normalize
    2) resample to TARGET_SR
    3) frame into overlapping windows
    4) rFFT magnitude per frame
    5) average magnitude over time
    6) log(1 + mag) and L2 normalize

    Result shape: (N_FFT//2 + 1,) = (513,)
    """
    if y.size == 0:
        raise ValueError("Empty audio signal")

    # DC remove + amplitude normalize
    y = y - np.mean(y)
    max_abs = np.max(np.abs(y)) + 1e-8
    y = y / max_abs

    # resample to target SR
    y = resample_to_target(y, sr, TARGET_SR)

    if y.size < N_FFT:
        # pad to at least one frame
        pad = N_FFT - y.size
        y = np.pad(y, (0, pad), mode="constant")

    hop = N_FFT // 2  # 50% overlap
    num_frames = 1 + (y.size - N_FFT) // hop
    if num_frames <= 0:
        num_frames = 1

    frames = []
    window = np.hanning(N_FFT).astype(np.float32)

    for i in range(num_frames):
        start = i * hop
        end = start + N_FFT
        if end > y.size:
            frame = np.zeros(N_FFT, dtype=np.float32)
            frame[:y.size - start] = y[start:]
        else:
            frame = y[start:end]

        frame = frame * window
        fft_mag = np.abs(np.fft.rfft(frame, n=N_FFT))
        frames.append(fft_mag)

    spec = np.mean(np.stack(frames, axis=0), axis=0)  # (513,)
    spec = np.log1p(spec).astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(spec)
    if norm > 0:
        spec = spec / norm

    return spec


def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    y, sr = load_wav_from_bytes(audio_bytes)
    return compute_fingerprint(y, sr)


# ==========================
# Load fingerprints.npz
# ==========================

F = None           # feature matrix (num_tracks, feat_dim)
TRACK_NAMES = []   # list of track names

try:
    data = np.load("fingerprints.npz", allow_pickle=True)

    feat_key = None
    for k in ("features", "feats", "fingerprints", "fps"):
        if k in data.files:
            feat_key = k
            break

    if feat_key is None:
        raise KeyError(
            "fingerprints.npz missing feature array "
            "(expected one of: features, feats, fingerprints, fps)"
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
    print(f"Feature shape: {F.shape}")

except Exception as e:
    print("Failed to load fingerprints:", e)
    F = None
    TRACK_NAMES = []


# ==========================
# Matching
# ==========================

def find_best_match(query_fp: np.ndarray):
    """
    Cosine similarity between query and all stored fingerprints.
    F is assumed L2-normalized row-wise.
    """
    if F is None or len(TRACK_NAMES) == 0:
        return None, 0.0

    q = query_fp.reshape(1, -1)

    if q.shape[1] != F.shape[1]:
        print(f"Feature dim mismatch: query {q.shape}, F {F.shape}")
        return None, 0.0

    sims = (F @ q.T).ravel()  # (num_tracks,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = TRACK_NAMES[best_idx]
    return best_name, best_score


# ==========================
# Routes
# ==========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "loaded": F is not None,
        "tracks": len(TRACK_NAMES),
        "feat_dim": int(F.shape[1]) if F is not None else 0,
    })


@app.route("/match", methods=["POST"])
def match():
    """
    Accepts:
      - Raw audio bytes with Content-Type: audio/wav (from scanner.html)
      - OR legacy multipart/form-data with a file field named 'audio'
    """
    try:
        ct = request.content_type or ""
        audio_bytes = None

        if ct.startswith("audio/"):
            audio_bytes = request.data
        else:
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
            print("Error decoding / fingerprinting audio:", e)
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
    # Local testing
    app.run(debug=True, host="0.0.0.0", port=5000)
