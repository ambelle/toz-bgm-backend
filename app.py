import io
import os
import wave
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# === Audio / fingerprint settings ===
TARGET_SR = 22050
N_FFT = 1024       # 1024 -> rfft bins = 513
HOP_LENGTH = 512
SIM_THRESHOLD = 0.75  # tweak if needed

app = Flask(__name__)
CORS(app)

# === Core fingerprint function (NO librosa, NO numba) ===

def compute_fingerprint(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Turn a mono waveform into a fixed-length fingerprint vector.
    Pure NumPy implementation, no librosa / numba.
    Output dim = N_FFT//2 + 1 (for N_FFT=1024 -> 513).
    """
    if y.ndim > 1:
        # mixdown to mono if needed
        y = np.mean(y, axis=0)

    y = np.asarray(y, dtype=np.float32)

    # simple resample to TARGET_SR using linear interpolation
    if sr != TARGET_SR and y.size > 1:
        old_len = y.shape[0]
        new_len = int(round(old_len * TARGET_SR / sr))
        if new_len < 2:
            new_len = 2
        t_old = np.linspace(0.0, 1.0, num=old_len, endpoint=False, dtype=np.float32)
        t_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float32)
        y = np.interp(t_new, t_old, y).astype(np.float32)
        sr = TARGET_SR

    # keep at most 10 seconds
    max_len = int(10 * sr)
    if y.size > max_len:
        y = y[:max_len]

    # if super short, pad to at least one frame
    if y.size < N_FFT:
        pad = N_FFT - y.size
        y = np.pad(y, (0, pad), mode="constant")

    # frame into overlapping windows using stride tricks
    n_frames = 1 + (y.size - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        # fallback: single frame with padding
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")
        n_frames = 1

    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP_LENGTH][:n_frames]

    # apply Hann window
    window = np.hanning(N_FFT).astype(np.float32)
    frames = frames * window

    # rFFT over frames -> (n_frames, N_FFT//2+1)
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    mag = np.abs(spec).astype(np.float32)

    # power & average over time
    power = mag ** 2
    avg_spec = power.mean(axis=0)  # (N_FFT//2+1,)

    # log-compress
    log_spec = np.log1p(avg_spec)  # log(1 + x)

    feat = log_spec.astype(np.float32)

    # L2-normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm

    return feat


def load_wav_from_bytes(audio_bytes: bytes):
    """
    Decode a WAV (PCM) from bytes using the standard 'wave' module.
    Returns (waveform np.ndarray, sample_rate).
    """
    bio = io.BytesIO(audio_bytes)
    with wave.open(bio, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        # 16-bit PCM
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        audio /= 32768.0
    elif sampwidth == 1:
        # 8-bit PCM unsigned
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sr


def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Wrapper: decode WAV bytes -> waveform -> fingerprint.
    """
    y, sr = load_wav_from_bytes(audio_bytes)

    if y.size == 0:
        raise ValueError("Empty decoded audio")

    feat = compute_fingerprint(y, sr)
    return feat


def find_best_match(query_fp: np.ndarray):
    """
    Cosine similarity between query_fp and all stored fingerprints F.
    F is assumed L2-normalized per row.
    """
    if F is None or len(TRACK_NAMES) == 0:
        return None, 0.0

    q = query_fp.reshape(1, -1)  # (1, D)
    sims = (F @ q.T).ravel()     # (num_tracks,)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_name = TRACK_NAMES[best_idx]
    return best_name, best_score


# === Load fingerprints once on startup ===

F = None
TRACK_NAMES = []

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


# === Routes ===

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "loaded": F is not None,
        "tracks": len(TRACK_NAMES),
        "feature_dim": int(F.shape[1]) if F is not None else 0,
    })


@app.route("/match", methods=["POST"])
def match():
    """
    Accepts:
      - Raw audio bytes with Content-Type: audio/wav
      - OR legacy multipart/form-data with field name 'audio'
    """
    try:
        ct = request.content_type or ""
        audio_bytes = None

        if ct.startswith("audio/"):
            audio_bytes = request.data
        else:
            f = request.files.get("audio")
            if f:
                audio_bytes = f.read()

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

        return jsonify({
            "ok": True,
            "match": name,
            "confidence": score,
            "threshold": SIM_THRESHOLD,
            "is_confident": bool(score >= SIM_THRESHOLD),
        })

    except Exception:
        print("Error in /match:", traceback.format_exc())
        return jsonify({"error": "server_error"}), 500


if __name__ == "__main__":
    # For local testing (NOT used on Render)
    app.run(debug=True, host="0.0.0.0", port=5000)
