import io
import logging
import time
from typing import Tuple, Optional

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import signal
from scipy.io import wavfile

# =====================
# CONFIG
# =====================

TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
FMIN = 0.0
FMAX = TARGET_SR / 2.0

FINGERPRINTS_PATH = "fingerprints.npz"
MATCH_THRESHOLD = 0.75  # cosine similarity threshold
MIN_CONFIDENCE_TO_NAME = 0.05  # below this, treat as "no match"

# =====================
# LOGGING
# =====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# =====================
# MEL FILTER
# =====================

def hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filter(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float
) -> np.ndarray:
    """Create a mel filterbank similar to librosa.filters.mel but without librosa."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)

    mels = np.linspace(hz_to_mel(np.array([fmin]))[0],
                       hz_to_mel(np.array([fmax]))[0],
                       n_mels + 2)
    hz_points = mel_to_hz(mels)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_left = hz_points[m - 1]
        f_center = hz_points[m]
        f_right = hz_points[m + 1]

        # rising slope
        left_inds = np.logical_and(fft_freqs >= f_left, fft_freqs <= f_center)
        if f_center > f_left:
            fb[m - 1, left_inds] = (
                (fft_freqs[left_inds] - f_left) / (f_center - f_left)
            )

        # falling slope
        right_inds = np.logical_and(fft_freqs >= f_center, fft_freqs <= f_right)
        if f_right > f_center:
            fb[m - 1, right_inds] = (
                (f_right - fft_freqs[right_inds]) / (f_right - f_center)
            )

    # Normalize filters to have equal area (optional but common)
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    fb *= enorm[:, np.newaxis].astype(np.float32)
    return fb


MEL_FILTER = build_mel_filter(
    sr=TARGET_SR,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=FMIN,
    fmax=FMAX,
)

# =====================
# FINGERPRINT LOADING
# =====================

def load_fingerprints(path: str = FINGERPRINTS_PATH) -> Tuple[np.ndarray, list]:
    logger.info(f"Loading fingerprints from {path}")
    data = np.load(path, allow_pickle=True)

    feats = data["feats"].astype(np.float32)  # shape (N, 64)
    names_raw = data["names"]                # shape (N,)

    # L2-normalize database
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    feats = feats / norms

    # Ensure names are nice strings
    names_list = [str(x) for x in names_raw.tolist()]
    logger.info(
        f"Loaded {feats.shape[0]} tracks with dim={feats.shape[1]}"
    )

    return feats, names_list


F_DB, TRACK_NAMES = load_fingerprints()

# =====================
# AUDIO → FINGERPRINT
# =====================

def pcm_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert various PCM types to float32 in [-1, 1]."""
    if x.dtype == np.float32 or x.dtype == np.float64:
        return x.astype(np.float32)

    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0)
    if x.dtype == np.uint8:
        # 8-bit unsigned PCM, 0..255 -> -1..1
        return ((x.astype(np.float32) - 128.0) / 128.0)

    # Fallback: scale by max abs
    x = x.astype(np.float32)
    m = np.max(np.abs(x)) or 1.0
    return x / m


def waveform_to_fingerprint(wave: np.ndarray, sr: int) -> np.ndarray:
    """Core fingerprint logic shared by backend and generator."""
    if wave.ndim > 1:
        wave = wave.mean(axis=1)

    wave = pcm_to_float32(wave)

    # Remove DC offset
    wave = wave - np.mean(wave)

    # Resample if needed
    if sr != TARGET_SR:
        wave = signal.resample_poly(wave, TARGET_SR, sr)
        sr = TARGET_SR

    if wave.size == 0:
        raise ValueError("Empty audio after preprocessing")

    # STFT
    _, _, Zxx = signal.stft(
        wave,
        fs=TARGET_SR,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx).astype(np.float32)

    if mag.size == 0:
        raise ValueError("Empty STFT magnitude")

    # Power spectrogram
    power_spec = mag ** 2

    # Mel projection
    mel_spec = MEL_FILTER @ power_spec  # (N_MELS, T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log10(mel_spec).astype(np.float32)

    # Mean over time
    fp = log_mel.mean(axis=1)

    # L2-normalize
    norm = np.linalg.norm(fp)
    if norm > 0.0:
        fp = fp / norm

    return fp.astype(np.float32)


def make_fingerprint_from_bytes(raw: bytes) -> np.ndarray:
    """Decode 16-bit WAV bytes and create a fingerprint."""
    # Read WAV from bytes
    sr, data = wavfile.read(io.BytesIO(raw))

    if data.ndim == 0 or data.size == 0:
        raise ValueError("No audio data in WAV")

    return waveform_to_fingerprint(data, sr)


# =====================
# FLASK APP
# =====================

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return jsonify({"ok": True, "message": "TOZ BGM backend is running"})


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/match", methods=["POST", "OPTIONS"])
def match():
    if request.method == "OPTIONS":
        # CORS preflight
        return ("", 200)

    t0 = time.time()
    raw = request.get_data()
    logger.info(f"[MATCH] received {len(raw)} bytes, content_type={request.content_type}")

    if not raw:
        return jsonify({"ok": False, "error": "Empty body"}), 400

    try:
        q_vec = make_fingerprint_from_bytes(raw)  # shape (64,)
    except Exception as e:
        logger.exception("[MATCH] Error decoding / fingerprinting audio")
        return jsonify({"ok": False, "error": f"decode_error: {e}"}), 400

    # Ensure shape (D, 1) for matrix multiply; F_DB is (N, D)
    if q_vec.ndim != 1 or q_vec.shape[0] != F_DB.shape[1]:
        return jsonify({"ok": False, "error": "query_dim_mismatch"}), 400

    # Dot product = cosine similarity (because everything is L2-normalized)
    sims = F_DB @ q_vec  # shape (N,)
    best_idx = int(np.argmax(sims))
    best_raw = float(sims[best_idx])

    # Clamp to [0, 1] so no -13% nonsense
    confidence = max(0.0, min(1.0, best_raw))

    is_confident = confidence >= MATCH_THRESHOLD
    elapsed_ms = int((time.time() - t0) * 1000)

    if confidence < MIN_CONFIDENCE_TO_NAME:
        match_name: Optional[str] = None
    else:
        match_name = TRACK_NAMES[best_idx]

    logger.info(
        f"[MATCH] best={match_name} raw={best_raw:.3f} "
        f"conf={confidence:.3f} thr={MATCH_THRESHOLD:.2f} ms={elapsed_ms}"
    )

    return jsonify(
        {
            "ok": True,
            "match": match_name,
            "confidence": confidence,
            "threshold": MATCH_THRESHOLD,
            "is_confident": bool(is_confident),
            "elapsed_ms": elapsed_ms,
        }
    )


if __name__ == "__main__":
    # For local debug only; Render uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
