import io
import os
import logging

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fft import dct

from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------- CONFIG -----------------
TARGET_SR = 22050
N_MFCC = 20
N_MELS = 64
THRESHOLD_CONFIDENT = 0.75  # on [0, 1]

FINGERPRINTS_PATH = "fingerprints.npz"

# Globals
F = None            # (num_tracks, N_MFCC)
TRACK_NAMES = None  # list[str]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Flask app
app = Flask(__name__)
CORS(app)


# ----------------- AUDIO / MFCC HELPERS -----------------
def decode_wav_bytes(audio_bytes: bytes):
    """
    Decode raw WAV bytes using scipy.io.wavfile.
    Returns float32 mono signal y, sample rate sr.
    """
    try:
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
    except Exception as e:
        raise RuntimeError(f"wavfile.read failed: {e}")

    if data is None or data.size == 0:
        raise RuntimeError("Empty audio data")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert to float32 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        y = data.astype(np.float32) / max_val
    else:
        y = data.astype(np.float32)

    return y, sr


def resample_to_target(y: np.ndarray, sr: int, target_sr: int = TARGET_SR):
    if sr == target_sr:
        return y, sr
    # Use polyphase resampling for better quality
    y_rs = resample_poly(y, target_sr, sr).astype(np.float32)
    return y_rs, target_sr


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None):
    """
    Create a (n_mels, 1 + n_fft//2) mel filterbank matrix similar to librosa.
    """
    if fmax is None:
        fmax = sr / 2.0

    # FFT bins
    n_fft_bins = 1 + n_fft // 2
    fft_freqs = np.linspace(0, sr / 2.0, n_fft_bins)

    # Mel scale points
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = mel_to_hz(m_points)

    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft_bins - 1)

    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left = bin_indices[i - 1]
        center = bin_indices[i]
        right = bin_indices[i + 1]

        if center == left:
            center += 1
        if right == center:
            right += 1

        # Rising slope
        for j in range(left, center):
            fb[i - 1, j] = (fft_freqs[j] - hz_points[i - 1]) / max(hz_points[i] - hz_points[i - 1], 1e-10)
        # Falling slope
        for j in range(center, right):
            fb[i - 1, j] = (hz_points[i + 1] - fft_freqs[j]) / max(hz_points[i + 1] - hz_points[i], 1e-10)

    # Normalise filters so each row sums to 1 (approx)
    fb_sum = fb.sum(axis=1, keepdims=True)
    fb_sum[fb_sum == 0.0] = 1.0
    fb /= fb_sum

    return fb


def compute_mfcc(y: np.ndarray, sr: int,
                 n_mfcc: int = N_MFCC,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = N_MELS) -> np.ndarray:
    """
    Compute MFCCs similar to librosa.feature.mfcc:
    - STFT -> magnitude^2
    - Mel filterbank
    - log
    - DCT-II (ortho)
    Returns array of shape (n_mfcc, T).
    """
    if y.size == 0:
        raise RuntimeError("No audio samples for MFCC")

    # Ensure at least one frame
    if y.size < n_fft:
        pad_width = n_fft - y.size
        y = np.pad(y, (0, pad_width), mode="constant")

    # Framing
    n_frames = 1 + (y.size - n_fft) // hop_length
    if n_frames <= 0:
        n_frames = 1

    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        if end <= y.size:
            frames[i, :] = y[start:end]
        else:
            # Zero-pad last frame if needed
            temp = np.zeros(n_fft, dtype=np.float32)
            part = y[start:y.size]
            temp[:part.size] = part
            frames[i, :] = temp
        frames[i, :] *= window

    # STFT power spectrogram
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spec = (np.abs(spec) ** 2).astype(np.float32)  # (n_frames, 1 + n_fft//2)

    # Mel filterbank
    mel_fb = build_mel_filterbank(sr, n_fft, n_mels)  # (n_mels, n_bins)
    mel_spec = mel_fb @ power_spec.T  # (n_mels, n_frames)

    # Log-energy
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    # DCT over mel-axis -> MFCC (n_mfcc, n_frames)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")
    mfcc = mfcc[:n_mfcc, :]

    return mfcc.astype(np.float32)


def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Decode WAV bytes, resample to TARGET_SR, compute MFCCs,
    average over time, L2-normalise.
    Returns vector of shape (N_MFCC,).
    """
    y, sr = decode_wav_bytes(audio_bytes)
    y, sr = resample_to_target(y, sr, TARGET_SR)

    # Use only center 3 seconds from what browser sends
    max_seconds = 3.0
    max_len = int(max_seconds * TARGET_SR)
    if y.size > max_len:
        start = (y.size - max_len) // 2
        y = y[start:start + max_len]

    mfcc = compute_mfcc(y, sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512, n_mels=N_MELS)
    if mfcc.size == 0:
        raise RuntimeError("Empty MFCC matrix")

    v = mfcc.mean(axis=1)  # (N_MFCC,)
    v = v.astype(np.float32)

    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise RuntimeError("Zero-norm MFCC vector")

    v /= norm
    return v


def cosine_best_match(F_mat: np.ndarray, q_vec: np.ndarray):
    """
    Given:
      F_mat: (N, D) L2-normalised
      q_vec: (D,) L2-normalised
    Returns:
      best_idx, best_sim
    """
    sims = F_mat @ q_vec  # (N,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    return best_idx, best_sim


def sim_to_confidence(sim: float) -> float:
    """
    Map cosine similarity [-1, 1] -> [0, 1].
    """
    conf = (sim + 1.0) / 2.0
    conf = max(0.0, min(1.0, conf))
    return conf


# ----------------- FINGERPRINT LOADING -----------------
def load_fingerprints():
    global F, TRACK_NAMES

    if F is not None and TRACK_NAMES is not None:
        return

    logger.info(f"Loading fingerprints from {FINGERPRINTS_PATH}...")
    data = np.load(FINGERPRINTS_PATH, allow_pickle=True)

    feats = data["feats"].astype(np.float32)   # (N, D)
    names = [str(x) for x in data["names"].tolist()]

    # L2-normalise rows
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    feats = feats / norms

    F = feats
    TRACK_NAMES = names

    logger.info(f"Loaded {F.shape[0]} fingerprints, dim={F.shape[1]}")


# ----------------- ROUTES -----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/match", methods=["POST", "OPTIONS"])
def match():
    if request.method == "OPTIONS":
        # CORS preflight
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
