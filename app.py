import io
import wave
import logging
import time

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------
# Config
# ---------------------------

FINGERPRINTS_PATH = "fingerprints.npz"
TARGET_SR = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.75  # dot-product threshold (since vectors are L2-normalised)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ---------------------------
# Utilities: WAV decoding
# ---------------------------

def decode_wav_to_mono_float32(audio_bytes: bytes):
    """
    Decode a 16-bit PCM WAV from bytes into (mono_float32_array, sample_rate).
    Returns (None, None) if decoding fails or format unsupported.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth != 2:
            logger.error(f"Unsupported sample width: {sampwidth * 8} bits")
            return None, None

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels)
            audio = audio.mean(axis=1)

        return audio, sr

    except Exception as e:
        logger.exception(f"Failed to decode WAV: {e}")
        return None, None

# ---------------------------
# Utilities: Resampling
# ---------------------------

def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple linear interpolation resampler.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    ratio = float(target_sr) / float(orig_sr)
    new_len = int(len(audio) * ratio)
    if new_len <= 1:
        return audio.astype(np.float32, copy=False)

    old_idx = np.arange(len(audio), dtype=np.float32)
    new_idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
    resampled = np.interp(new_idx, old_idx, audio).astype(np.float32)
    return resampled

# ---------------------------
# Utilities: Mel filterbank
# ---------------------------

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def build_mel_filter(sr, n_fft, n_mels):
    """
    Build a simple mel triangular filterbank (similar in spirit to librosa.filters.mel).
    Returns a matrix of shape (n_mels, n_fft//2 + 1).
    """
    f_min = 0.0
    f_max = float(sr) / 2.0

    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        right = min(right, n_freqs - 1)

        for i in range(left, center):
            fb[m - 1, i] = (i - left) / max(1, (center - left))
        for i in range(center, right):
            fb[m - 1, i] = (right - i) / max(1, (right - center))

    return fb

# Precompute mel filter for TARGET_SR
MEL_FILTER = build_mel_filter(TARGET_SR, N_FFT, N_MELS)

# ---------------------------
# Utilities: STFT + fingerprint
# ---------------------------

def stft_magnitude(audio: np.ndarray, n_fft: int, hop_length: int):
    """
    Simple STFT using Hann window, returns power spectrogram (freqs x frames).
    """
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < n_fft:
        # pad if too short
        pad = n_fft - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    n_samples = len(audio)
    n_frames = 1 + (n_samples - n_fft) // hop_length
    if n_frames <= 0:
        n_frames = 1

    window = np.hanning(n_fft).astype(np.float32)
    n_freqs = n_fft // 2 + 1
    S = np.empty((n_freqs, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = audio[start:end]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)), mode="constant")
        frame = frame * window
        spec = np.fft.rfft(frame, n=n_fft)
        S[:, i] = (np.abs(spec) ** 2).astype(np.float32)

    return S

def make_fingerprint_from_pcm(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Turn raw PCM (float32 mono) + sr into a 64-dim L2-normalised fingerprint.
    """
    if audio is None or len(audio) == 0 or sr <= 0:
        return None

    # 1) Resample to TARGET_SR
    audio_22050 = resample_linear(audio, sr, TARGET_SR)

    # 2) STFT -> power spectrogram
    S = stft_magnitude(audio_22050, N_FFT, HOP_LENGTH)  # (n_freqs, frames)

    # 3) Apply mel filterbank: (n_mels, n_freqs) @ (n_freqs, frames) -> (n_mels, frames)
    mel_spec = MEL_FILTER @ S

    # 4) Take mean over time to get 1D vector (n_mels,)
    mel_mean = np.mean(mel_spec, axis=1)

    # 5) L2 normalise
    vec = mel_mean.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        return None

    vec /= norm
    return vec

def make_fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Convenience: decode WAV bytes -> mono float -> fingerprint.
    """
    audio, sr = decode_wav_to_mono_float32(audio_bytes)
    if audio is None or sr is None:
        return None
    return make_fingerprint_from_pcm(audio, sr)

# ---------------------------
# Load fingerprints from disk
# ---------------------------

logger.info(f"Loading fingerprints from {FINGERPRINTS_PATH} ...")
try:
    data = np.load(FINGERPRINTS_PATH, allow_pickle=True)
    feat_key = "feats" if "feats" in data else list(data.keys())[0]
    name_key = "names" if "names" in data else list(data.keys())[1]

    F = data[feat_key].astype(np.float32)  # (N, D)
    TRACK_NAMES = [str(x) for x in data[name_key].tolist()]

    # L2 normalise all stored vectors (safety)
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    F = F / norms

    logger.info(f"Loaded {F.shape[0]} fingerprints of dimension {F.shape[1]}")
except Exception as e:
    logger.exception(f"Failed to load fingerprints: {e}")
    F = None
    TRACK_NAMES = []

# ---------------------------
# Routes
# ---------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "healthy"}), 200

@app.route("/match", methods=["POST", "OPTIONS"])
def match():
    if request.method == "OPTIONS":
        # CORS preflight
        return ("", 200)

    if F is None or len(TRACK_NAMES) == 0:
        return jsonify({"ok": False, "error": "fingerprints_not_loaded"}), 500

    audio_bytes = request.get_data()
    content_type = request.headers.get("Content-Type", "")

    logger.info(f"[MATCH] received {len(audio_bytes)} bytes, content_type={content_type}")

    start_time = time.time()

    q_vec = make_fingerprint_from_bytes(audio_bytes)
    if q_vec is None:
        logger.warning("[MATCH] Could not build fingerprint from audio.")
        return jsonify({"ok": False, "error": "decode_or_feature_error"}), 400

    # Compute cosine similarity (dot product because both are L2 normalised)
    sims = F @ q_vec  # (N,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    match_name = TRACK_NAMES[best_idx]
    is_confident = best_sim >= CONFIDENCE_THRESHOLD

    elapsed = (time.time() - start_time) * 1000.0
    logger.info(
        f"[MATCH] best='{match_name}' sim={best_sim:.3f} "
        f"confident={is_confident} in {elapsed:.1f} ms"
    )

    return jsonify({
        "ok": True,
        "match": match_name,
        "confidence": best_sim,
        "threshold": CONFIDENCE_THRESHOLD,
        "is_confident": is_confident,
    }), 200

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    # For local testing only; Render uses gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
