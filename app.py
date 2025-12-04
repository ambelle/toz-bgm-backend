import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf     # Only needed to decode WAV/WEBM safely

app = Flask(__name__)
CORS(app)

TARGET_SR = 22050
N_MELS = 64


# -----------------------------
#  LOAD FINGERPRINTS
# -----------------------------
data = np.load("fingerprints.npz", allow_pickle=True)

F = data["feats"].astype(np.float32)
names = [str(x) for x in data["names"]]

# L2 normalize rows
norms = np.linalg.norm(F, axis=1, keepdims=True)
norms[norms == 0] = 1
F = F / norms


# -----------------------------
#  UTIL: compute mel features
# -----------------------------
def mel_filterbank(sr, n_fft=2048, n_mels=64):
    """Manually create mel filterbank (librosa-free)."""
    # mel scale helpers
    def hz_to_mel(hz): 
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(m): 
        return 700 * (10**(m / 2595) - 1)

    # mel scale
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = mel_to_hz(mels)

    # convert to FFT bins
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(1, n_mels + 1):
        f_m0 = bins[m - 1]
        f_m1 = bins[m]
        f_m2 = bins[m + 1]

        if f_m0 < f_m1:
            fb[m - 1, f_m0:f_m1] = (np.arange(f_m0, f_m1) - f_m0) / (f_m1 - f_m0)
        if f_m1 < f_m2:
            fb[m - 1, f_m1:f_m2] = (f_m2 - np.arange(f_m1, f_m2)) / (f_m2 - f_m1)

    return fb


MEL_FB = mel_filterbank(TARGET_SR, n_fft=2048, n_mels=N_MELS)


def make_fingerprint_from_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV/WEBM → mel → log → mean → normalized vector."""
    try:
        y, sr = sf.read(io.BytesIO(wav_bytes))
    except Exception as e:
        raise ValueError(f"soundfile failed: {e}")

    if y.ndim > 1:
        y = y.mean(axis=1)

    # resample if needed
    if sr != TARGET_SR:
        # simple linear resample
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, int(len(y) * TARGET_SR / sr))
        y = np.interp(x_new, x_old, y)

    # STFT
    n_fft = 2048
    hop = 512

    frames = []
    for i in range(0, len(y) - n_fft, hop):
        win = y[i:i+n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(win))
        frames.append(spec)

    if len(frames) == 0:
        raise ValueError("Not enough audio")

    S = np.array(frames).T  # shape (freq, time)

    # mel projection
    mel = MEL_FB @ S

    mel = np.maximum(mel, 1e-9)
    log_mel = np.log10(mel)

    feat = log_mel.mean(axis=1).astype(np.float32)

    # normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm

    return feat


# -----------------------------
#  MATCH ENDPOINT
# -----------------------------
@app.route("/match", methods=["POST"])
def match():
    raw = request.data
    if not raw:
        return jsonify({"error": "No audio received"}), 400

    try:
        q = make_fingerprint_from_bytes(raw)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    sims = F @ q
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    result = names[idx]

    return jsonify({
        "match": result,
        "confidence": round(score, 4)
    })


@app.route("/")
def health():
    return "BGM Matcher Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
