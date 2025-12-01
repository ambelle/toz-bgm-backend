# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import soundfile as sf
import io

TARGET_SR = 22050
N_MFCC = 20
DURATION = 15

# Load fingerprint database
data = np.load("fingerprints.npz", allow_pickle=True)
NAMES = data["names"]
VECTORS = data["vectors"]

app = Flask(__name__)
CORS(app)  # allow frontend to call API

def fingerprint_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Convert uploaded audio bytes → MFCC mean vector."""
    buf = io.BytesIO(audio_bytes)
    y, sr = sf.read(buf)

    if y.ndim > 1:
        y = y.mean(axis=1)  # stereo to mono

    if sr != TARGET_SR:
        y = librosa.resample(y, sr, TARGET_SR)

    y = y[:TARGET_SR * DURATION]
    if len(y) == 0:
        raise ValueError("Audio too short or invalid")

    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    return mfcc.mean(axis=1)

@app.route("/")
def home():
    return "TOZ BGM Finder backend is running."

@app.route("/recognize", methods=["POST"])
def recognize():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    try:
        q = fingerprint_from_bytes(audio_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Compare fingerprints
    dists = np.linalg.norm(VECTORS - q, axis=1)
    idx = int(dists.argmin())

    return jsonify({
        "match": str(NAMES[idx]),
        "distance": float(dists[idx])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
