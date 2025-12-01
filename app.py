from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import os

from bgm_list import TRACKS

TARGET_SR = 22050
FINGERPRINTS_PATH = "fingerprints.npz"

# Load fingerprints once at startup
data = np.load(FINGERPRINTS_PATH, allow_pickle=True)
NAMES = data["names"]
FPS = data["fps"]

app = Flask(__name__)
CORS(app)


def make_fingerprint(path: str):
    """
    Load an audio file from disk and compute a simple spectral
    fingerprint compatible with the ones in fingerprints.npz.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec = np.mean(S, axis=1)
    spec = spec / (np.linalg.norm(spec) + 1e-10)
    return spec


@app.route("/")
def home():
    return "TOZ BGM Finder backend is running."


@app.route("/match", methods=["POST"])
def match():
    # The scanner sends the blob as "file"
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save to a temporary file so librosa can use its normal path-based loaders
    tmp_path = "/tmp/upload_audio.webm"
    try:
        f.save(tmp_path)

        # Compute fingerprint for the uploaded clip
        query_fp = make_fingerprint(tmp_path)

        # Compare against existing fingerprints
        # FPS shape: (num_tracks, feature_dim)
        # query_fp shape: (feature_dim,)
        dists = np.linalg.norm(FPS - query_fp, axis=1)
        best_idx = int(np.argmin(dists))
        best_name = str(NAMES[best_idx])
        best_dist = float(dists[best_idx])

        return jsonify({
            "match": best_name,
            "distance": best_dist
        })

    except Exception as e:
        return jsonify({
            "error": f"Error processing audio: {str(e)}"
        }), 500

    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # For local testing only. On Render, gunicorn runs this.
    app.run(host="0.0.0.0", port=5000)
