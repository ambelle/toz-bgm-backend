import os
import numpy as np
import librosa

# If you keep it as "audio", leave this.
# If you rename the folder, change this to that folder name.
AUDIO_DIR = "audio"

TARGET_SR = 22050
N_MELS = 64  # MUST match app.py


def compute_fingerprint_from_file(path: str) -> np.ndarray:
    """Load audio file, compute log-mel mean vector, L2-normalized."""
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio in {path}")

    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    feat = log_mel.mean(axis=1).astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm

    return feat


def clean_name(path: str) -> str:
    """
    Use the filename (without extension) as the display name.

    e.g. "audio/Henesys.mp3" -> "Henesys"
         "audio/Kerning City.mp3" -> "Kerning City"
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def main():
    feats = []
    names = []

    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(f"{AUDIO_DIR} folder not found")

    for fname in sorted(os.listdir(AUDIO_DIR)):
        path = os.path.join(AUDIO_DIR, fname)
        if not os.path.isfile(path):
            continue

        lower = fname.lower()
        if not (lower.endswith(".mp3") or lower.endswith(".wav")):
            continue

        print(f"Processing: {path}")
        try:
            fvec = compute_fingerprint_from_file(path)
            feats.append(fvec)
            names.append(clean_name(path))
        except Exception as e:
            print(f"  !! Failed on {path}: {e}")

    if not feats:
        raise RuntimeError("No valid audio files found to fingerprint.")

    feats = np.vstack(feats).astype(np.float32)

    # Safety: ensure row-wise L2 normalization
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    feats = feats / norms

    names_arr = np.array(names, dtype=object)

    np.savez("fingerprints.npz", feats=feats, names=names_arr)

    print("Saved fingerprints.npz with:")
    print("  feats shape:", feats.shape)
    print("  names shape:", names_arr.shape)


if __name__ == "__main__":
    main()
