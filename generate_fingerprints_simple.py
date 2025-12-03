import os
import numpy as np
import librosa  # only used locally, fine for Windows dev

# Must match app.py
TARGET_SR = 22050
N_FFT = 1024

AUDIO_DIR = "audio"
OUT_PATH = "fingerprints.npz"

def compute_fingerprint(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Same logic as app.py, but here y is already loaded by librosa.
    """
    if y.size == 0:
        raise ValueError("Empty audio signal")

    # DC remove + normalize
    y = y - np.mean(y)
    max_abs = np.max(np.abs(y)) + 1e-8
    y = y / max_abs

    # ensure float32
    y = y.astype(np.float32)

    if y.size < N_FFT:
        pad = N_FFT - y.size
        y = np.pad(y, (0, pad), mode="constant")

    hop = N_FFT // 2
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

    norm = np.linalg.norm(spec)
    if norm > 0:
        spec = spec / norm

    return spec


def main():
    features = []
    names = []

    if not os.path.isdir(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            continue

        path = os.path.join(AUDIO_DIR, fname)
        print(f"Processing: {path}")

        # Load with librosa at TARGET_SR mono
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        fp = compute_fingerprint(y, sr)
        features.append(fp)

        # friendly name: strip extension + double .mp3.mp3
        base = fname
        if base.lower().endswith(".mp3.mp3"):
            base = base[:-9]
        else:
            base = os.path.splitext(base)[0]
        names.append(base)

    features = np.stack(features, axis=0)
    names = np.array(names, dtype=object)

    np.savez_compressed(
        OUT_PATH,
        features=features,
        names=names,
    )

    print(f"Saved {OUT_PATH} with:")
    print(f"  features: {features.shape}")
    print(f"  names:    {names.shape}")


if __name__ == "__main__":
    main()
