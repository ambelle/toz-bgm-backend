import os
from pathlib import Path

import numpy as np
import librosa
from scipy import signal

# =====================
# CONFIG (same as app.py)
# =====================

TARGET_SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
FMIN = 0.0
FMAX = TARGET_SR / 2.0

AUDIO_DIR = Path("audio")
OUT_PATH = Path("fingerprints.npz")


# =====================
# MEL FILTER (same as app.py)
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

        left_inds = np.logical_and(fft_freqs >= f_left, fft_freqs <= f_center)
        if f_center > f_left:
            fb[m - 1, left_inds] = (
                (fft_freqs[left_inds] - f_left) / (f_center - f_left)
            )

        right_inds = np.logical_and(fft_freqs >= f_center, fft_freqs <= f_right)
        if f_right > f_center:
            fb[m - 1, right_inds] = (
                (f_right - fft_freqs[right_inds]) / (f_right - f_center)
            )

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
# FINGERPRINT PIPELINE (same as app.py but using librosa to load mp3)
# =====================

def waveform_to_fingerprint_float(wave: np.ndarray, sr: int) -> np.ndarray:
    """Same logic as backend's waveform_to_fingerprint, but input is already float32 [-1,1]."""
    if wave.ndim > 1:
        wave = wave.mean(axis=0)

    # Remove DC offset
    wave = wave - np.mean(wave)

    if sr != TARGET_SR:
        wave = signal.resample_poly(wave, TARGET_SR, sr)
        sr = TARGET_SR

    if wave.size == 0:
        raise ValueError("Empty audio after preprocessing")

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

    power_spec = mag ** 2

    mel_spec = MEL_FILTER @ power_spec  # (N_MELS, T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log10(mel_spec).astype(np.float32)

    fp = log_mel.mean(axis=1)
    norm = np.linalg.norm(fp)
    if norm > 0.0:
        fp = fp / norm

    return fp.astype(np.float32)


def main():
    audio_files = sorted(
        [p for p in AUDIO_DIR.glob("*.mp3") if p.is_file()],
        key=lambda x: x.name.lower(),
    )

    if not audio_files:
        print(f"No .mp3 files found in {AUDIO_DIR}")
        return

    feats = []
    names = []

    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}:\n")

    for path in audio_files:
        print(f"Processing: {path}")
        try:
            # librosa.load: returns float32 in [-1, 1]
            y, sr = librosa.load(str(path), sr=None, mono=False)
            fp = waveform_to_fingerprint_float(y, sr)
            feats.append(fp)
            # Use nice clean track name from filename (without extension)
            names.append(path.stem)
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")

    if not feats:
        print("No valid fingerprints generated.")
        return

    F = np.stack(feats, axis=0).astype(np.float32)
    names_arr = np.array(names, dtype=object)

    np.savez(OUT_PATH, feats=F, names=names_arr)

    print(f"\nSaved {OUT_PATH} with:")
    print(f"  feats shape: {F.shape}")
    print(f"  names shape: {names_arr.shape}")


if __name__ == "__main__":
    main()
