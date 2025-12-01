# build_fingerprints.py
import numpy as np
import librosa
from bgm_list import TRACKS

TARGET_SR = 22050
DURATION = 15        # use first 15 seconds of each track
N_MFCC = 20

def make_fingerprint(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    y = y[:TARGET_SR * DURATION]
    if len(y) == 0:
        raise ValueError(f"Empty or invalid audio for {path}")
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    return mfcc.mean(axis=1)

def main():
    names = []
    vectors = []

    for name, path in TRACKS:
        print("Processing:", name)
        fp = make_fingerprint(path)
        names.append(name)
        vectors.append(fp)

    names = np.array(names, dtype=object)
    vectors = np.vstack(vectors)

    np.savez("fingerprints.npz", names=names, vectors=vectors)
    print("Saved fingerprints.npz")

if __name__ == "__main__":
    main()
