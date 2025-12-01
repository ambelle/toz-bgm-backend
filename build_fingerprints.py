import os
import numpy as np
import librosa

from bgm_list import TRACKS

TARGET_SR = 22050


def make_fingerprint(path: str):
    """
    Load an audio file from disk and compute a simple spectral fingerprint.
    Must match the logic that the backend uses.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec = np.mean(S, axis=1)
    spec = spec / (np.linalg.norm(spec) + 1e-10)
    return spec


def main():
    names = []
    fps = []

    for track in TRACKS:
        # Handle both dict and tuple formats
        if isinstance(track, dict):
            name = track["name"]
            filename = track["file"]
        else:
            # assume tuple/list like ("Amoria", "amoria.mp3.mp3") or ("Amoria", "bgm/amoria.mp3.mp3")
            if len(track) >= 2:
                name, filename = track[0], track[1]
            else:
                print(f"  !!! Unexpected track format, skipping: {track}")
                continue

        # If filename already has 'bgm/' at the start, don't prepend again
        if filename.lower().startswith("bgm/") or filename.lower().startswith("bgm\\"):
            path = filename
        else:
            path = os.path.join("bgm", filename)

        print(f"Processing: {name} ({path})")

        if not os.path.exists(path):
            print(f"  !!! File not found, skipping: {path}")
            continue

        fp = make_fingerprint(path)
        names.append(name)
        fps.append(fp)

    if not fps:
        print("No fingerprints generated. Check your bgm files and TRACKS.")
        return

    fps_array = np.vstack(fps)

    np.savez("fingerprints.npz", names=np.array(names, dtype=object), fps=fps_array)
    print("Saved fingerprints.npz with keys: names, fps")


if __name__ == "__main__":
    main()
