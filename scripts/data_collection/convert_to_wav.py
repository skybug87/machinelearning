import os
import librosa
import soundfile as sf

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'raw')

def is_mono_wav(filepath):
    if not filepath.lower().endswith('.wav'):
        return False
    try:
        with sf.SoundFile(filepath) as f:
            return f.channels == 1
    except Exception:
        return False

def convert_to_mono_wav(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    sf.write(filepath, y, sr, subtype='PCM_16')

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            if not is_mono_wav(fpath):
                try:
                    convert_to_mono_wav(fpath)
                    print(f"Converted: {fpath}")
                except Exception as e:
                    print(f"Failed to convert {fpath}: {e}")
                    with open("failed_conversions.log", "a", encoding="utf-8") as logf:
                        logf.write(f"{fpath}: {e}\n")

if __name__ == "__main__":
    process_directory(RAW_DIR)