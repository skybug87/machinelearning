import os
import noisereduce as nr
import soundfile as sf

RAW_DIR = os.path.join("dataset", "raw")
PROCESSED_DIR = os.path.join("dataset", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith(".wav"):
        raw_path = os.path.join(RAW_DIR, filename)
        processed_path = os.path.join(PROCESSED_DIR, filename)
        data, rate = sf.read(raw_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        sf.write(processed_path, reduced_noise, rate)