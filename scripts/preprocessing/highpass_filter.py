import os
import glob
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_directory(directory, cutoff=300.0, order=5):
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    for file_path in audio_files:
        data, samplerate = sf.read(file_path)
        # If stereo, process each channel
        if len(data.shape) == 1:
            filtered = highpass_filter(data, cutoff, samplerate, order)
        else:
            filtered = np.stack([highpass_filter(data[:, ch], cutoff, samplerate, order) for ch in range(data.shape[1])], axis=1)
        sf.write(file_path, filtered, samplerate)

if __name__ == "__main__":
    process_directory("dataset/processed")