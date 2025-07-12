import os
import glob
import librosa
import numpy as np
import pandas as pd

# Paths
PROCESSED_DIR = os.path.join("dataset", "processed")
FEATURES_DIR = os.path.join("dataset", "features", "raw")
os.makedirs(FEATURES_DIR, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}

    # STFT
    stft = np.abs(librosa.stft(y))
    features['stft_mean'] = np.mean(stft)
    features['stft_std'] = np.std(stft)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(mfcc.shape[0]):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['centroid_mean'] = np.mean(centroid)
    features['centroid_std'] = np.std(centroid)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['bandwidth_mean'] = np.mean(bandwidth)
    features['bandwidth_std'] = np.std(bandwidth)

    # Spectral roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    return features

def infer_species_label(filename):
    # Assumes filename format: species_label_someid.wav
    base = os.path.basename(filename)
    label = base.split('_')[0]
    return label

def main():
    audio_files = glob.glob(os.path.join(PROCESSED_DIR, "*.wav"))
    if not audio_files:
        print("No audio files found in dataset/processed/.")
        return

    all_features = []
    for file_path in audio_files:
        features = extract_features(file_path)
        features['file'] = os.path.basename(file_path)
        features['species'] = infer_species_label(file_path)
        all_features.append(features)

    df = pd.DataFrame(all_features)
    output_path = os.path.join(FEATURES_DIR, "features.csv")
    df.to_csv(output_path, index=False)
    print(f"Extracted features saved to {output_path}")

if __name__ == "__main__":
    main()