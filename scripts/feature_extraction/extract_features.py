import pandas as pd
import os
import numpy as np
import librosa

# Path to the CSV files
csv_path = os.path.join("dataset", "filtered", "train_filtered.csv")
features_csv_path = os.path.join("dataset", "filtered", "train_filtered_features.csv")

# Load the CSV
df = pd.read_csv(csv_path)

# Prepare columns for all features
df['stft_mean_logmag'] = np.nan
num_mfcc = 13
for i in range(1, num_mfcc + 1):
    df[f'mfcc_{i}'] = np.nan
df['rms_mean'] = np.nan
df['spectral_centroid_mean'] = np.nan
df['spectral_bandwidth_mean'] = np.nan
df['spectral_rolloff_mean'] = np.nan
df['zero_crossing_rate_mean'] = np.nan

for idx, row in df.iterrows():
    ebird_code = row['ebird_code']
    filename = row['filename']
    mp3_path = os.path.join("dataset", "raw", "A-M", ebird_code, filename)
    try:
        # Load audio file
        y, sr = librosa.load(mp3_path, sr=None)
        # Compute STFT
        stft = librosa.stft(y)
        stft_mag = np.abs(stft)
        stft_logmag = np.log1p(stft_mag)
        mean_logmag = np.mean(stft_logmag)
        df.at[idx, 'stft_mean_logmag'] = mean_logmag
        print(f"Extracted STFT feature for {mp3_path}: mean log-magnitude = {mean_logmag:.4f}")

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        for i in range(num_mfcc):
            df.at[idx, f'mfcc_{i+1}'] = mfccs_mean[i]
        print(f"Extracted MFCCs for {mp3_path}: {[round(val, 4) for val in mfccs_mean]}")

        # Compute RMS
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        df.at[idx, 'rms_mean'] = rms_mean
        print(f"Extracted RMS for {mp3_path}: mean RMS = {rms_mean:.4f}")

        # Compute Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        df.at[idx, 'spectral_centroid_mean'] = spectral_centroid_mean
        print(f"Extracted Spectral Centroid for {mp3_path}: mean = {spectral_centroid_mean:.4f}")

        # Compute Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        df.at[idx, 'spectral_bandwidth_mean'] = spectral_bandwidth_mean
        print(f"Extracted Spectral Bandwidth for {mp3_path}: mean = {spectral_bandwidth_mean:.4f}")

        # Compute Spectral Roll-Off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        df.at[idx, 'spectral_rolloff_mean'] = spectral_rolloff_mean
        print(f"Extracted Spectral Roll-Off for {mp3_path}: mean = {spectral_rolloff_mean:.4f}")

        # Compute Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        df.at[idx, 'zero_crossing_rate_mean'] = zero_crossing_rate_mean
        print(f"Extracted Zero Crossing Rate for {mp3_path}: mean = {zero_crossing_rate_mean:.4f}")

    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")

# Save updated features CSV
df.to_csv(features_csv_path, index=False)