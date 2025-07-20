import pandas as pd
import os 

from tqdm import tqdm

# Load filtered DataFrame
df = pd.read_csv('dataset/numeric_features.csv', index_col=0)

# Create a copy for processing and tracking spectrogram paths
df_copy = df.copy()
df_copy['spectrogram_path_original'] = pd.NA
df_copy['spectrogram_path_reduced'] = pd.NA

# create .datasets/spectrograms directory if it doesn't exist
spectrograms_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'spectrograms')
os.makedirs(spectrograms_dir, exist_ok=True)

import numpy as np
import librosa

# Collect all audio paths to process
audio_paths = []
for idx, row in df.iterrows():
    for col in ['filename_original', 'filename_reduced']:
        audio_path = row[col]
        if pd.isnull(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
            continue
        audio_paths.append(audio_path)

# Process with tqdm progress bar
for idx, row in tqdm(df_copy.iterrows(), total=df_copy.shape[0], desc="Processing audio files"):
    for col, spec_col in [('filename_original', 'spectrogram_path_original'), ('filename_reduced', 'spectrogram_path_reduced')]:
        audio_path = row[col]
        if pd.isnull(audio_path) or not isinstance(audio_path, str) or not audio_path.strip():
            continue
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, power=2.0)
            # Resize spectrogram to fixed shape (128, 432) for consistency with deep learning pipeline
            import tensorflow as tf
            mel_spec_resized = tf.image.resize(mel_spec[..., np.newaxis], [128, 432]).numpy().squeeze()
            # Save as .npy in dataset/spectrograms/
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(spectrograms_dir, f"{basename}_mel.npy")
            np.save(out_path, mel_spec_resized)
            # Record the spectrogram path in the copy
            df_copy.at[idx, spec_col] = out_path
        except Exception as e:
            print(f"Failed for {audio_path}: {e}")

# Save the updated DataFrame copy to a new CSV
# Plot and save the first spectrogram as an image
import matplotlib.pyplot as plt

# Find the first available spectrogram file path
first_spec_path = None
for col in ['spectrogram_path_original', 'spectrogram_path_reduced']:
    paths = df_copy[col].dropna().tolist()
    if paths:
        first_spec_path = paths[0]
        break

if first_spec_path and os.path.exists(first_spec_path):
    spec = np.load(first_spec_path)
# Plot and save the first original and reduced spectrograms as images
for col, label, out_img in [
    ('spectrogram_path_original', 'Original Mel Spectrogram', 'outputs/first_spectrogram_original.png'),
    ('spectrogram_path_reduced', 'Reduced Mel Spectrogram', 'outputs/first_spectrogram_reduced.png')
]:
    paths = df_copy[col].dropna().tolist()
    if paths:
        spec_path = paths[0]
        if os.path.exists(spec_path):
            spec = np.load(spec_path)
            plt.figure(figsize=(10, 4))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
            plt.title(label)
            plt.xlabel('Time')
            plt.ylabel('Mel Frequency')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(out_img)
            plt.close()
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.title('First Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('outputs/first_spectrogram.png')
    plt.close()
df_copy.to_csv('dataset/numeric_features_with_spectrograms.csv')
