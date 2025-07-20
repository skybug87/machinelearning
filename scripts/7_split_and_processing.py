# Step 4 Modified: Data Split and Audio Preprocessing (No External Dependencies)
import librosa
import numpy as np

# Hotfix for deprecated NumPy types used in older librosa versions
np.complex = np.complex128
np.float = float
np.int = int
np.bool = bool

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter, filtfilt
import warnings
import urllib.request
from urllib.error import HTTPError, URLError
import time
import os

warnings.filterwarnings('ignore')

GDRIVE_AUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudio_xtrain'
os.makedirs(GDRIVE_AUDIO_DIR, exist_ok=True)

GDRIVE_TESTAUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudioxtest'
os.makedirs(GDRIVE_TESTAUDIO_DIR, exist_ok=True)

GDRIVE_VALAUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudioxval'
os.makedirs(GDRIVE_VALAUDIO_DIR, exist_ok=True)

GDRIVE_YAUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudio_ytrain'
os.makedirs(GDRIVE_YAUDIO_DIR, exist_ok=True)

GDRIVE_YTESTAUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudioytest'
os.makedirs(GDRIVE_YTESTAUDIO_DIR, exist_ok=True)

GDRIVE_YVALAUDIO_DIR = '/content/drive/MyDrive/masters /machinelearning/birdaudioyval'
os.makedirs(GDRIVE_YVALAUDIO_DIR, exist_ok=True)



GDRIVE_MELSPEC_DIR = '/content/drive/MyDrive/masters /machinelearning/melspec'
os.makedirs(GDRIVE_MELSPEC_DIR, exist_ok=True)

print("=== STEP 4: DATA SPLIT & AUDIO PREPROCESSING (MODIFIED) ===\n")

AUDIO_CONFIG = {
    'SAMPLE_RATE': 22050,
    'MAX_AUDIO_LENGTH': 10,
    'N_MELS': 128,
    'N_FFT': 2048,
    'HOP_LENGTH': 512,
    'SPECTROGRAM_HEIGHT': 128,
    'SPECTROGRAM_WIDTH': 432,
}
# ========================================
# 1. STRATIFIED TRAIN/VALIDATION/TEST SPLIT
# ========================================
# The following block is now obsolete and replaced by loading precomputed splits from CSVs above.
# print(f"\n{'='*50}")
# print("STRATIFIED DATA SPLIT:")
# print("="*50)
#
# # First split: train+val vs test (80/20)
# X = filtered_df[['download_url', 'filename', 'ebird_code']].copy()
# y = filtered_df['class_id'].copy()
#
# X_temp, X_test, y_temp, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # Second split: train vs validation (75/25 of remaining = 60/20 of total)
# X_train, X_val, y_train, y_val = train_test_split(
#     X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
# )
#
# print(f"Data split completed:")
# print(f"  Training set:   {len(X_train):,} samples ({len(X_train)/len(filtered_df)*100:.1f}%)")
# print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(filtered_df)*100:.1f}%)")
# print(f"  Test set:       {len(X_test):,} samples ({len(X_test)/len(filtered_df)*100:.1f}%)")
#
# # Verify class distribution is maintained
# print(f"\nClass distribution verification:")
# train_dist = y_train.value_counts().sort_index()
# val_dist = y_val.value_counts().sort_index()
# test_dist = y_test.value_counts().sort_index()
#
# sample_classes = train_dist.index[:5]
# for class_id in sample_classes:
#     species = selected_species[class_id]
#     total_samples = train_dist[class_id] + val_dist[class_id] + test_dist[class_id]
#     train_pct = train_dist[class_id] / total_samples * 100
#     val_pct = val_dist[class_id] / total_samples * 100
#     test_pct = test_dist[class_id] / total_samples * 100
#     print(f"  {species}: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")

# ========================================
# 2. AUDIO PREPROCESSING FUNCTIONS
# ========================================
print(f"\n{'='*50}")
print("AUDIO PREPROCESSING FUNCTIONS:")
print("="*50)

def apply_highpass_filter(audio, sr, cutoff_freq=300):
    """Apply high-pass filter to remove low-frequency noise"""
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99  # Prevent error if cutoff too high
    b, a = butter(5, normalized_cutoff, btype='high')
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def apply_bandpass_filter(audio, sr, low_freq=300, high_freq=8000):
    """Apply band-pass filter to keep bird frequency range"""
    nyquist = sr / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist

    # Ensure frequencies are in valid range
    low_normalized = max(0.01, min(low_normalized, 0.99))
    high_normalized = max(low_normalized + 0.01, min(high_normalized, 0.99))

    b, a = butter(5, [low_normalized, high_normalized], btype='band')
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def reduce_noise_spectral_subtraction(audio, sr, noise_factor=0.5):
    """Simple spectral subtraction for noise reduction"""
    try:
        # Convert to frequency domain
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Estimate noise from the first 10% of the signal
        noise_length = len(audio) // 10
        noise_magnitude = np.mean(np.abs(np.fft.fft(audio[:noise_length])))

        # Subtract noise estimate
        clean_magnitude = magnitude - noise_factor * noise_magnitude
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)  # Floor at 10% of original

        # Reconstruct signal
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.real(np.fft.ifft(clean_fft))

        return clean_audio
    except Exception as e:
        print(f"Warning: Spectral subtraction failed, returning filtered audio. Error: {e}")
        return audio

def normalize_audio(audio):
    """Normalize audio to [-1, 1] range"""
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio

def load_and_preprocess_audio(file_path, target_sr=22050, max_length=10):
    """Load and preprocess a single audio file"""
    try:
        # Load audio
        filename = os.path.join("temp_audio", url.split("/")[-2] + '.mp3')



        # Load audio
        #audio, sr = librosa.load(filename, sr=None, duration=10)
        audio, sr = librosa.load(filename, sr=target_sr, duration=max_length)

        # Apply band-pass filter to focus on bird frequency range
        audio_filtered = apply_bandpass_filter(audio, sr, low_freq=300, high_freq=8000)

        # Apply simple noise reduction
        audio_denoised = reduce_noise_spectral_subtraction(audio_filtered, sr)

        # Normalize
        audio_normalized = normalize_audio(audio_denoised)

        # Ensure consistent length (pad or truncate)
        target_length = int(max_length * target_sr)
        if len(audio_normalized) < target_length:
            # Pad with zeros
            audio_normalized = np.pad(audio_normalized, (0, target_length - len(audio_normalized)))
        else:
            # Truncate
            audio_normalized = audio_normalized[:target_length]

        return audio_normalized, sr

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
def save_melspectrogram_to_drive(audio, sr, ebird_code, rec_id, base_dir=GDRIVE_MELSPEC_DIR):
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=AUDIO_CONFIG['N_MELS'],
            n_fft=AUDIO_CONFIG['N_FFT'],
            hop_length=AUDIO_CONFIG['HOP_LENGTH'],
            power=2.0
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        filename = f"{ebird_code}_{rec_id}.npy"
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            np.save(filepath, mel_db)
            print(f"Saved mel-spec to: {filepath}")
        else:
            print(f"Skipped (already exists): {filepath}")

        return True
    except Exception as e:
        print(f"Failed saving mel-spec for {rec_id}: {e}")
        return False
def audio_to_melspectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


print("Audio preprocessing functions defined:")
print("  - Band-pass filtering (300-8000 Hz for bird sounds)")
print("  - Spectral subtraction noise reduction")
print("  - Audio normalization")
print("  - Mel-spectrogram generation")


def fetch_to_drive(url: str, base_folder=GDRIVE_AUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def fetch_xtestto_drive(url: str, base_folder=GDRIVE_TESTAUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def fetch_xvalto_drive(url: str, base_folder=GDRIVE_VALAUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def fetch_yvalto_drive(url: str, base_folder=GDRIVE_YVALAUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def fetch_ytestto_drive(url: str, base_folder=GDRIVE_YTESTAUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def fetch_ytrainto_drive(url: str, base_folder=GDRIVE_YAUDIO_DIR, max_retry=1, sleep_secs=3):
    """Download audio to Google Drive"""
    rec_id = url.rstrip("/").split("/")[-2]
    local_path = os.path.join(base_folder, f"{rec_id}.mp3")

    if os.path.exists(local_path):
        return local_path  # already downloaded

    for attempt in range(max_retry):
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except (HTTPError, URLError) as e:
            print(f"   ⚠️  {rec_id}: {e} (attempt {attempt+1}/{max_retry})")
            time.sleep(sleep_secs)

    return None

def download_audio(split_df, split_name):
  print(f"\n{'='*50}")
  print("DOWNLOADING VALIDATION AND TEST AUDIO TO GOOGLE DRIVE:")
  print("="*50)
  success_count = 0
  for idx, row in split_df.iterrows():
      url = row['download_url']
        # Choose correct audio fetcher

      if split_name== "X_val":
            local_file = fetch_xvalto_drive(url)
      elif split_name == "y_test":
          local_file = fetch_ytestto_drive(url)
      elif split_name == "X_test":
            local_file = fetch_xtestto_drive(url)



      if local_file is not None:
          success_count += 1
      else:
            print(f"⚠️  Failed to download {url}")
  print(f" {split_name} download complete: {success_count}/{len(split_df)} files downloaded.\n")

def download_audio_from_series(label_series, split_name, reference_df):
    """
    Downloads audio files using label Series (e.g., y_train), matched with reference DataFrame (e.g., X_train)
    """
    print(f"\n{'='*50}")
    print(f"DOWNLOADING AUDIO FOR {split_name.upper()} (from Series)")
    print("="*50)

    success_count = 0

    # Reindex to align with reference DataFrame
    matched_df = reference_df.loc[label_series.index]

    for idx, row in matched_df.iterrows():
        url = row['download_url']

        if split_name == "y_train":
            local_file = fetch_ytrainto_drive(url)
        elif split_name == "y_val":
            local_file = fetch_yvalto_drive(url)
        elif split_name == "y_test":
            local_file = fetch_ytestto_drive(url)
        else:
            print(f"⚠️ Unknown split name: {split_name}")
            continue

        if local_file is not None:
            success_count += 1
        else:
            print(f"⚠️ Failed to download {url}")

    print(f"{split_name} download complete: {success_count}/{len(label_series)} files downloaded.\n")


# ========================================
# 3. TEST PREPROCESSING ON SAMPLE FILES
# ========================================
print(f"\n{'='*50}")
print("TESTING PREPROCESSING PIPELINE:")
print("="*50)

# Use the entire training set
test_samples = X_train.copy()
n_samples = len(test_samples)

fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
if n_samples == 1:
    axes = np.expand_dims(axes, axis=0)

fig.suptitle('Audio Preprocessing Pipeline Demo', fontsize=16)

for idx, (_, row) in enumerate(test_samples.iterrows()):
    url = row['download_url']
    species = row['ebird_code']


    file_mp3 = fetch_to_drive(url)
    if file_mp3 is None or not os.path.exists(file_mp3):
      print(f"Skipping {species} — could not fetch or does not exist.")
      continue


    print(f"Processing sample {idx+1}/{n_samples}: {url} ({species})")

    try:
        original_audio, sr = librosa.load(file_mp3, sr=AUDIO_CONFIG['SAMPLE_RATE'], duration=3)

        filtered_audio = apply_bandpass_filter(original_audio, sr)
        denoised_audio = reduce_noise_spectral_subtraction(filtered_audio, sr)

        final_audio = normalize_audio(denoised_audio)

        rec_id = url.rstrip("/").split("/")[-2]
        mel_file_path = os.path.join(GDRIVE_MELSPEC_DIR, f"{species}_{rec_id}.npy")

        if os.path.exists(mel_file_path):
          print(f"Skipping {species}/{rec_id} — already saved.")
          continue

        original_spec = audio_to_melspectrogram(original_audio, sr,
                                        AUDIO_CONFIG['N_MELS'], AUDIO_CONFIG['N_FFT'], AUDIO_CONFIG['HOP_LENGTH'])
        final_spec = audio_to_melspectrogram(final_audio, sr,
                                     AUDIO_CONFIG['N_MELS'], AUDIO_CONFIG['N_FFT'], AUDIO_CONFIG['HOP_LENGTH'])

        # Save only preprocessed mel-spectrogram
        if final_spec is not None:
          np.save(mel_file_path, final_spec)
          print(f"Saved: {mel_file_path}")


        axes[idx, 0].plot(original_audio)
        axes[idx, 0].set_title(f'Original Audio\n{species}')
        axes[idx, 0].grid(True, alpha=0.3)

        axes[idx, 1].plot(final_audio)
        axes[idx, 1].set_title('Preprocessed Audio')
        axes[idx, 1].grid(True, alpha=0.3)

        axes[idx, 2].imshow(original_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[idx, 2].set_title('Original Spectrogram')

        axes[idx, 3].imshow(final_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[idx, 3].set_title('Preprocessed Spectrogram')

    except Exception as e:
        print(f"   Error: {e}")
        for j in range(4):
            axes[idx, j].text(0.5, 0.5, f'Error\n{str(e)[:30]}...',
                              ha='center', va='center', transform=axes[idx, j].transAxes)

download_audio(X_val, "X_val")
download_audio(X_test, "X_test")
download_audio_from_series(y_train, "y_train", X_train)
download_audio_from_series(y_val, "y_val", X_val)
download_audio_from_series(y_test, "y_test", X_test)

# def generate_mels_for_split(X_split, split_name):
#     print(f"\n{'='*50}")
#     print(f"GENERATING MEL-SPECTROGRAMS FOR {split_name.upper()} SET")
#     print("="*50)

#     for idx, (_, row) in enumerate(X_split.iterrows()):
#         url = row['download_url']
#         ebird_code = row['ebird_code']
#         rec_id = url.rstrip("/").split("/")[-2]
#         mel_file_path = os.path.join(GDRIVE_MELSPEC_DIR, f"{ebird_code}_{rec_id}.npy")

#         if os.path.exists(mel_file_path):
#             print(f"✓ Skipped {rec_id} — already exists")
#             continue

#         # Choose correct audio fetcher
#         if split_name == "train":
#             audio_path = fetch_to_drive(url)
#         elif split_name == "val":
#             audio_path = fetch_valto_drive(url)
#         elif split_name == "test":
#             audio_path = fetch_testto_drive(url)
#         else:
#             print(f"⚠️ Unknown split name: {split_name}")
#             continue

#         if audio_path is None:
#             print(f"⚠️ Skipping {rec_id} — audio file not available")
#             continue

#         try:
#             audio, sr = librosa.load(audio_path, sr=AUDIO_CONFIG['SAMPLE_RATE'], duration=AUDIO_CONFIG['MAX_AUDIO_LENGTH'])
#             audio = apply_bandpass_filter(audio, sr)
#             audio = reduce_noise_spectral_subtraction(audio, sr)
#             audio = normalize_audio(audio)

#             save_melspectrogram_to_drive(audio, sr, ebird_code, rec_id)
#         except Exception as e:
#             print(f"❌ Error processing {rec_id}: {e}")


#generate_mels_for_split(X_train, "train")
#generate_mels_for_split(X_val, "val")
#generate_mels_for_split(X_test, "test")


plt.tight_layout()
plt.show()


# ========================================
# 4. PREPROCESSING STATISTICS
# ========================================
print(f"\n{'='*50}")
print("PREPROCESSING PIPELINE SUMMARY:")
print("="*50)
print(f" Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
print(" Audio preprocessing pipeline tested (no external dependencies)")
print(f" Target audio length: {AUDIO_CONFIG['MAX_AUDIO_LENGTH']}s at {AUDIO_CONFIG['SAMPLE_RATE']}Hz")
print(f" Target spectrogram shape: ({AUDIO_CONFIG['N_MELS']}, ~430)")
print(f" Classes: {len(selected_species)} bird species")
print(f" Noise reduction: Band-pass filtering + spectral subtraction")

# Store data splits for next step
SPLITS_DATA = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'class_mapping': class_mapping,
    'selected_species': selected_species
}

def load_melspec_dataset(X, y, mel_dir, target_shape=(128, 432)):
    X_data, y_data = [], []

    for _, row in X.iterrows():
        rec_id = row['download_url'].split('/')[-2]
        ebird_code = row['ebird_code']
        filename = f"{ebird_code}_{rec_id}.npy"
        mel_path = os.path.join(mel_dir, filename)

        if os.path.exists(mel_path):
            mel = np.load(mel_path)

            # Resize or pad mel-spec
            if mel.shape[1] < target_shape[1]:
                # pad to the right
                pad_width = target_shape[1] - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
            elif mel.shape[1] > target_shape[1]:
                # crop to the target shape
                mel = mel[:, :target_shape[1]]

            X_data.append(mel)
            y_data.append(y.loc[row.name])
        else:
            print(f"Missing mel-spec file: {filename}")

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


mel_dir = GDRIVE_MELSPEC_DIR

sample_X = X_test.iloc[[0]]
sample_y = y_test.iloc[[0]]
X_test_np, y_test_np = load_melspec_dataset(sample_X, sample_y, mel_dir)

print(X_test_np.shape)
print(y_test_np)
print(f"\n Next step: CNN model architecture and batch data generation")