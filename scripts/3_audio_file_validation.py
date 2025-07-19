# Step 3: Audio File Validation and Processing Setup
import os
import librosa
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import urllib.request
import pandas as pd

print("=== STEP 3: AUDIO FILE VALIDATION & PROCESSING SETUP ===\n")

# Load the filtered DataFrame from CSV
filtered_df = pd.read_csv('./dataset/filtered_df.csv')

# # Use the filtered dataset from Step 2
# # For this step, we'll work with the configuration from Step 2
# CONFIG = {
#     'MIN_RATING_THRESHOLD': 2.0,
#     'NUM_CLASSES': 30,
#     'MAX_DURATION': 300,
#     'MIN_DURATION': 5,
#     'MIN_SAMPLES_PER_CLASS': 20,
# }

# # Re-apply the same filtering logic to recreate df_final
# print("Recreating filtered dataset...")
# df_filtered = df[df['rating'] >= CONFIG['MIN_RATING_THRESHOLD']].copy()
# df_filtered = df_filtered[
#     (df_filtered['duration'] >= CONFIG['MIN_DURATION']) &
#     (df_filtered['duration'] <= CONFIG['MAX_DURATION'])
# ].copy()

# species_counts_filtered = df_filtered['ebird_code'].value_counts()
# valid_species = species_counts_filtered[species_counts_filtered >= CONFIG['MIN_SAMPLES_PER_CLASS']].index
# df_filtered = df_filtered[df_filtered['ebird_code'].isin(valid_species)].copy()

# final_species_counts = df_filtered['ebird_code'].value_counts()
# selected_species = final_species_counts.head(CONFIG['NUM_CLASSES']).index.tolist()
# df_final = df_filtered[df_filtered['ebird_code'].isin(selected_species)].copy()

# print(f"Filtered dataset ready: {len(df_final)} samples, {len(selected_species)} species")

final_species_counts = filtered_df['ebird_code'].value_counts()
selected_species = final_species_counts.index.tolist()

# ========================================
# 1. FIND AUDIO FILES AND VALIDATE PATHS
# ========================================
print(f"\n{'='*50}")
print("AUDIO FILE PATH VALIDATION:")
print("="*50)

# Look for the audio files
base_path = Path('/content/train_extended.csv')
print(f"Base path: {base_path}")
print(f"Base path exists: {base_path.exists()}")


# print(f"\nDirectories containing MP3 files:")
# for mp3_dir in mp3_dirs[:5]:  # Show first 5
#     mp3_count = len([f for f in os.listdir(mp3_dir) if f.endswith('.mp3')])
#     print(f"  - {mp3_dir}: {mp3_count} MP3 files")

# if len(mp3_dirs) > 5:
#     print(f"  ... and {len(mp3_dirs)-5} more directories")

# ========================================
# 2. CREATE FULL FILE PATHS
# ========================================
print(f"\n{'='*50}")
print("CREATING FILE PATHS:")
print("="*50)

def find_audio_file_path(filename, base_dirs):
    """Find the full path of an audio file"""
    for base_dir in base_dirs:
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            return full_path
    return None


# Check how many files we found
files_found = filtered_df['url'].notna().sum()
files_missing = filtered_df['url'].isna().sum()

print(f"Files found: {files_found}")
print(f"Files missing: {files_missing}")
print(f"Success rate: {files_found/(files_found+files_missing)*100:.1f}%")


# ========================================
# 3. AUDIO VALIDATION AND BASIC ANALYSIS
# ========================================
print(f"\n{'='*50}")
print("AUDIO FILE VALIDATION:")
print("="*50)

# Test loading a few random audio files

filtered_df['download_url'] = filtered_df['url'].astype(str).str.rstrip('/') + '/download'

sample_files = filtered_df.sample(n=min(5, len(filtered_df)))

# Create a directory to store downloaded audio
os.makedirs("temp_audio", exist_ok=True)




audio_info = []

for idx, row in sample_files.iterrows():
    try:


        # Get file name from URL
        url = row['download_url']
        filename = os.path.join("temp_audio", url.split("/")[-2] + '.mp3')

        # Download if not already present
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded: {filename}")

        # Load audio
        audio, sr = librosa.load(filename, sr=None, duration=10)
        duration = len(audio) / sr
        print("Testing audio file loading...")

        info = {
            'filename': row['filename'],
            'species': row['ebird_code'],
            'csv_duration': row['duration'],
            'actual_duration': duration,
            'sample_rate': sr,
            'channels_csv': row['channels'],
            'audio_shape': audio.shape,
            'success': True
        }
        audio_info.append(info)
        print(f" {row['download_url']}: {duration:.1f}s, {sr}Hz, shape={audio.shape}")

    except Exception as e:
        info = {
            'filename': row['url'],
            'species': row['ebird_code'],
            'error': str(e),
            'success': False
        }
        audio_info.append(info)
        print(f" {row['url']}: Error - {str(e)}")

# ========================================
# 4. DATASET SPLIT PREPARATION
# ========================================
print(f"\n{'='*50}")
print("DATASET SPLIT PREPARATION:")
print("="*50)

# Create class mapping
class_mapping = {species: idx for idx, species in enumerate(selected_species )}
reverse_mapping = {idx: species for species, idx in class_mapping.items()}

print("Class mapping created:")
for species, idx in list(class_mapping.items())[:5]:
    print(f"  {idx}: {species}")
print(f"  ... (showing first 5 of {len(class_mapping)} classes)")

# Add numeric labels
filtered_df['class_id'] = filtered_df['ebird_code'].map(class_mapping)

# Prepare for stratified split
print(f"\nPreparing stratified split...")
print(f"Class distribution before split:")
class_dist = filtered_df['ebird_code'].value_counts()
print(class_dist.head(10))


# ========================================
# 5. AUDIO PROCESSING CONFIGURATION
# ========================================
print(f"\n{'='*50}")
print("AUDIO PROCESSING CONFIGURATION:")
print("="*50)

AUDIO_CONFIG = {
    'SAMPLE_RATE': 22050,           # Standard sample rate for audio ML
    'MAX_AUDIO_LENGTH': 10,         # Maximum audio length in seconds
    'N_MELS': 128,                  # Number of mel bands for spectrogram
    'N_FFT': 2048,                  # FFT window size
    'HOP_LENGTH': 512,              # Hop length for STFT
    'SPECTROGRAM_HEIGHT': 128,      # Height of spectrogram image
    'SPECTROGRAM_WIDTH': 432,       # Width of spectrogram image (for 10s audio)
}

print("Audio processing configuration:")
for key, value in AUDIO_CONFIG.items():
    print(f"  {key}: {value}")

# Calculate expected spectrogram dimensions
expected_time_steps = (AUDIO_CONFIG['MAX_AUDIO_LENGTH'] * AUDIO_CONFIG['SAMPLE_RATE']) // AUDIO_CONFIG['HOP_LENGTH']
print(f"\nExpected spectrogram shape: ({AUDIO_CONFIG['N_MELS']}, {expected_time_steps})")

# ========================================
# 6. SUMMARY FOR NEXT STEPS
# ========================================
print(f"\n{'='*50}")
print("STEP 3 SUMMARY:")
print("="*50)
print(f" Audio files validated: {files_found}/{files_found+files_missing} found")
print(f" Dataset ready: {len(filtered_df)} samples")
print(f" Classes: {len(selected_species)} species")
print(f" Class mapping created")
print(f" Audio processing config set")
print(f" Ready for train/validation/test split")

# Save some key info for next step
STEP3_INFO = {
    'df_final': filtered_df,
    'class_mapping': class_mapping,
    'reverse_mapping': reverse_mapping,
    'selected_species': selected_species,
    'audio_config': AUDIO_CONFIG,
    'config': CONFIG
}

print(f"\n Next step: Train/Validation/Test split and first audio preprocessing")