# Step 3: Audio File Validation and Processing Setup
import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import random
from tqdm import tqdm
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from configurations import CONFIG
import noisereduce as nr

print("=== STEP 3: AUDIO FILE VALIDATION & PROCESSING SETUP ===\n")

# Load the filtered DataFrame from CSV
filtered_df = pd.read_csv('./dataset/filtered_df.csv')

# Create ./dataset/processed directory if it doesn't exist
processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Create new dataframe to store processed file paths
processed_df = pd.DataFrame(columns=['filename', 'ebird_code'])


def apply_noise_reduction(y, sr):
    """
    Apply noise reduction to the audio signal using spectral gating.
    Estimates noise profile from the first 0.5 seconds.
    """
    noise_duration = int(0.5 * sr)
    noise_clip = y[:noise_duration]
    reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)
    return reduced_noise

# Function to validate and process audio files
def validate_and_process_audio(row):
    filename = row['filename']
    #will save as WAV instead of MP3
    target_filename = filename.replace('.mp3', '')
    ebird_code = row['ebird_code']
    
    # Full path to the audio file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'raw', 'A-M', ebird_code, filename)
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        print(f"Attempting to download from url: {row['url']}")
        # Download the file if it doesn't exist
        try:
            urllib.request.urlretrieve(row['url'], file_path)
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")
            print("Skipping this file.")
            return None
    
    # if the output file already exists, skip processing
    processed_file_path = os.path.join(processed_dir, f"{ebird_code}_{target_filename}")
    if os.path.isfile(processed_file_path):
        print(f"File already processed: {processed_file_path}")
        return {'filename': str(processed_file_path), 'ebird_code': ebird_code}
    
    # Load the audio file
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        # Convert to mono if not already
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # Apply noise reduction
        y_reduced = apply_noise_reduction(y, sr)

        processed_file_path = os.path.join(processed_dir, f"{ebird_code}_{target_filename}_original.wav")
        reduced_processed_file_path = os.path.join(processed_dir, f"{ebird_code}_{target_filename}_reduced.wav")
        sf.write(processed_file_path, y, sr, format='WAV', subtype='PCM_16')
        sf.write(reduced_processed_file_path, y_reduced, sr, format='WAV', subtype='PCM_16')
        
        return {'filename_1': str(processed_file_path), 'filename_2': str(reduced_processed_file_path), 'ebird_code': ebird_code}
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process each row in the filtered DataFrame
for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Processing audio files"):
    processed_row = validate_and_process_audio(row)
    if processed_row:
        processed_df = pd.concat([processed_df, pd.DataFrame([processed_row])], ignore_index=True)
# Save the processed DataFrame to CSV
processed_df.to_csv(os.path.join(processed_dir, 'processed_audio_files.csv'), index=False)
print(f"Processed audio files saved to {processed_dir}/processed_audio_files.csv")

# Summary of processed files
print(f"\nProcessed {processed_df.shape[0]} audio files successfully.")
updatedvalue_counts = processed_df['ebird_code'].value_counts()
print("Species value counts sorted: ", updatedvalue_counts)

# create a histogram of the number of samples per species
plt.figure(figsize=(10, 6))
processed_df['ebird_code'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Species (ebird_code)')
plt.ylabel('Number of Samples')
plt.title('Number of Samples per Species')
plt.legend()
plt.tight_layout()
plt.savefig('./outputs/samples_per_species_histogram.png')
print(f"Histogram saved to './outputs/samples_per_species_histogram.png'")