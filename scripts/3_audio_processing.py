import os
import librosa
import numpy as np
import soundfile as sf
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import noisereduce as nr
from tqdm import tqdm

print("=== STEP 3: AUDIO FILE VALIDATION & PROCESSING SETUP ===\n")

# Load the filtered DataFrame from CSV
filtered_df = pd.read_csv('./dataset/filtered_df.csv')

# Create ./dataset/processed directory if it doesn't exist
processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'processed')
os.makedirs(processed_dir, exist_ok=True)

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
    # will save as WAV instead of MP3
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
    
    # if both original and reduced files already exist, skip processing
    original_path = os.path.join(processed_dir, f"{ebird_code}_{target_filename}_original.wav")
    reduced_path = os.path.join(processed_dir, f"{ebird_code}_{target_filename}_reduced.wav")
    output_data = [{
        'source': row['title'], # used to group segments later on
        'filename_original': str(original_path),
        'filename_reduced': str(reduced_path),
        'ebird_code': ebird_code,
    }]
    if os.path.isfile(original_path) and os.path.isfile(reduced_path):
        print(f"Files already processed: {row['title']}")
        return output_data

    # Load the audio file
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        # Convert to mono if not already
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        segment_length = 20  # seconds
        segment_samples = int(segment_length * sr)

        # Always crop the first 20 seconds (or full file if shorter)
        segment = y[:segment_samples] if len(y) > segment_samples else y

        # Apply noise reduction after cropping
        segment_reduced = apply_noise_reduction(segment, sr)

        sf.write(original_path, segment, sr, format='WAV', subtype='PCM_16')
        sf.write(reduced_path, segment_reduced, sr, format='WAV', subtype='PCM_16')

        return output_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Create new dataframe to store processed file paths
processed_df = pd.DataFrame(columns=['source', 'filename_original', 'filename_reduced', 'ebird_code'])

# Process each row in the filtered DataFrame
for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Processing audio files"):
    processed_row = validate_and_process_audio(row)
    if processed_row:
        processed_df = pd.concat([processed_df, pd.DataFrame(processed_row)], ignore_index=True)
# Save the processed DataFrame to CSV
processed_df.to_csv('./dataset/processed_audio_files.csv', index=False)
print("Processed audio files saved to ./dataset/processed_audio_files.csv")

# Summary of processed files
print(f"\nProcessed {processed_df.shape[0]} audio files successfully.")
updatedvalue_counts = processed_df['ebird_code'].value_counts()
print("Species value counts sorted: ", updatedvalue_counts)

# create a histogram of the number of samples per species
if not processed_df.empty and processed_df['ebird_code'].notnull().any():
    plt.figure(figsize=(10, 6))
    processed_df['ebird_code'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Species (ebird_code)')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Species')
    plt.tight_layout()
    plt.savefig('./outputs/samples_per_species_histogram.png')
    print("Histogram saved to './outputs/samples_per_species_histogram.png'")
else:
    print("No processed files to plot. Skipping histogram.")


##########################################################################################
# Extract Features from Processed Audio Files
##########################################################################################
features_df = processed_df.copy()

# Prepare columns for all features
features_df['stft_mean_logmag'] = np.nan
features_df['stft_mean_logmag_reduced'] = np.nan
num_mfcc = 13
for i in range(1, num_mfcc + 1):
    features_df[f'mfcc_{i}'] = np.nan
    features_df[f'mfcc_{i}_reduced'] = np.nan
features_df['rms_mean'] = np.nan
features_df['rms_mean_reduced'] = np.nan
features_df['spectral_centroid_mean'] = np.nan
features_df['spectral_centroid_mean_reduced'] = np.nan
features_df['spectral_bandwidth_mean'] = np.nan
features_df['spectral_bandwidth_mean_reduced'] = np.nan
features_df['spectral_rolloff_mean'] = np.nan
features_df['spectral_rolloff_mean_reduced'] = np.nan
features_df['zero_crossing_rate_mean'] = np.nan
features_df['zero_crossing_rate_mean_reduced'] = np.nan

def extract_features_for_file(audio_path, idx, suffix=""):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        # Compute STFT
        stft = librosa.stft(y)
        stft_mag = np.abs(stft)
        stft_logmag = np.log1p(stft_mag)
        mean_logmag = np.mean(stft_logmag)
        features_df.at[idx, f'stft_mean_logmag{suffix}'] = mean_logmag

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        for i in range(num_mfcc):
            features_df.at[idx, f'mfcc_{i+1}{suffix}'] = mfccs_mean[i]

        # Compute RMS
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        features_df.at[idx, f'rms_mean{suffix}'] = rms_mean

        # Compute Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        features_df.at[idx, f'spectral_centroid_mean{suffix}'] = spectral_centroid_mean

        # Compute Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        features_df.at[idx, f'spectral_bandwidth_mean{suffix}'] = spectral_bandwidth_mean

        # Compute Spectral Roll-Off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        features_df.at[idx, f'spectral_rolloff_mean{suffix}'] = spectral_rolloff_mean

        # Compute Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        features_df.at[idx, f'zero_crossing_rate_mean{suffix}'] = zero_crossing_rate_mean

    except Exception as e:
        print(f"Error extracting features for {audio_path}: {e}")
        pass

print("=== STEP 3 - Part 2: FEATURE EXTRACTION FROM AUDIO FILES ===\n")
for idx, row in tqdm(features_df.iterrows(), total=features_df.shape[0], desc="Extracting features"):
    ebird_code = row['ebird_code']
    original_path = row['filename_original']
    reduced_path = row['filename_reduced']

    # Extract features for original audio
    extract_features_for_file(original_path, idx, suffix="")

    # Extract features for noise reduced audio
    extract_features_for_file(reduced_path, idx, suffix="_reduced")

# Save updated features CSV
features_df.to_csv('./dataset/numeric_features.csv', index=False)