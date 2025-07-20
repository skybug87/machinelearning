import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
input_path = './dataset/numeric_features_with_spectrograms.csv'
output_path = './dataset/normalized_features_with_spectrograms.csv'

# Load data
df = pd.read_csv(input_path)

# Columns to exclude from normalization
exclude_cols = [
    'source', 'filename_original', 'filename_reduced', 'ebird_code', 'spectrogram_path_original', 'spectrogram_path_reduced', 'yamnet_embedding_path_original', 'yamnet_embedding_path_reduced',
]
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Normalize features (z-score)
scaler = StandardScaler()
df_norm = df.copy()
df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save normalized DataFrame
df_norm.to_csv(output_path, index=False)
print(f"Normalized features saved to {output_path}")