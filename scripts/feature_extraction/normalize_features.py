import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES_PATH = os.path.join("dataset", "features", "raw", "features.csv")
NORMALIZED_FEATURES_PATH = os.path.join("dataset", "features", "normalized", "features.csv")

def main():
    # Check if input file exists
    if not os.path.exists(RAW_FEATURES_PATH):
        raise FileNotFoundError(f"Input file not found: {RAW_FEATURES_PATH}")

    # Load raw features
    df = pd.read_csv(RAW_FEATURES_PATH)

    # Identify non-feature columns (assume first two: filename, label)
    non_feature_cols = ["filename", "label"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Normalize feature columns
    scaler = StandardScaler()
    df_features = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )

    # Combine non-feature and normalized feature columns
    df_out = pd.concat([df[non_feature_cols], df_features], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(NORMALIZED_FEATURES_PATH), exist_ok=True)

    # Save normalized features
    df_out.to_csv(NORMALIZED_FEATURES_PATH, index=False)

if __name__ == "__main__":
    main()