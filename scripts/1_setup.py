# Step 1: Initial Setup and Data Exploration
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


print("=== BEGIN BIRD SPECIES AUDIO CLASSIFICATION PROJECT ===")
print("=== STEP 1: Initial Setup and Data Exploration ===\n")

# Load the metadata CSV
print("Loading train_extended.csv...")
df = pd.read_csv('./dataset/train_extended.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n" + "="*50)

# Display basic info about the dataset
print("\nDATASET OVERVIEW:")
print("="*50)
df.info()

print("\nFIRST FEW ROWS:")
print("="*50)
print(df.head())

print("\nBASIC STATISTICS:")
print("="*50)
print(df.describe())

# Check for missing values
print("\nMISSING VALUES:")
print("="*50)
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# Explore unique species (ebird_code)
print("\nSPECIES INFORMATION:")
print("="*50)
print(f"Total unique species: {df['ebird_code'].nunique()}")
print(f"Species list: {sorted(df['ebird_code'].unique())}")

# Check rating distribution
print("\nRATING DISTRIBUTION:")
print("="*50)
rating_counts = df['rating'].value_counts().sort_index()
print(rating_counts)

# Check duration statistics
print("\nDURATION STATISTICS:")
print("="*50)
print(f"Min duration: {df['duration'].min():.2f}s")
print(f"Max duration: {df['duration'].max():.2f}s")
print(f"Mean duration: {df['duration'].mean():.2f}s")
print(f"Median duration: {df['duration'].median():.2f}s")