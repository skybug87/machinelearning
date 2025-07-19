# Step 1: Initial Setup and Data Exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')



# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== BIRD SPECIES AUDIO CLASSIFICATION PROJECT ===")
print("Step 1: Initial Setup and Data Exploration\n")

# Load the metadata CSV
print("Loading train_extended.csv...")
df = pd.read_csv('/content/drive/MyDrive/machinelearning/train_extended.csv')

print(f"Dataset loaded successfully!")
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
print(f"\nSPECIES INFORMATION:")
print("="*50)
print(f"Total unique species: {df['ebird_code'].nunique()}")
print(f"Species list: {sorted(df['ebird_code'].unique())}")

# Check rating distribution
print(f"\nRATING DISTRIBUTION:")
print("="*50)
rating_counts = df['rating'].value_counts().sort_index()
print(rating_counts)

# Check duration statistics
print(f"\nDURATION STATISTICS:")
print("="*50)
print(f"Min duration: {df['duration'].min():.2f}s")
print(f"Max duration: {df['duration'].max():.2f}s")
print(f"Mean duration: {df['duration'].mean():.2f}s")
print(f"Median duration: {df['duration'].median():.2f}s")