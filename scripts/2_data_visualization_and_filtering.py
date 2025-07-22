# Script Lead: Crystal Matheny
# Collaborators: Duncan Hord, Yehong Huang, Sepehr Goshayeshi

import matplotlib.pyplot as plt
import pandas as pd

from configurations import CONFIG

print("=== STEP 2: DATA VISUALIZATION & CONFIGURATION ===\n")

# Load the metadata CSV
print("Loading train_extended.csv...")
df = pd.read_csv('./dataset/train_extended.csv')

# Filter out all bird codes that do not start with a letter between a and m
df = df[df['ebird_code'].str[0].str.lower().between('a', 'm')]

print("CURRENT CONFIGURATION:")
print("="*50)
for key, value in CONFIG.items():
    print(f"{key}: {value}")

print(f"\n{'='*50}")
print("DATA ANALYSIS BEFORE FILTERING:")
print("="*50)

# 1. Rating Distribution Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Rating distribution
axes[0,0].hist(df['rating'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(CONFIG['MIN_RATING_THRESHOLD'], color='red', linestyle='--',
                 label=f'Min Rating Threshold: {CONFIG["MIN_RATING_THRESHOLD"]}')
axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Rating Distribution')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Duration distribution
axes[0,1].hist(df['duration'], bins=200, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].axvline(CONFIG['MAX_DURATION'], color='red', linestyle='--',
                 label=f'Max Duration: {CONFIG["MAX_DURATION"]}s')
axes[0,1].axvline(CONFIG['MIN_DURATION'], color='orange', linestyle='--',
                 label=f'Min Duration: {CONFIG["MIN_DURATION"]}s')
axes[0,1].set_xlabel('Duration (seconds)')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Duration Distribution')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xlim(0, 500)  # Focus on reasonable duration range

# Species count distribution
species_counts = df['ebird_code'].value_counts().head(300) #crystal

axes[1,0].hist(species_counts.values, bins=30, alpha=0.7, color='coral', edgecolor='black')
axes[1,0].axvline(CONFIG['MIN_SAMPLES_PER_CLASS'], color='red', linestyle='--',
                 label=f'Min Samples: {CONFIG["MIN_SAMPLES_PER_CLASS"]}')
axes[1,0].set_xlabel('Number of Samples per Species')
axes[1,0].set_ylabel('Number of Species')
axes[1,0].set_title('Samples per Species Distribution')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Channel distribution
channel_counts = df['channels'].value_counts()
axes[1,1].pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%',
             colors=['lightblue', 'lightcoral'])
axes[1,1].set_title('Channel Distribution')

# save figure to outputs directory
plt.tight_layout()
plt.savefig('./outputs/data_visualization.png')

# 2. Detailed Species Analysis
print("\nSPECIES STATISTICS:")
print("="*50)
print(f"Total species: {species_counts.shape[0]}")


print("APPLYING PARALLEL FILTERS:")
parallel_filtered_df = df[df['rating'] > CONFIG['MIN_RATING_THRESHOLD']].copy()
parallel_filtered_df = parallel_filtered_df[(parallel_filtered_df['duration'] < CONFIG['MAX_DURATION'])]
valid_species = parallel_filtered_df['ebird_code'].value_counts()
valid_species = valid_species[valid_species >= CONFIG['MIN_SAMPLES_PER_CLASS']].index
parallel_filtered_df = parallel_filtered_df[parallel_filtered_df['ebird_code'].isin(valid_species)].copy()
print(f"Species with sufficient samples: {valid_species.shape[0]}")
print("="*50)


print("APPLYING SEQUENTIAL FILTERS INSTEAD:")

# Filter dataset to files with rating >= MIN_RATING_THRESHOLD
filtered_df = df[df['rating'] >= CONFIG['MIN_RATING_THRESHOLD']].copy()
species_counts = filtered_df['ebird_code'].value_counts()
print(f"Number of species after rating filter: {species_counts.shape[0]}")

# Filter by duration
filtered_df = filtered_df[filtered_df['duration'] <= CONFIG['MAX_DURATION']].copy()
species_counts = filtered_df ['ebird_code'].value_counts()
print(f"number of species  with duration < {CONFIG['MAX_DURATION']}: ", species_counts.shape[0])

# select the top 30 species by count
top_species = species_counts.head(CONFIG['NUM_CLASSES']).index.tolist()
filtered_df = filtered_df[filtered_df['ebird_code'].isin(top_species)].copy()
# For each of the top 30 species, keep only the best 100 samples by rating
filtered_df = (
    filtered_df.sort_values(['ebird_code', 'rating'], ascending=[True, False])
    .groupby('ebird_code')
    .head(100)
    .reset_index(drop=True)
)
updatedvalue_counts = filtered_df['ebird_code'].value_counts()
print("Top 30 species value counts sorted (after limiting to best 100 by rating): ", updatedvalue_counts)

filtered_df.to_csv('./dataset/filtered_df.csv', index=False)
