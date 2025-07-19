# Step 2: Data Visualization and Configuration Setup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from configurations import CONFIG

print("=== STEP 2: DATA VISUALIZATION & CONFIGURATION ===\n")

# Load the metadata CSV
print("Loading train_extended.csv...")
df = pd.read_csv('./dataset/train_extended.csv')

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
axes[0,0].hist(df['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(CONFIG['MIN_RATING_THRESHOLD'], color='red', linestyle='--',
                 label=f'Min Rating Threshold: {CONFIG["MIN_RATING_THRESHOLD"]}')
axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Rating Distribution')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Duration distribution
axes[0,1].hist(df['duration'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
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
species_counts = df['ebird_code'].value_counts().head(153) #crystal

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

#plt.tight_layout()
#plt.show()

# 2. Detailed Species Analysis
print(f"\nSPECIES STATISTICS:")
print("="*50)
print(f"Total species: {species_counts.shape[0]}")

# Filter the original DataFrame to keep only those species
# Get the top 153 ebird_codes
# Step 1: Get value counts (i.e., frequency) of unique values
value_counts = df['ebird_code'].value_counts()

print("value counts: ", value_counts)

# Step 2: Sort the index (unique values) alphabetically
value_counts_sorted = value_counts.sort_index()

print("value counts sorted: ", value_counts_sorted)

top_153_unique = value_counts_sorted[:153]

print("value counts sorted am: ", top_153_unique)


# Step 2: Filter to keep only ebird_codes with at least 100 occurrences
valid_codes = value_counts[value_counts > CONFIG['MIN_SAMPLES_PER_CLASS']].index

# Step 5: Get all rows from the original df with those top ebird_codes
filtered_df = df[df['ebird_code'].isin(top_153_unique.index)]


print("filter data with top 153: ", top_153_unique)
print(filtered_df)


print("filter data with atleast 100 samples in 153: ")
# Step 5: Get all rows from the original df with those top ebird_codes
filtered_df = filtered_df[filtered_df['ebird_code'].isin(valid_codes)]
species_counts = filtered_df ['ebird_code'].value_counts()
print(f"number of species  with 100 samples: ", species_counts.shape[0])
print(filtered_df)

updatedvalue_counts = filtered_df['ebird_code'].value_counts()
print("updated species value counts sorted a to m: ", updatedvalue_counts)



# 3. Apply Filters and Show Impact
print(f"\n{'='*50}")
print("APPLYING FILTERS:")
print("="*50)

# Filter by duration
filtered_df = filtered_df[filtered_df['duration'] < 20]
species_counts = filtered_df ['ebird_code'].value_counts()
print(f"number of species  with duration < 20: ", species_counts.shape[0])
print(filtered_df)

# Filter by rating
filtered_df = filtered_df[filtered_df['rating'] > 3]
species_counts = filtered_df['ebird_code'].value_counts()
print(f"number of species  with rating > 3: ", species_counts.shape[0])
print(filtered_df)

# Remove rows where 'url' is missing or empty
filtered_df = filtered_df[filtered_df['url'].notna() & (filtered_df['url'].str.strip() != '')]

print(f"Remaining rows after removing missing/empty URLs: {len(filtered_df)}")
print(f"number of species  that contain url data ", species_counts.shape[0])
print(filtered_df)

print(f"number of species COUNTS ", species_counts)

# # 4. Select Top N Classes
# print(f"\n{'='*50}")
# print(f"SELECTING TOP {CONFIG['NUM_CLASSES']} CLASSES:")
# print("="*50)

# Get top N species by sample count (after filtering)
#final_species_counts = df_filtered['ebird_code'].value_counts().head(153) #crystal
# if CONFIG['NUM_CLASSES'] <= len(final_species_counts):
#     selected_species = final_species_counts.head(CONFIG['NUM_CLASSES']).index.tolist()
#     df_final = df_filtered[df_filtered['ebird_code'].isin(selected_species)].copy()

#     print(f"Selected {len(selected_species)} species:")
#     for i, species in enumerate(selected_species, 1):
#         count = final_species_counts[species]
#         print(f"{i:2d}. {species}: {count:3d} samples")

#     print(f"\nFinal dataset: {len(df_final):,} samples across {len(selected_species)} species")

#     # Class balance visualization
#     plt.figure(figsize=(12, 6))
#     final_counts = df_final['ebird_code'].value_counts()
#     plt.bar(range(len(final_counts)), final_counts.values, color='steelblue', alpha=0.7)
#     plt.xlabel('Species (ordered by sample count)')
#     plt.ylabel('Number of Samples')
#     plt.title(f'Class Distribution - Top {CONFIG["NUM_CLASSES"]} Species (After Filtering)')
#     plt.xticks(range(len(final_counts)), final_counts.index, rotation=45, ha='right')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

#     # Calculate class imbalance ratio
#     max_samples = final_counts.max()
#     min_samples = final_counts.min()
#     imbalance_ratio = max_samples / min_samples
#     print(f"\nClass imbalance ratio: {imbalance_ratio:.2f} (max: {max_samples}, min: {min_samples})")

# else:
#     print(f" Not enough species meet the criteria! Only {len(final_species_counts)} species available.")
#     print("Consider reducing MIN_SAMPLES_PER_CLASS or NUM_CLASSES")

# # 5. Save configuration and filtered dataset info
# print(f"\n{'='*50}")
# print("SUMMARY FOR NEXT STEPS:")
# print("="*50)
# print(f" Configuration set for {CONFIG['NUM_CLASSES']} classes")
# print(f" {len(df_final):,} total samples after filtering")
# print(f" Rating threshold: ≥{CONFIG['MIN_RATING_THRESHOLD']}")
# print(f" Duration range: {CONFIG['MIN_DURATION']}-{CONFIG['MAX_DURATION']} seconds")
# print(f" Ready for audio file validation and preprocessing")

# # Create a summary for the next step
# STEP2_SUMMARY = {
#     'df_final': df_final,
#     'selected_species': selected_species,
#     'config': CONFIG,
#     'class_counts': final_counts.to_dict()
# }

print(f"\n Next step: Audio file validation and path checking")
# Save the final filtered DataFrame to CSV
filtered_df.to_csv('./dataset/filtered_df.csv', index=False)