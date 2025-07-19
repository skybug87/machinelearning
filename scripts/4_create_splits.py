import pandas as pd
from sklearn.model_selection import train_test_split
from configurations import CONFIG

# Load filtered DataFrame
filtered_df = pd.read_csv('dataset/filtered_df.csv', index_col=0)

# Features and labels
X = filtered_df[['title', 'filename']]
y = filtered_df[['ebird_code']]

# First split: train (TRAIN_SPLIT), temp (1-TRAIN_SPLIT)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=1 - CONFIG['TRAIN_SPLIT'],
    stratify=y,
    random_state=CONFIG['RANDOM_SEED']
)

# Compute validation and test proportions relative to temp
val_prop = CONFIG['VAL_SPLIT'] / (CONFIG['VAL_SPLIT'] + CONFIG['TEST_SPLIT'])
test_prop = CONFIG['TEST_SPLIT'] / (CONFIG['VAL_SPLIT'] + CONFIG['TEST_SPLIT'])

# Second split: validation (VAL_SPLIT), test (TEST_SPLIT) from temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=test_prop,
    stratify=y_temp,
    random_state=CONFIG['RANDOM_SEED']
)

# Save splits with index for alignment
X_train.to_csv('dataset/X_train.csv')
X_val.to_csv('dataset/X_val.csv')
X_test.to_csv('dataset/X_test.csv')
y_train.to_csv('dataset/y_train.csv')
y_val.to_csv('dataset/y_val.csv')
y_test.to_csv('dataset/y_test.csv')