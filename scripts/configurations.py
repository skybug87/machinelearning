# Authors: Sepehr Goshayeshi, Duncan Hord, Crystal Matheny, Yehong Huang

CONFIG = {
    'MIN_RATING_THRESHOLD': 3.0,      # Filter recordings below this rating #crystal
    'NUM_CLASSES': 30,                # Number of bird species to use (3, 5, 30, or custom) this value should be calculated #crystal
    'MAX_DURATION': 300,              # Maximum duration in seconds (20 seconds) #crystal
    'MIN_DURATION': 1,                # Minimum duration in seconds, if its 0 then theres nothing #crystal
    'MIN_SAMPLES_PER_CLASS': 100,      # Minimum samples needed per species #crystal
    'TRAIN_SPLIT': 0.6,                # Proportion of data for training
    'VAL_SPLIT': 0.2,                  # Proportion for validation
    'TEST_SPLIT': 0.2,                 # Proportion for test
    'RANDOM_SEED': 42,                 # Random seed for reproducibility
    'MAX_SEGMENTS': 1,                # Maximum number of segments per audio file
}
