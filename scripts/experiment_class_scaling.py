"""
Experiment: Class Scaling for Bird Call Classification (CNN, Deep Learning Pipeline)
-------------------------------------------------------------------------------
Runs the deep learning pipeline for N in [3, 5, 30] most frequent classes.
- Loads splits, filters to top-N classes, remaps labels, loads spectrograms,
  trains CNN, evaluates, and plots accuracy/F1 vs N.
- Saves plot to outputs/class_scaling_metrics.png and prints summary table.

Instructions:
- Do not modify the original pipeline script.
- This script must be run from the project root.

Author: Kilo Code
Date: 2025-07-20
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
pipeline = importlib.import_module("scripts.7_deep_learning_pipeline")

from collections import Counter

def filter_top_n_classes(df, top_classes):
    """Filter DataFrame to only include rows with ebird_code in top_classes."""
    return df[df['ebird_code'].isin(top_classes)].reset_index(drop=True)

def remap_labels(df, class_order):
    """Remap ebird_code labels to 0..N-1 based on class_order."""
    label2idx = {label: idx for idx, label in enumerate(class_order)}
    df = df.copy()
    df['ebird_code'] = df['ebird_code'].map(label2idx)
    return df, label2idx

def run_experiment(N, splits, spectrogram_col='spectrogram_path_original', batch_size=32, epochs=40):
    # 1. Identify N most frequent classes in train
    train_counts = Counter(splits['train']['ebird_code'])
    top_classes = [c for c, _ in train_counts.most_common(N)]

    # 2. Filter all splits to only these classes
    filtered = {split: filter_top_n_classes(df, top_classes) for split, df in splits.items()}

    # 3. Remap class labels to 0..N-1
    filtered_remap = {}
    for split, df in filtered.items():
        filtered_remap[split], label2idx = remap_labels(df, top_classes)

    # 4. Load spectrogram arrays
    X = {}
    y = {}
    for split in ['train', 'val', 'test']:
        X[split] = pipeline.load_spectrogram_arrays(filtered_remap[split], spectrogram_col)
        y[split] = filtered_remap[split]['ebird_code'].values
        if X[split].ndim == 3:
            X[split] = np.expand_dims(X[split], -1)

    # 5. Build tf.data.Dataset
    train_ds = pipeline.get_tf_data(X['train'], y['train'], batch_size=batch_size)
    val_ds = pipeline.get_tf_data(X['val'], y['val'], batch_size=batch_size, shuffle=False)
    test_ds = pipeline.get_tf_data(X['test'], y['test'], batch_size=batch_size, shuffle=False)

    # 6. Build and train model
    input_shape = X['train'].shape[1:]
    num_classes = N
    model = pipeline.create_cnn_model(input_shape, num_classes)
    history = pipeline.train_cnn_model(model, train_ds, val_ds, epochs=epochs)

    # 7. Evaluate
    metrics = pipeline.evaluate_model(model, test_ds)
    return metrics

def main():
    dataset_dir = "./dataset"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load splits
    splits = pipeline.load_split_csvs(dataset_dir)

    results = []
    for N in [3, 5, 30]:
        print(f"\n=== Running experiment for N={N} classes ===")
        metrics = run_experiment(N, splits)
        results.append({'N': N, 'accuracy': metrics['accuracy'], 'f1': metrics['f1']})
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    # Plot results
    df = pd.DataFrame(results)
    plt.figure(figsize=(8,5))
    plt.plot(df['N'], df['accuracy'], marker='o', label='Accuracy')
    plt.plot(df['N'], df['f1'], marker='s', label='F1 Score')
    plt.xlabel('Number of Classes (N)')
    plt.ylabel('Score')
    plt.title('CNN Performance vs Number of Classes')
    plt.xticks(df['N'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "class_scaling_metrics.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

    # Print summary table
    print("\nSummary Table:")
    print(df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()