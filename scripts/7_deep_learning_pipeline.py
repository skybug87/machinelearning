# Authors: Sepehr Goshayeshi, Duncan Hord, Crystal Matheny, Yehong Huang

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
import logging
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
#  Data Loading Utilities
# =========================

def load_split_csvs(dataset_dir):
    """
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' mapping to DataFrames.
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        X = pd.read_csv(os.path.join(dataset_dir, f'X_{split}.csv'), index_col=0)
        y = pd.read_csv(os.path.join(dataset_dir, f'y_{split}.csv'), index_col=0)
        df = X.copy()
        df['ebird_code'] = y.values.ravel()
        splits[split] = df
    return splits

def load_spectrogram_arrays(df, spectrogram_col='yamnet_spectrogram_path_reduced'):
    """
    Returns:
        np.ndarray: Array of loaded spectrograms.
    """
    specs = []
    for path in df[spectrogram_col]:
        arr = np.load(path)
        specs.append(arr)
    return np.stack(specs)

def create_label_mapping(labels):
    """
    Returns:
        tuple: (np.ndarray of integer labels, dict mapping ebird_code to int)
    """
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label2idx[label] for label in labels])
    return int_labels, label2idx

def get_tf_data(X, y, batch_size=32, shuffle=True):
    """
    Returns:
        tf.data.Dataset: Dataset yielding (X, y) batches.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def create_cnn_model(input_shape, num_classes):
    model = Sequential([

        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((4, 4)),
        
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_yamnet_classifier(num_classes, embedding_size=1024):
    """
    Returns:
        tf.keras.Model: Classifier model for YAMNet embeddings (uncompiled)
    """
    model = Sequential([
        layers.Input(shape=(embedding_size,)),
        layers.Dense(512, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.4, name='dropout_1'),

        layers.Dense(256, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.4, name='dropout_2'),

        layers.Dense(128, activation='relu', name='dense_3'),
        layers.Dropout(0.3, name='dropout_3'),

        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='YAMNet_Bird_Classifier')
    return model
# =========================
# 3. Data Preparation Utilities
# =========================

def train_cnn_model(model, train_ds, val_ds, epochs=50, patience=15, checkpoint_path=None):
    """
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, verbose=1)
    ]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1))
    logging.info("Starting CNN training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    logging.info("CNN training complete.")
    return history

def train_yamnet_model(model, train_ds, val_ds, epochs=50, patience=15, checkpoint_path=None):
    """
    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, verbose=1)
    ]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1))
    logging.info("Starting YAMNet classifier training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    logging.info("YAMNet classifier training complete.")
    return history

def evaluate_model(model, dataset, label_map=None, average='macro'):
    """
    Returns:
        dict: Dictionary with 'accuracy' and 'f1' scores.
    """
    y_true = []
    y_pred = []
    for X_batch, y_batch in dataset:
        preds = model.predict(X_batch, verbose=0)
        preds = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    logging.info(f"Evaluation - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return {'accuracy': acc, 'f1': f1}
    
def save_training_history(history, filepath, as_json=False):
    hist_dict = history.history
    if as_json or filepath.lower().endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(hist_dict, f, indent=2)
    else:
        pd.DataFrame(hist_dict).to_csv(filepath, index=False)

def save_metrics(metrics_dict, filepath, as_json=False):
    if as_json or filepath.lower().endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    else:
        pd.DataFrame([metrics_dict]).to_csv(filepath, index=False)

def save_class_mapping(label2idx, filepath):
    pd.DataFrame(list(label2idx.items()), columns=['label', 'index']).to_csv(filepath, index=False)

# =========================
# Plotting Utilities
# =========================

def plot_and_save_loss(history, filepath, title_info=""):
    hist = history.history
    plt.figure(figsize=(8, 6))
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if title_info:
        plt.title(f'Training and Validation Loss - {title_info}')
    else:
        plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, filepath, accuracy=None, title_info=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    if accuracy is not None:
        if title_info:
            plt.title(f'Confusion Matrix - {title_info} (Accuracy: {accuracy:.4f})')
        else:
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    else:
        if title_info:
            plt.title(f'Confusion Matrix - {title_info}')
        else:
            plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# =========================
# 7. Run Script
# =========================

if __name__ == "__main__":
    dataset_dir = "./dataset"
    output_dir = "./outputs"
    models_dir = "./models"
    epochs = 100

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ==================================================
    # CNN Embedding Classifier Training and Evaluation
    # ==================================================
    print("CNN Classifier Training and Evaluation:")
    # Load splits
    splits = load_split_csvs(dataset_dir)
    train_df = splits['train']
    val_df = splits['val']
    test_df = splits['test']

    # Load spectrogram arrays using the spectrogram path column present in the split CSVs
    X_train = load_spectrogram_arrays(train_df)
    X_val = load_spectrogram_arrays(val_df)
    X_test = load_spectrogram_arrays(test_df)

    # Prepare labels
    y_train, label2idx = create_label_mapping(train_df['ebird_code'])
    y_val, _ = create_label_mapping(val_df['ebird_code'])
    y_test, _ = create_label_mapping(test_df['ebird_code'])
    
    # Expand dims if needed (ensure channel last)
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, -1)
        X_val = np.expand_dims(X_val, -1)
        X_test = np.expand_dims(X_test, -1)
    
    # =========================
    # Normalize spectrograms (zero mean, unit variance using train stats)
    # =========================
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Build datasets (default batch size)
    train_ds = get_tf_data(X_train, y_train)
    val_ds = get_tf_data(X_val, y_val, shuffle=False)
    test_ds = get_tf_data(X_test, y_test, shuffle=False)

    # Model
    input_shape = X_train.shape[1:]
    num_classes = len(label2idx)
    model = create_cnn_model(input_shape, num_classes)

    # Train
    history = train_cnn_model(model, train_ds, val_ds, epochs=epochs)

    # Evaluate
    metrics = evaluate_model(model, test_ds)

    # Save model, history, metrics, and label mapping
    model.save(os.path.join(models_dir, "cnn_model.keras"))
    save_training_history(history, os.path.join(output_dir, "cnn_history.csv"))
    save_metrics(metrics, os.path.join(output_dir, "cnn_metrics.csv"))
    save_class_mapping(label2idx, os.path.join(output_dir, "cnn_class_mapping.csv"))

    # Plot and save loss curves
    plot_and_save_loss(history, os.path.join(output_dir, "cnn_loss_curve.png"), title_info="CNN")

    # Confusion matrix and accuracy for test set
    y_true = []
    y_pred = []
    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch, verbose=0)
        preds = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds)
    class_names = list(label2idx.keys())
    total_accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    plot_and_save_confusion_matrix(
        y_true, y_pred, class_names, os.path.join(output_dir, "cnn_confusion_matrix.png"), accuracy=total_accuracy, title_info="CNN"
    )
    print(f"Test set total accuracy: {total_accuracy:.4f}")

    print("Training complete. Model saved to ./models and artifacts saved to ./outputs.")

    # ==================================================
    # YAMNet Embedding Classifier Training and Evaluation
    # ==================================================
    print("YAMNet Classifier Training and Evaluation:")

    # Load YAMNet embeddings (expects columns: 'yamnet_embedding_path')
    def load_yamnet_embeddings(df, embedding_col='yamnet_embedding_path_reduced'):
        yamnet_embeds = []
        for path in df[embedding_col]:
            arr = np.load(path)
            yamnet_embeds.append(arr)
        return np.stack(yamnet_embeds)

    # Check if yamnet_embedding_path exists in DataFrame
    X_train_yam = load_yamnet_embeddings(train_df)
    X_val_yam = load_yamnet_embeddings(val_df)
    X_test_yam = load_yamnet_embeddings(test_df)

    # Normalize embeddings (optional, but often beneficial)
    mean_yam = X_train_yam.mean()
    std_yam = X_train_yam.std()
    X_train_yam = (X_train_yam - mean_yam) / (std_yam + 1e-8)
    X_val_yam = (X_val_yam - mean_yam) / (std_yam + 1e-8)
    X_test_yam = (X_test_yam - mean_yam) / (std_yam + 1e-8)

    train_ds_yam = get_tf_data(X_train_yam, y_train)
    val_ds_yam = get_tf_data(X_val_yam, y_val, shuffle=False)
    test_ds_yam = get_tf_data(X_test_yam, y_test, shuffle=False)

    yamnet_model = create_yamnet_classifier(num_classes, embedding_size=X_train_yam.shape[1])
    history_yam = train_yamnet_model(yamnet_model, train_ds_yam, val_ds_yam, epochs=100)

    metrics_yam = evaluate_model(yamnet_model, test_ds_yam)

    # Save YAMNet model, history, metrics, and label mapping
    yamnet_model.save(os.path.join(models_dir, "yamnet_model.keras"))
    save_training_history(history_yam, os.path.join(output_dir, "yamnet_history.csv"))
    save_metrics(metrics_yam, os.path.join(output_dir, "yamnet_metrics.csv"))
    save_class_mapping(label2idx, os.path.join(output_dir, "yamnet_class_mapping.csv"))

    # Plot and save loss curves
    plot_and_save_loss(history_yam, os.path.join(output_dir, "yamnet_loss_curve.png"), title_info="YAMNet")

    # Confusion matrix and accuracy for test set
    y_true_yam = []
    y_pred_yam = []
    for X_batch, y_batch in test_ds_yam:
        preds = yamnet_model.predict(X_batch, verbose=0)
        preds = np.argmax(preds, axis=1)
        y_true_yam.extend(y_batch.numpy())
        y_pred_yam.extend(preds)
    class_names_yam = list(label2idx.keys())
    total_accuracy_yam = np.mean(np.array(y_true_yam) == np.array(y_pred_yam))
    plot_and_save_confusion_matrix(
        y_true_yam, y_pred_yam, class_names_yam, os.path.join(output_dir, "yamnet_confusion_matrix.png"), accuracy=total_accuracy_yam, title_info="YAMNet"
    )
    print(f"YAMNet test set total accuracy: {total_accuracy_yam:.4f}")
    print("YAMNet training complete. Model and artifacts saved.")

