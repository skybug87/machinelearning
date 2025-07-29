# Authors: Sepehr Goshayeshi, Duncan Hord, Crystal Matheny, Yehong Huang

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load normalized numeric features and splits
print("Loading numeric feature splits...")

X_train = pd.read_csv("dataset/X_train.csv", index_col=0)
X_val   = pd.read_csv("dataset/X_val.csv", index_col=0)
X_test  = pd.read_csv("dataset/X_test.csv", index_col=0)
y_train = pd.read_csv("dataset/y_train.csv", index_col=0).squeeze()
y_val   = pd.read_csv("dataset/y_val.csv", index_col=0).squeeze()
y_test  = pd.read_csv("dataset/y_test.csv", index_col=0).squeeze()

# Drop excluded columns (from scripts/5_normalize_features.py)
exclude_cols = [
    'source',
    'filename_original',
    'filename_reduced',
    'ebird_code',
    'spectrogram_path_original',
    'spectrogram_path_reduced',
    'yamnet_embedding_path_original',
    'yamnet_embedding_path_reduced',
    'yamnet_spectrogram_path_original',
    'yamnet_spectrogram_path_reduced',
]
X_train = X_train.drop(columns=[col for col in exclude_cols if col in X_train.columns])
X_val   = X_val.drop(columns=[col for col in exclude_cols if col in X_val.columns])
X_test  = X_test.drop(columns=[col for col in exclude_cols if col in X_test.columns])

print("Shapes after dropping excluded columns:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Use only numeric features (already normalized)
X_tr, y_tr = X_train.values, y_train.values
X_va, y_va = X_val.values, y_val.values
X_te, y_te = X_test.values, y_test.values

# Collect accuracy scores for each model
model_accuracies = {}

# Random Forest
RF_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])
RF_pipe.fit(X_tr, y_tr)
pred = RF_pipe.predict(X_te)
rf_acc = accuracy_score(y_te, pred)
model_accuracies['Random Forest'] = rf_acc
print("RF accuracy:", rf_acc)
print(classification_report(y_te, pred))

# SGD Classifier
sgd_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('sgd', SGDClassifier(
        loss='log_loss',
        learning_rate='optimal',
        max_iter=1000, tol=1e-3,
        random_state=42))
])
sgd_pipe.fit(X_tr, y_tr)
pred_sgd = sgd_pipe.predict(X_te)
sgd_acc = accuracy_score(y_te, pred_sgd)
model_accuracies['SGD'] = sgd_acc
print("SGD Accuracy:", sgd_acc)
print(classification_report(y_te, pred_sgd))

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_tr, y_tr)
pred_nb = nb.predict(X_te)
nb_acc = accuracy_score(y_te, pred_nb)
model_accuracies['Naive Bayes'] = nb_acc
print("NB Accuracy:", nb_acc)
print(classification_report(y_te, pred_nb))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr, y_tr)
pred_dt = dt.predict(X_te)
dt_acc = accuracy_score(y_te, pred_dt)
model_accuracies['Decision Tree'] = dt_acc
print("Decision Tree Accuracy:", dt_acc)
print(classification_report(y_te, pred_dt))

# KNN
knn_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_pipe.fit(X_tr, y_tr)
pred_knn = knn_pipe.predict(X_te)
knn_acc = accuracy_score(y_te, pred_knn)
model_accuracies['KNN'] = knn_acc
print("KNN Accuracy:", knn_acc)
print(classification_report(y_te, pred_knn))

# SVM
svm_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
])
svm_pipe.fit(X_tr, y_tr)
pred_svm = svm_pipe.predict(X_te)
svm_acc = accuracy_score(y_te, pred_svm)
model_accuracies['SVM'] = svm_acc
print("SVM Accuracy:", svm_acc)
print(classification_report(y_te, pred_svm))

# Plot and save model accuracies
plt.figure(figsize=(8, 5))
bars = plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.title('Traditional Model Accuracies')
plt.ylim(0, 1)

# Add accuracy percentage labels on each bar
for bar, acc in zip(bars, model_accuracies.values()):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{acc * 100:.1f}%",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig('./outputs/traditional_model_accuracies.png')
plt.close()
print("Saved accuracy comparison plot to ./outputs/traditional_model_accuracies.png")

################################################################
# Subset experiments for first 3, 5, and 30 unique labels
################################################################

def run_subset_experiment(label_subset, subset_name):
    # Filter splits to only include samples with labels in label_subset
    mask_train = y_train.isin(label_subset)
    mask_val = y_val.isin(label_subset)
    mask_test = y_test.isin(label_subset)

    X_tr_sub = X_train[mask_train].values
    y_tr_sub = y_train[mask_train].values
    # X_va_sub = X_val[mask_val].values # not used
    # y_va_sub = y_val[mask_val].values # not used
    X_te_sub = X_test[mask_test].values
    y_te_sub = y_test[mask_test].values

    print(f"\n--- Subset: {subset_name} ({len(label_subset)} classes) ---")
    print("Train shape:", X_tr_sub.shape, "Test shape:", X_te_sub.shape)

    subset_accuracies = {}

    # Random Forest
    RF_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    RF_pipe.fit(X_tr_sub, y_tr_sub)
    pred = RF_pipe.predict(X_te_sub)
    rf_acc = accuracy_score(y_te_sub, pred)
    subset_accuracies['Random Forest'] = rf_acc
    print("RF accuracy:", rf_acc)
    print(classification_report(y_te_sub, pred))

    # SGD Classifier
    sgd_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('sgd', SGDClassifier(
            loss='log_loss',
            learning_rate='optimal',
            max_iter=1000, tol=1e-3,
            random_state=42))
    ])
    sgd_pipe.fit(X_tr_sub, y_tr_sub)
    pred_sgd = sgd_pipe.predict(X_te_sub)
    sgd_acc = accuracy_score(y_te_sub, pred_sgd)
    subset_accuracies['SGD'] = sgd_acc
    print("SGD Accuracy:", sgd_acc)
    print(classification_report(y_te_sub, pred_sgd))

    # Gaussian Naive Bayes
    nb = GaussianNB()
    nb.fit(X_tr_sub, y_tr_sub)
    pred_nb = nb.predict(X_te_sub)
    nb_acc = accuracy_score(y_te_sub, pred_nb)
    subset_accuracies['Naive Bayes'] = nb_acc
    print("NB Accuracy:", nb_acc)
    print(classification_report(y_te_sub, pred_nb))

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_tr_sub, y_tr_sub)
    pred_dt = dt.predict(X_te_sub)
    dt_acc = accuracy_score(y_te_sub, pred_dt)
    subset_accuracies['Decision Tree'] = dt_acc
    print("Decision Tree Accuracy:", dt_acc)
    print(classification_report(y_te_sub, pred_dt))

    # KNN
    knn_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipe.fit(X_tr_sub, y_tr_sub)
    pred_knn = knn_pipe.predict(X_te_sub)
    knn_acc = accuracy_score(y_te_sub, pred_knn)
    subset_accuracies['KNN'] = knn_acc
    print("KNN Accuracy:", knn_acc)
    print(classification_report(y_te_sub, pred_knn))

    # SVM
    svm_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, random_state=42))
    ])
    svm_pipe.fit(X_tr_sub, y_tr_sub)
    pred_svm = svm_pipe.predict(X_te_sub)
    svm_acc = accuracy_score(y_te_sub, pred_svm)
    subset_accuracies['SVM'] = svm_acc
    print("SVM Accuracy:", svm_acc)
    print(classification_report(y_te_sub, pred_svm))

    # Return the accuracy dictionary instead of plotting
    return subset_accuracies

# Run for first 3, 15, and 30 unique y_train values and collect results
first3_labels = y_train.unique()[:3]
first5_labels = y_train.unique()[:5]
first30_labels = y_train.unique()[:30]

acc_3 = run_subset_experiment(first3_labels, "3-bird")
acc_5 = run_subset_experiment(first5_labels, "5-bird")
acc_30 = run_subset_experiment(first30_labels, "30-bird")

# Combine results into a DataFrame for plotting
all_models = sorted(set(acc_3) | set(acc_5) | set(acc_30))
subset_names = ['3', '5', '30']
acc_dict = {'3': acc_3, '5': acc_5, '30': acc_30}

data = []
for model in all_models:
    row = [acc_dict[s].get(model, 0) for s in subset_names]
    data.append(row)

df = pd.DataFrame(data, index=all_models, columns=subset_names)

bar_width = 0.22
ordered_models = ['SGD', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes'] # Enforce consistent model order to match image from paper
x = np.arange(len(ordered_models))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
subset_labels = ['3 birds', '5 birds', '30 birds']

plt.figure(figsize=(12, 6))
for idx, subset in enumerate(subset_names):
    accs = [acc_dict[subset].get(model, 0) for model in ordered_models]
    plt.bar(x + idx * bar_width, accs, width=bar_width, label=subset_labels[idx], color=colors[idx])
    for i, acc in enumerate(accs):
        plt.text(x[i] + idx * bar_width, acc + 0.01, f"{acc*100:.1f}%", ha='center', va='bottom', fontsize=9)

plt.xticks(x + bar_width, ordered_models, rotation=30)
plt.ylabel('Accuracy')
plt.title('Traditional Model Accuracies by Subset Size')
plt.ylim(0, 1)
plt.legend(title='Subset')
plt.tight_layout()
plt.savefig('./outputs/traditional_model_accuracies_combined.png')
plt.close()
print("Saved combined accuracy comparison plot to ./outputs/traditional_model_accuracies_combined.png")
