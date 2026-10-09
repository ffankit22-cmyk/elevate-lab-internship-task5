#!/usr/bin/env python3
"""
Decision Trees and Random Forests - AI & ML Internship Task 5
Author: Ashish Kushwaha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the Heart Disease Dataset (Cleveland)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=column_names, na_values='?')

# Convert target to binary: 0 -> no disease, 1-4 -> disease
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Drop rows with missing values
df = df.dropna()

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Convert features to numeric (some columns might be read as object due to missing values)
X = X.apply(pd.to_numeric, errors='coerce')
# Drop any rows that became NaN after conversion (shouldn't happen if we dropped missing values earlier)
X = X.dropna()
y = y[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Decision Tree ==========
print("=== Decision Tree ===")
# Train with default parameters
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

# Analyze overfitting by controlling tree depth
print("\n--- Overfitting Analysis ---")
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_accs = []
test_accs = []
for depth in depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_accs.append(dt_temp.score(X_train, y_train))
    test_accs.append(dt_temp.score(X_test, y_test))
    print(f"Depth {depth}: Train Accuracy = {train_accs[-1]:.4f}, Test Accuracy = {test_accs[-1]:.4f}")

# Find the best depth (highest test accuracy)
best_depth = depths[np.argmax(test_accs)]
print(f"\nBest depth based on test accuracy: {best_depth}")

# Train the best decision tree
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_best.fit(X_train, y_train)
y_pred_dt_best = dt_best.predict(X_test)
accuracy_dt_best = accuracy_score(y_test, y_pred_dt_best)
print(f"Best Decision Tree Accuracy: {accuracy_dt_best:.4f}")

# Visualize the best decision tree (if not too deep)
plt.figure(figsize=(20,10))
plot_tree(dt_best, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, rounded=True, fontsize=10)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.savefig("decision_tree.png", bbox_inches='tight', dpi=300)
plt.close()

# ========== Random Forest ==========
print("\n=== Random Forest ===")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Feature importances
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Importances (Random Forest):")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# Plot feature importances
plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importances.png", bbox_inches='tight', dpi=300)
plt.close()

# ========== Cross-Validation ==========
print("\n=== Cross-Validation ===")
dt_scores = cross_val_score(dt_best, X, y, cv=5)
rf_scores = cross_val_score(rf_clf, X, y, cv=5)
print(f"Decision Tree CV Accuracy: {dt_scores.mean():.4f} (+/- {dt_scores.std() * 2:.4f})")
print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")

# ========== Save Results ==========
results = {
    'Decision Tree Accuracy': accuracy_dt,
    'Best Decision Tree Accuracy': accuracy_dt_best,
    'Random Forest Accuracy': accuracy_rf,
    'Decision Tree CV Mean': dt_scores.mean(),
    'Random Forest CV Mean': rf_scores.mean()
}
results_df = pd.DataFrame([results])
results_df.to_csv('results.csv', index=False)

print("\n=== Task Completed ===")
print("Files generated:")
print("- decision_tree.png: Visualization of the best decision tree")
print("- feature_importances.png: Feature importances from Random Forest")
print("- results.csv: Summary of accuracies")
