"""
models.py
---------
FitMetrics Intelligence - Predictive Modeling Module

Trains and evaluates three models:
  1. Random Forest Regression    -> Predict Calories Burned
  2. Random Forest Classification -> Predict Experience Level
  3. K-Means Clustering           -> Member Segmentation

All results and visualisations are saved to visuals/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score,
                              confusion_matrix, classification_report)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import run_pipeline

VISUALS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visuals')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PALETTE = ['#2C4A6E', '#4A7BA7', '#6BA3BE', '#A8C5DA', '#D4E6F1']
ACCENT = '#2C4A6E'
sns.set_theme(style='whitegrid', font='DejaVu Sans')

MODEL_FEATURES = [
    'Age', 'Gender_enc', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
    'Resting_BPM', 'Session_Duration (hours)', 'Fat_Percentage',
    'Water_Intake (liters)', 'Workout_Frequency (days/week)',
    'Experience_Level', 'BMI', 'WorkoutType_enc',
    'HR_Reserve', 'HR_Intensity_%', 'Calories_Per_Hour'
]

CLUSTER_FEATURES = [
    'Calories_Burned', 'Session_Duration (hours)', 'Workout_Frequency (days/week)',
    'Fat_Percentage', 'BMI', 'Water_Intake (liters)'
]


def train_regression(df):
    X = df[MODEL_FEATURES].copy()
    y = df['Calories_Burned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[models] Regression | R2={r2:.4f} | MAE={mae:.2f}")

    pickle.dump(model, open(os.path.join(MODELS_DIR, 'calorie_regressor.pkl'), 'wb'))
    return model, X_test, y_test, y_pred, r2, mae


def train_classifier(df):
    X = df[MODEL_FEATURES].copy()
    y = df['Experience_Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv = cross_val_score(model, X, y, cv=5)

    print(f"[models] Classifier | CV Accuracy={cv.mean():.4f} (+/- {cv.std():.4f})")
    print(classification_report(y_test, y_pred, target_names=['Beginner', 'Intermediate', 'Expert']))

    pickle.dump(model, open(os.path.join(MODELS_DIR, 'experience_classifier.pkl'), 'wb'))
    return model, y_test, y_pred, cv


def train_clustering(df):
    X = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
                for k in range(2, 9)]

    best = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = best.fit_predict(X_scaled)
    print(f"[models] Clustering | 3 segments created")
    return labels, inertias, X_scaled, X


def plot_model_results(reg_data, clf_data, cluster_data):
    model_r, X_test_r, y_test, y_pred_r, r2, mae = reg_data
    model_c, y_test_c, y_pred_c, cv = clf_data
    labels, inertias, X_scaled, X_cluster = cluster_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Predictive Modeling Results', fontsize=16, fontweight='bold', color=ACCENT)

    # Feature importance
    imp = pd.Series(model_r.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=True).tail(10)
    imp.plot(kind='barh', ax=axes[0, 0], color=PALETTE[1])
    axes[0, 0].set_title(f'Feature Importance: Calorie Prediction\nR2={r2:.3f} | MAE={mae:.1f}',
                          fontweight='bold', color=ACCENT)
    axes[0, 0].set_xlabel('Importance Score')

    # Actual vs Predicted
    axes[0, 1].scatter(y_test, y_pred_r, alpha=0.4, color=PALETTE[1], s=20, edgecolors='none')
    mn, mx = y_test.min(), y_test.max()
    axes[0, 1].plot([mn, mx], [mn, mx], color=PALETTE[0], linewidth=2, linestyle='--')
    axes[0, 1].set_title('Actual vs Predicted Calories', fontweight='bold', color=ACCENT)
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].text(0.05, 0.92, f'R2 = {r2:.3f}', transform=axes[0, 1].transAxes, color=ACCENT, fontsize=11)

    # Confusion matrix
    cm = confusion_matrix(y_test_c, y_pred_c)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Beginner', 'Intermediate', 'Expert'],
                yticklabels=['Beginner', 'Intermediate', 'Expert'])
    axes[1, 0].set_title('Experience Level Classification\nConfusion Matrix', fontweight='bold', color=ACCENT)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # CV scores
    axes[1, 1].bar(range(1, 6), cv, color=PALETTE[1], edgecolor='white')
    axes[1, 1].axhline(cv.mean(), color=PALETTE[0], linewidth=2, linestyle='--',
                        label=f'Mean Accuracy: {cv.mean():.3f}')
    axes[1, 1].set_title('5-Fold Cross-Validation: Classifier Accuracy', fontweight='bold', color=ACCENT)
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '05_model_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[models] Saved 05_model_results.png")


def plot_clustering(labels, inertias):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Member Segmentation via K-Means Clustering', fontsize=16, fontweight='bold', color=ACCENT)

    axes[0].plot(range(2, 9), inertias, marker='o', color=PALETTE[0], linewidth=2, markersize=8)
    axes[0].axvline(3, color=PALETTE[2], linestyle='--', linewidth=1.5, label='Optimal k=3')
    axes[0].set_title('Elbow Method', fontweight='bold', color=ACCENT)
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].legend()

    colors = [PALETTE[0], PALETTE[1], PALETTE[3]]
    for c in range(3):
        mask = labels == c
    axes[1].set_title('Segment Distribution', fontweight='bold', color=ACCENT)

    unique, counts = np.unique(labels, return_counts=True)
    bars = axes[1].bar([f'Segment {u+1}' for u in unique], counts, color=colors[:len(unique)], edgecolor='white')
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha='center', fontweight='bold', color=ACCENT)
    axes[1].set_ylabel('Number of Members')

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, '04_clustering.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[models] Saved 04_clustering.png")


def run_models():
    df = run_pipeline()
    reg_data = train_regression(df)
    clf_data = train_classifier(df)
    labels, inertias, X_scaled, X_cluster = train_clustering(df)
    plot_model_results(reg_data, clf_data, (labels, inertias, X_scaled, X_cluster))
    plot_clustering(labels, inertias)
    print("[models] All models trained and saved.")


if __name__ == '__main__':
    run_models()
