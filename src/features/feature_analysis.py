"""
Feature Analysis and Visualization
====================================

Analyzes extracted features to understand their discriminative power
for fault classification. Produces publication-quality plots for:
- Feature importance ranking
- Correlation analysis
- Class distribution visualization
- Feature selection

Usage:
    python -m src.features.feature_analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import json
import os


def analyze_features(features_dir: str = "data/features",
                     output_dir: str = "results/feature_analysis"):
    """
    Comprehensive feature analysis with visualizations.

    Args:
        features_dir: Path to extracted features
        output_dir:   Path to save analysis results and plots
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Feature Analysis and Visualization")
    print("=" * 70)

    # Load feature data
    feature_csv = os.path.join(features_dir, "features.csv")
    if not os.path.exists(feature_csv):
        print("  ERROR: features.csv not found. Run feature extraction first.")
        return

    df = pd.read_csv(feature_csv)

    # Load feature names
    with open(os.path.join(features_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    # Filter to training data only for analysis
    train_df = df[df['split'] == 'train'].copy()

    if len(train_df) == 0:
        print("  WARNING: No training data found, using all data")
        train_df = df.copy()

    X = train_df[feature_names].values
    y = train_df['fault_type'].values

    fault_names = {
        0: "Normal", 1: "Bearing", 2: "Imbalance",
        3: "Misalignment", 4: "Electrical"
    }

    print(f"\n  Dataset: {len(train_df)} samples, {len(feature_names)} features")
    print(f"  Classes: {len(np.unique(y))}")

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ─── 1. Feature Importance (Random Forest) ───────────────────────
    print("\n  [1/4] Computing feature importance (Random Forest)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_

    # Top 20 features
    top_k = 20
    top_idx = np.argsort(importances)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    bars = ax.barh(range(top_k), importances[top_idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in top_idx][::-1], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Top 5 features:")
    for i in range(min(5, top_k)):
        print(f"      {i+1}. {feature_names[top_idx[i]]:35s} "
              f"(importance: {importances[top_idx[i]]:.4f})")

    # ─── 2. Mutual Information ───────────────────────────────────────
    print("\n  [2/4] Computing mutual information...")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 8))
    mi_top_idx = np.argsort(mi_scores)[::-1][:top_k]
    colors_mi = plt.cm.plasma(np.linspace(0.3, 0.9, top_k))
    ax.barh(range(top_k), mi_scores[mi_top_idx][::-1], color=colors_mi[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in mi_top_idx][::-1], fontsize=10)
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "mutual_information.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 3. Correlation Matrix (Top Features) ────────────────────────
    print("\n  [3/4] Computing correlation matrix...")
    top_features = [feature_names[i] for i in top_idx[:15]]
    corr_matrix = train_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=[f.split('_', 1)[1][:15] for f in top_features],
                yticklabels=[f.split('_', 1)[1][:15] for f in top_features])
    ax.set_title('Feature Correlation Matrix (Top 15)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path / "correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 4. Class Distribution Plots ─────────────────────────────────
    print("\n  [4/4] Generating class distribution plots...")

    # Select top 6 features for box plots
    top6 = [feature_names[i] for i in top_idx[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (feat, ax) in enumerate(zip(top6, axes.ravel())):
        class_data = []
        labels = []
        for ft in sorted(fault_names.keys()):
            mask = y == ft
            class_data.append(X_scaled[mask, feature_names.index(feat)])
            labels.append(fault_names[ft])

        try:
            bp = ax.boxplot(class_data, tick_labels=labels, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(class_data, labels=labels, patch_artist=True)
        colors_box = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Feature Distribution by Fault Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path / "class_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Save Analysis Summary ───────────────────────────────────────
    summary = {
        'n_samples': len(train_df),
        'n_features': len(feature_names),
        'n_classes': len(np.unique(y)),
        'top_features_rf': [
            {'name': feature_names[i], 'importance': float(importances[i])}
            for i in top_idx[:20]
        ],
        'top_features_mi': [
            {'name': feature_names[i], 'mi_score': float(mi_scores[i])}
            for i in mi_top_idx[:20]
        ],
        'rf_oob_accuracy': float(rf.oob_score_) if hasattr(rf, 'oob_score_') else None,
    }

    with open(out_path / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Analysis complete! Results saved to: {out_path}")
    print(f"    - feature_importance.png")
    print(f"    - mutual_information.png")
    print(f"    - correlation_matrix.png")
    print(f"    - class_distributions.png")
    print(f"    - analysis_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    analyze_features()
