"""
Model Evaluation and Visualization
=====================================

Generates comprehensive evaluation visualizations for trained models:
- Confusion matrices (heatmaps)
- ROC curves and AUC scores
- Precision-Recall curves
- Model comparison bar charts
- Training history plots (for DNN)

Usage:
    python -m src.models.evaluate_models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             accuracy_score, f1_score)
from sklearn.preprocessing import label_binarize, StandardScaler

FAULT_NAMES = {
    0: "Normal", 1: "Bearing", 2: "Imbalance",
    3: "Misalign.", 4: "Electrical"
}


def evaluate_and_visualize(features_dir: str = "data/features",
                            models_dir: str = "models",
                            output_dir: str = "results/model_evaluation"):
    """
    Generate comprehensive evaluation visualizations.

    Args:
        features_dir: Path to feature data
        models_dir:   Path to trained models
        output_dir:   Path to save evaluation plots
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Model Evaluation & Visualization")
    print("=" * 70)

    # Load test data
    fdir = Path(features_dir)
    X_test = np.load(fdir / "X_test.npy")
    y_test = np.load(fdir / "y_test.npy")
    np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Load scaler
    mdir = Path(models_dir)
    scaler = joblib.load(mdir / "scaler.joblib")
    X_test_scaled = scaler.transform(X_test)

    n_classes = len(np.unique(y_test))
    class_names = [FAULT_NAMES[i] for i in range(n_classes)]

    # Load training results
    with open(mdir / "training_results.json") as f:
        results = json.load(f)

    # ─── 1. Model Comparison Bar Chart ───────────────────────────────
    print("\n  [1/4] Generating model comparison chart...")
    model_names = []
    accuracies = []
    f1_scores = []

    for name, res in results.items():
        if name.startswith('_'):
            continue
        model_names.append(name)
        accuracies.append(res['accuracy'])
        f1_scores.append(res['f1_score'])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score',
                   color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 2. Confusion Matrices ───────────────────────────────────────
    print("  [2/4] Generating confusion matrices...")

    # Load each model and generate predictions
    model_predictions = {}

    for name in model_names:
        try:
            if name == 'Random Forest':
                model = joblib.load(mdir / "random_forest.joblib")
                pred = model.predict(X_test_scaled)
            elif name == 'SVM':
                model = joblib.load(mdir / "svm.joblib")
                pred = model.predict(X_test_scaled)
            elif name == 'Gradient Boosting':
                model = joblib.load(mdir / "gradient_boosting.joblib")
                pred = model.predict(X_test_scaled)
            elif name == 'DNN':
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(str(mdir / "dnn_model.keras"))
                    pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
                except Exception:
                    continue
            else:
                continue
            model_predictions[name] = pred
        except Exception as e:
            print(f"    WARNING: Could not load {name}: {e}")

    n_models = len(model_predictions)
    if n_models > 0:
        fig, axes = plt.subplots(1, min(n_models, 4), figsize=(6 * min(n_models, 4), 5))
        if n_models == 1:
            axes = [axes]

        for ax, (name, pred) in zip(axes, model_predictions.items()):
            cm = confusion_matrix(y_test, pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar=False)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)

        fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path / "confusion_matrices.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ─── 3. ROC Curves ───────────────────────────────────────────────
    print("  [3/4] Generating ROC curves...")

    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(6 * min(n_models, 4), 5))
    if n_models == 1:
        axes = [axes]

    colors_roc = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    for ax, (name, pred) in zip(axes, model_predictions.items()):
        # Get probability predictions if available
        try:
            if name == 'Random Forest':
                model = joblib.load(mdir / "random_forest.joblib")
                proba = model.predict_proba(X_test_scaled)
            elif name == 'SVM':
                model = joblib.load(mdir / "svm.joblib")
                proba = model.predict_proba(X_test_scaled)
            elif name == 'Gradient Boosting':
                model = joblib.load(mdir / "gradient_boosting.joblib")
                proba = model.predict_proba(X_test_scaled)
            elif name == 'DNN':
                from tensorflow import keras
                model = keras.models.load_model(str(mdir / "dnn_model.keras"))
                proba = model.predict(X_test_scaled, verbose=0)
            else:
                continue

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                       label=f'{class_names[i]} (AUC={roc_auc:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'N/A\n{str(e)[:30]}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name}', fontsize=12)

    fig.suptitle('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path / "roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 4. Per-Class Performance ─────────────────────────────────────
    print("  [4/4] Generating per-class performance chart...")

    best_name = results.get('_best_model', model_names[0])
    if best_name in model_predictions:
        best_pred = model_predictions[best_name]
    else:
        best_pred = list(model_predictions.values())[0]

    report = classification_report(y_test, best_pred, target_names=class_names,
                                   output_dict=True)

    metrics = ['precision', 'recall', 'f1-score']
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [report[cn][metric] for cn in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                     alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance ({best_name})', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path / "per_class_performance.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Evaluation complete! Plots saved to: {out_path}")
    print(f"    - model_comparison.png")
    print(f"    - confusion_matrices.png")
    print(f"    - roc_curves.png")
    print(f"    - per_class_performance.png")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_and_visualize()
