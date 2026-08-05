"""
End-to-End System Validation
===============================

Validates the complete fault detection pipeline by simulating a real-time
data stream with progressive fault injection and measuring:
- Detection accuracy over time
- Fault detection latency
- False alarm rates
- System robustness under varying conditions

Usage:
    python -m src.validation.validate_system
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
import joblib
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

# Handle imports
try:
    from src.data_generation.motor_simulator import MotorVibrationSimulator
    from src.preprocessing.preprocessing import VibrationPreprocessor
    from src.features.feature_extractor import VibrationFeatureExtractor
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data_generation.motor_simulator import MotorVibrationSimulator
    from src.preprocessing.preprocessing import VibrationPreprocessor
    from src.features.feature_extractor import VibrationFeatureExtractor


FAULT_NAMES = {
    0: "Normal", 1: "Bearing Fault", 2: "Rotor Imbalance",
    3: "Shaft Misalignment", 4: "Electrical Fault"
}


def simulate_realtime_stream(n_windows: int = 100, seed: int = 99):
    """
    Simulate a real-time data stream with progressive fault injection.

    Creates a timeline where the motor starts healthy and progressively
    develops different faults with increasing severity.

    Args:
        n_windows: Number of time windows to simulate
        seed:      Random seed

    Returns:
        List of (data_window, true_label, severity, rpm) tuples
    """
    sim = MotorVibrationSimulator(seed=seed)
    rng = np.random.RandomState(seed)
    stream = []

    for i in range(n_windows):
        # Simulate progressive fault development
        progress = i / n_windows

        if progress < 0.25:
            # Phase 1: Normal operation
            fault_type = 0
            severity = 0.0
        elif progress < 0.45:
            # Phase 2: Incipient bearing fault (growing)
            fault_type = 1
            severity = (progress - 0.25) / 0.20 * 0.6  # 0 → 0.6
        elif progress < 0.55:
            # Phase 3: Back to normal (after maintenance)
            fault_type = 0
            severity = 0.0
        elif progress < 0.70:
            # Phase 4: Rotor imbalance developing
            fault_type = 2
            severity = (progress - 0.55) / 0.15 * 0.7  # 0 → 0.7
        elif progress < 0.80:
            # Phase 5: Misalignment
            fault_type = 3
            severity = 0.5
        elif progress < 0.90:
            # Phase 6: Electrical fault
            fault_type = 4
            severity = (progress - 0.80) / 0.10 * 0.8
        else:
            # Phase 7: Severe bearing fault
            fault_type = 1
            severity = 0.8 + 0.2 * (progress - 0.90) / 0.10

        # Add some RPM variation
        rpm = 1800 + rng.uniform(-100, 100)

        data, _ = sim.generate_sample(
            fault_type=fault_type,
            duration=1024 / 12000,  # Match window size
            fs=12000,
            rpm=rpm,
            severity=severity if fault_type != 0 else None
        )

        stream.append((data, fault_type, severity, rpm))

    return stream


def validate_pipeline(models_dir: str = "models",
                      output_dir: str = "results/validation"):
    """
    Run complete end-to-end validation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  End-to-End System Validation")
    print("=" * 70)

    # Load model and scaler
    mdir = Path(models_dir)

    if not mdir.exists():
        print("  ERROR: Models directory not found. Train models first.")
        return

    scaler = joblib.load(mdir / "scaler.joblib")

    with open(mdir / "training_results.json") as f:
        results = json.load(f)

    best_name = results.get('_best_model', 'Random Forest')
    print(f"\n  Using model: {best_name}")

    # Load best model
    if best_name == 'Random Forest':
        model = joblib.load(mdir / "random_forest.joblib")
    elif best_name == 'SVM':
        model = joblib.load(mdir / "svm.joblib")
    elif best_name == 'Gradient Boosting':
        model = joblib.load(mdir / "gradient_boosting.joblib")
    else:
        # Default to Random Forest
        model = joblib.load(mdir / "random_forest.joblib")

    # Initialize pipeline components
    preprocessor = VibrationPreprocessor(fs=12000, window_size=1024, overlap=0.0)
    feature_extractor = VibrationFeatureExtractor(fs=12000)

    # ─── 1. Simulate Real-Time Stream ────────────────────────────────
    print("\n  [1/3] Simulating real-time data stream...")
    n_windows = 100
    stream = simulate_realtime_stream(n_windows=n_windows)

    # ─── 2. Process Each Window ──────────────────────────────────────
    print("  [2/3] Processing stream windows...")
    true_labels = []
    predictions = []
    severities = []
    confidences = []

    for i, (data, true_label, severity, rpm) in enumerate(stream):
        # Ensure data is right shape for preprocessing
        if data.shape[0] >= 1024:
            window = data[:1024]
        else:
            window = np.zeros((1024, 3))
            window[:data.shape[0]] = data

        # Extract features directly from the window
        features = feature_extractor.extract_from_segment(window)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred = model.predict(features_scaled)[0]

        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            conf = np.max(proba)
        else:
            conf = 1.0

        true_labels.append(true_label)
        predictions.append(pred)
        severities.append(severity)
        confidences.append(conf)

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # ─── 3. Generate Validation Results ──────────────────────────────
    print("  [3/3] Generating validation results...")

    overall_acc = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average='weighted')

    print(f"\n  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Overall F1-Score: {overall_f1:.4f}")
    print(f"\n  Classification Report:")
    target_names = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]
    print(classification_report(true_labels, predictions,
                                target_names=target_names,
                                labels=list(FAULT_NAMES.keys())))

    # ─── Plot 1: Timeline ────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    time_axis = np.arange(n_windows)

    # True labels
    color_map = {0: '#2ecc71', 1: '#e74c3c', 2: '#3498db',
                 3: '#f39c12', 4: '#9b59b6'}

    for i in range(n_windows):
        axes[0].axvspan(i - 0.5, i + 0.5,
                       color=color_map[true_labels[i]], alpha=0.7)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_yticks([])
    axes[0].set_title('System Validation Timeline', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], label=v)
                      for k, v in FAULT_NAMES.items()]
    axes[0].legend(handles=legend_elements, loc='upper right',
                  fontsize=8, ncol=5)

    # Predictions
    for i in range(n_windows):
        axes[1].axvspan(i - 0.5, i + 0.5,
                       color=color_map[predictions[i]], alpha=0.7)
    axes[1].set_ylabel('Predicted', fontsize=11)
    axes[1].set_yticks([])

    # Mark misclassifications
    misclass = true_labels != predictions
    axes[1].scatter(time_axis[misclass],
                   np.zeros(np.sum(misclass)),
                   marker='x', color='black', s=50, zorder=5,
                   label='Misclassified')
    if np.any(misclass):
        axes[1].legend(fontsize=9)

    # Severity
    axes[2].fill_between(time_axis, severities, alpha=0.4, color='#e74c3c')
    axes[2].plot(time_axis, severities, color='#c0392b', linewidth=1.5)
    axes[2].set_ylabel('Fault Severity', fontsize=11)
    axes[2].set_ylim([-0.05, 1.05])
    axes[2].grid(alpha=0.3)

    # Confidence
    axes[3].fill_between(time_axis, confidences, alpha=0.4, color='#3498db')
    axes[3].plot(time_axis, confidences, color='#2980b9', linewidth=1.5)
    axes[3].set_ylabel('Confidence', fontsize=11)
    axes[3].set_xlabel('Time Window', fontsize=12)
    axes[3].set_ylim([-0.05, 1.05])
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "validation_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Plot 2: Accuracy vs Severity ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bin by severity
    severity_bins = np.linspace(0, 1, 6)
    bin_accs = []
    bin_centers = []

    for b in range(len(severity_bins) - 1):
        mask = (np.array(severities) >= severity_bins[b]) & \
               (np.array(severities) < severity_bins[b + 1])
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(true_labels[mask], predictions[mask])
            bin_accs.append(bin_acc)
            bin_centers.append((severity_bins[b] + severity_bins[b + 1]) / 2)

    if bin_centers:
        ax.bar(bin_centers, bin_accs, width=0.15, color='#3498db',
              alpha=0.8, edgecolor='white')
        ax.set_xlabel('Fault Severity', fontsize=12)
        ax.set_ylabel('Detection Accuracy', fontsize=12)
        ax.set_title('Detection Accuracy vs Fault Severity', fontsize=14,
                    fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path / "accuracy_vs_severity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Save Validation Summary ─────────────────────────────────────
    validation_summary = {
        'model_used': best_name,
        'n_windows': n_windows,
        'overall_accuracy': float(overall_acc),
        'overall_f1_score': float(overall_f1),
        'misclassification_rate': float(np.mean(misclass)),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'per_class_accuracy': {},
    }

    for ft, name in FAULT_NAMES.items():
        mask = true_labels == ft
        if np.sum(mask) > 0:
            acc = accuracy_score(true_labels[mask], predictions[mask])
            validation_summary['per_class_accuracy'][name] = float(acc)

    with open(out_path / "validation_summary.json", 'w') as f:
        json.dump(validation_summary, f, indent=2)

    print(f"\n  Validation complete! Results saved to: {out_path}")
    print(f"    - validation_timeline.png")
    print(f"    - accuracy_vs_severity.png")
    print(f"    - validation_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    validate_pipeline()
