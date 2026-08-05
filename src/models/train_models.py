"""
Fault Detection Model Training
================================

Trains and compares multiple machine learning models for motor fault
classification using extracted vibration features.

Models Trained:
    1. Random Forest (RF) — Ensemble of decision trees, robust baseline
    2. Support Vector Machine (SVM) — RBF kernel, good with high-dim features
    3. Gradient Boosting (XGBoost) — Sequential ensemble, state-of-art tabular
    4. Dense Neural Network (DNN) — Feature-based deep learning
    5. 1D Convolutional Neural Network (CNN) — Raw signal deep learning

Each model is trained with cross-validation and hyperparameter tuning.
The best model is selected based on weighted F1-score.

Usage:
    python -m src.models.train_models
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

warnings.filterwarnings('ignore')

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False


FAULT_NAMES = {
    0: "Normal", 1: "Bearing Fault", 2: "Rotor Imbalance",
    3: "Shaft Misalignment", 4: "Electrical Fault"
}


def load_feature_data(features_dir: str = "data/features"):
    """
    Load extracted features and labels.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    fdir = Path(features_dir)

    X_train = np.load(fdir / "X_train.npy")
    y_train = np.load(fdir / "y_train.npy")
    X_val = np.load(fdir / "X_val.npy")
    y_val = np.load(fdir / "y_val.npy")
    X_test = np.load(fdir / "X_test.npy")
    y_test = np.load(fdir / "y_test.npy")

    with open(fdir / "feature_names.json") as f:
        feature_names = json.load(f)

    # Clean data — replace NaN/Inf
    for arr in [X_train, X_val, X_test]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier with hyperparameter tuning.

    Random Forest is an ensemble of decision trees that uses bagging
    and feature randomization for robust classification.
    """
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Training Random Forest Classifier      │")
    print("  └─────────────────────────────────────────┘")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='f1_weighted',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    val_pred = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Val Accuracy: {val_acc:.4f}")
    print(f"    Val F1-Score: {val_f1:.4f}")

    return best_rf, val_acc, val_f1


def train_svm(X_train, y_train, X_val, y_val):
    """
    Train Support Vector Machine with RBF kernel.

    SVM finds the optimal hyperplane that separates different fault
    classes in the feature space with maximum margin.
    """
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Training SVM Classifier (RBF Kernel)   │")
    print("  └─────────────────────────────────────────┘")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
    }

    svm = SVC(random_state=42, probability=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        svm, param_grid, cv=cv, scoring='f1_weighted',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    val_pred = best_svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Val Accuracy: {val_acc:.4f}")
    print(f"    Val F1-Score: {val_f1:.4f}")

    return best_svm, val_acc, val_f1


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train Gradient Boosting classifier.

    Uses XGBoost if available, otherwise falls back to sklearn's
    GradientBoostingClassifier.
    """
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Training Gradient Boosting Classifier   │")
    print("  └─────────────────────────────────────────┘")

    if HAS_XGBOOST:
        print("    Using XGBoost backend")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
        model = xgb.XGBClassifier(
            random_state=42, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1
        )
    else:
        print("    Using sklearn GradientBoosting (XGBoost not installed)")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
        }
        model = GradientBoostingClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1_weighted',
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print(f"    Best params: {grid_search.best_params_}")
    print(f"    Val Accuracy: {val_acc:.4f}")
    print(f"    Val F1-Score: {val_f1:.4f}")

    return best_model, val_acc, val_f1


def build_dnn_model(input_dim: int, n_classes: int) -> 'keras.Model':
    """
    Build a Dense Neural Network for feature-based classification.

    Architecture:
        Input → Dense(256) → BN → Dropout(0.3) →
        Dense(128) → BN → Dropout(0.3) →
        Dense(64) → BN → Dropout(0.2) →
        Dense(n_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_dnn(X_train, y_train, X_val, y_val):
    """Train Dense Neural Network classifier."""
    if not HAS_TF:
        print("\n  [SKIP] DNN — TensorFlow not installed")
        return None, 0, 0

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Training Dense Neural Network (DNN)     │")
    print("  └─────────────────────────────────────────┘")

    n_classes = len(np.unique(y_train))
    model = build_dnn_model(X_train.shape[1], n_classes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print(f"    Epochs trained: {len(history.history['loss'])}")
    print(f"    Val Accuracy: {val_acc:.4f}")
    print(f"    Val F1-Score: {val_f1:.4f}")

    return model, val_acc, val_f1


def build_cnn_model(window_size: int, n_channels: int,
                    n_classes: int) -> 'keras.Model':
    """
    Build a 1D Convolutional Neural Network for raw signal classification.

    Architecture:
        Input(window, channels) →
        Conv1D(64, 7) → BN → MaxPool →
        Conv1D(128, 5) → BN → MaxPool →
        Conv1D(256, 3) → BN → GlobalAvgPool →
        Dense(128) → Dropout → Dense(n_classes, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, n_channels)),
        layers.Conv1D(64, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_all_models(features_dir: str = "data/features",
                     output_dir: str = "models"):
    """
    Train all models and compare performance.

    Args:
        features_dir: Path to extracted features
        output_dir:   Path to save trained models

    Returns:
        dict with model comparison results
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Motor Fault Detection — Model Training Pipeline")
    print("=" * 70)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        load_feature_data(features_dir)

    print(f"\n  Dataset Summary:")
    print(f"    Training:   {X_train.shape[0]:5d} samples × {X_train.shape[1]} features")
    print(f"    Validation: {X_val.shape[0]:5d} samples × {X_val.shape[1]} features")
    print(f"    Test:       {X_test.shape[0]:5d} samples × {X_test.shape[1]} features")
    print(f"    Classes:    {len(np.unique(y_train))} fault types")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, out_path / "scaler.joblib")

    # ─── Train all models ─────────────────────────────────────────────
    results = {}

    # 1. Random Forest
    rf_model, rf_acc, rf_f1 = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val)
    joblib.dump(rf_model, out_path / "random_forest.joblib")
    results['Random Forest'] = {'accuracy': rf_acc, 'f1_score': rf_f1,
                                 'model_file': 'random_forest.joblib'}

    # 2. SVM
    svm_model, svm_acc, svm_f1 = train_svm(
        X_train_scaled, y_train, X_val_scaled, y_val)
    joblib.dump(svm_model, out_path / "svm.joblib")
    results['SVM'] = {'accuracy': svm_acc, 'f1_score': svm_f1,
                       'model_file': 'svm.joblib'}

    # 3. Gradient Boosting
    gb_model, gb_acc, gb_f1 = train_gradient_boosting(
        X_train_scaled, y_train, X_val_scaled, y_val)
    joblib.dump(gb_model, out_path / "gradient_boosting.joblib")
    results['Gradient Boosting'] = {'accuracy': gb_acc, 'f1_score': gb_f1,
                                     'model_file': 'gradient_boosting.joblib'}

    # 4. DNN (if TensorFlow available)
    dnn_model, dnn_acc, dnn_f1 = train_dnn(
        X_train_scaled, y_train, X_val_scaled, y_val)
    if dnn_model is not None:
        dnn_model.save(str(out_path / "dnn_model.keras"))
        results['DNN'] = {'accuracy': dnn_acc, 'f1_score': dnn_f1,
                           'model_file': 'dnn_model.keras'}

    # ─── Select Best Model ────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_f1 = results[best_name]['f1_score']

    # Evaluate best model on test set
    print("\n  ╔═════════════════════════════════════════╗")
    print(f"  ║  Best Model: {best_name:28s} ║")
    print(f"  ║  Val F1-Score: {best_f1:.4f}                   ║")
    print("  ╚═════════════════════════════════════════╝")

    # Test evaluation with best model
    if best_name == 'Random Forest':
        test_pred = rf_model.predict(X_test_scaled)
    elif best_name == 'SVM':
        test_pred = svm_model.predict(X_test_scaled)
    elif best_name == 'Gradient Boosting':
        test_pred = gb_model.predict(X_test_scaled)
    elif best_name == 'DNN' and dnn_model:
        test_pred = np.argmax(dnn_model.predict(X_test_scaled, verbose=0), axis=1)
    else:
        test_pred = rf_model.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')

    print(f"\n  Test Set Performance:")
    print(f"    Accuracy: {test_acc:.4f}")
    print(f"    F1-Score: {test_f1:.4f}")
    print(f"\n  Classification Report:")
    target_names = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]
    print(classification_report(y_test, test_pred, target_names=target_names))

    # ─── Save Results ─────────────────────────────────────────────────
    results['_best_model'] = best_name
    results['_test_accuracy'] = float(test_acc)
    results['_test_f1_score'] = float(test_f1)
    results['_confusion_matrix'] = confusion_matrix(y_test, test_pred).tolist()
    results['_classification_report'] = classification_report(
        y_test, test_pred, target_names=target_names, output_dict=True)

    with open(out_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Model comparison table
    print("\n  ┌────────────────────┬──────────┬──────────┐")
    print("  │ Model              │ Val Acc  │ Val F1   │")
    print("  ├────────────────────┼──────────┼──────────┤")
    for name, res in results.items():
        if name.startswith('_'):
            continue
        marker = " ★" if name == best_name else "  "
        print(f"  │ {name:18s} │ {res['accuracy']:.4f}   │ {res['f1_score']:.4f}   │{marker}")
    print("  └────────────────────┴──────────┴──────────┘")

    print(f"\n  Models saved to: {out_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    train_all_models()
