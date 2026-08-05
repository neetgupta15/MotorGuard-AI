"""
Dataset Generation Script
=========================

Batch generates synthetic vibration data for all fault classes,
creating a complete train/validation/test dataset for ML model training.

Output Structure:
    data/raw/          — Raw 3-axis vibration CSV files
    data/processed/    — Preprocessed and segmented data
    data/features/     — Extracted feature matrices

Usage:
    python -m src.data_generation.generate_dataset
    python -m src.data_generation.generate_dataset --samples 500 --duration 2.0
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Handle both module and script execution
try:
    from .motor_simulator import MotorVibrationSimulator, MotorParameters
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data_generation.motor_simulator import MotorVibrationSimulator, MotorParameters


def generate_dataset(output_dir: str = "data",
                     samples_per_class: int = 200,
                     duration: float = 1.0,
                     fs: float = 12000.0,
                     seed: int = 42,
                     train_ratio: float = 0.70,
                     val_ratio: float = 0.15):
    """
    Generate complete dataset with all fault classes.

    Args:
        output_dir:         Root output directory
        samples_per_class:  Number of samples per fault class
        duration:           Duration of each sample in seconds
        fs:                 Sampling frequency in Hz
        seed:               Random seed for reproducibility
        train_ratio:        Fraction for training set
        val_ratio:          Fraction for validation set (rest = test)

    Generates:
        - CSV files with 3-axis vibration data
        - metadata.json with generation parameters
        - dataset_info.json with dataset statistics
    """
    output_path = Path(output_dir)
    raw_dir = output_path / "raw"

    # Create directories
    for split in ['train', 'val', 'test']:
        (raw_dir / split).mkdir(parents=True, exist_ok=True)

    # Initialize simulator
    sim = MotorVibrationSimulator(seed=seed)
    total_samples = samples_per_class * 5  # 5 fault classes

    print("=" * 70)
    print("  Electric Motor Fault Detection — Dataset Generation")
    print("=" * 70)
    print(f"  Samples per class : {samples_per_class}")
    print(f"  Total samples     : {total_samples}")
    print(f"  Duration          : {duration}s @ {fs:.0f} Hz")
    print(f"  Split             : {train_ratio:.0%} / {val_ratio:.0%} / "
          f"{1 - train_ratio - val_ratio:.0%}")
    print("=" * 70)

    all_metadata = []
    rng = np.random.RandomState(seed)

    for fault_type in range(5):
        fault_name = MotorVibrationSimulator.FAULT_NAMES[fault_type]
        print(f"\n  Generating [{fault_type}] {fault_name}...")

        for i in range(samples_per_class):
            # Generate sample with random variation
            sample_seed = seed + fault_type * 10000 + i
            sim_instance = MotorVibrationSimulator(seed=sample_seed)

            # Vary RPM slightly for each sample
            rpm_variation = rng.uniform(0.85, 1.15) * sim.params.nominal_rpm

            # Vary severity for fault samples
            if fault_type == 0:
                severity = 0.0
            else:
                severity = rng.uniform(0.15, 1.0)

            data, meta = sim_instance.generate_sample(
                fault_type=fault_type,
                duration=duration,
                fs=fs,
                rpm=rpm_variation,
                severity=severity if fault_type != 0 else None
            )

            # Determine split
            r = rng.random()
            if r < train_ratio:
                split = 'train'
            elif r < train_ratio + val_ratio:
                split = 'val'
            else:
                split = 'test'

            # Create filename
            sample_id = f"fault{fault_type}_{i:04d}"
            filename = f"{sample_id}.csv"

            # Save vibration data as CSV
            df = pd.DataFrame(data, columns=['accel_x', 'accel_y', 'accel_z'])
            df.to_csv(raw_dir / split / filename, index=False, float_format='%.6f')

            # Store metadata
            meta['sample_id'] = sample_id
            meta['filename'] = filename
            meta['split'] = split
            all_metadata.append(meta)

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"    Generated {i + 1}/{samples_per_class} samples")

    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(output_path / "metadata.csv", index=False)

    # Save dataset info
    dataset_info = {
        'generated_at': datetime.now().isoformat(),
        'total_samples': total_samples,
        'samples_per_class': samples_per_class,
        'num_classes': 5,
        'class_names': MotorVibrationSimulator.FAULT_NAMES,
        'duration_seconds': duration,
        'sampling_frequency_hz': fs,
        'axes': ['accel_x', 'accel_y', 'accel_z'],
        'seed': seed,
        'splits': {
            'train': int(metadata_df[metadata_df['split'] == 'train'].shape[0]),
            'val': int(metadata_df[metadata_df['split'] == 'val'].shape[0]),
            'test': int(metadata_df[metadata_df['split'] == 'test'].shape[0]),
        },
        'class_distribution': {
            name: int(metadata_df[metadata_df['fault_type'] == ft].shape[0])
            for ft, name in MotorVibrationSimulator.FAULT_NAMES.items()
        },
        'motor_parameters': {
            'nominal_rpm': sim.params.nominal_rpm,
            'num_poles': sim.params.num_poles,
            'num_slots': sim.params.num_slots,
            'rotor_mass': sim.params.rotor_mass,
            'line_frequency': sim.params.line_freq,
            'bearing_balls': sim.params.bearing.n_balls,
        }
    }

    with open(output_path / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  Dataset Generation Complete!")
    print("=" * 70)
    print(f"  Output directory: {output_path.resolve()}")
    print(f"\n  Split Distribution:")
    for split in ['train', 'val', 'test']:
        count = dataset_info['splits'][split]
        print(f"    {split:5s}: {count:4d} samples "
              f"({count/total_samples:.1%})")

    print(f"\n  Class Distribution:")
    for name, count in dataset_info['class_distribution'].items():
        print(f"    {name:25s}: {count:4d} samples")

    print(f"\n  Files saved:")
    print(f"    Raw data  : {raw_dir}")
    print(f"    Metadata  : {output_path / 'metadata.csv'}")
    print(f"    Info      : {output_path / 'dataset_info.json'}")
    print("=" * 70)

    return metadata_df, dataset_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic motor vibration dataset")
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Samples per class (default: 200)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Sample duration in seconds (default: 1.0)')
    parser.add_argument('--fs', type=float, default=12000.0,
                        help='Sampling frequency in Hz (default: 12000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        samples_per_class=args.samples,
        duration=args.duration,
        fs=args.fs,
        seed=args.seed
    )
