"""
Vibration Data Preprocessing Pipeline
=======================================

Comprehensive preprocessing for raw accelerometer vibration data including:
- Bandpass filtering (Butterworth IIR)
- DC offset removal
- Normalization (z-score and min-max)
- Signal segmentation into fixed-length windows
- Missing value handling
- Resampling for consistency

The pipeline is designed to process 3-axis accelerometer data (X, Y, Z)
and prepare it for feature extraction and ML model training.

Reference:
    - Oppenheim, A.V. "Discrete-Time Signal Processing"
    - IEC 10816: Mechanical vibration evaluation standards
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple, List, Optional
import json
import os
import sys


class VibrationPreprocessor:
    """
    Preprocesses raw vibration data for fault detection analysis.

    Pipeline steps:
        1. Load raw CSV data
        2. Remove DC offset (mean subtraction)
        3. Apply bandpass filter
        4. Handle missing values (interpolation)
        5. Normalize signals
        6. Segment into fixed-length windows

    Example:
        >>> prep = VibrationPreprocessor(fs=12000)
        >>> segments = prep.process_file("data/raw/train/fault0_0001.csv")
        >>> print(segments.shape)  # (num_segments, window_size, 3)
    """

    def __init__(self, fs: float = 12000.0,
                 lowcut: float = 10.0,
                 highcut: float = 5000.0,
                 filter_order: int = 5,
                 window_size: int = 1024,
                 overlap: float = 0.5,
                 normalize_method: str = 'zscore'):
        """
        Initialize the preprocessor.

        Args:
            fs:               Sampling frequency [Hz]
            lowcut:           Lower cutoff frequency for bandpass filter [Hz]
            highcut:          Upper cutoff frequency for bandpass filter [Hz]
            filter_order:     Butterworth filter order
            window_size:      Number of samples per segment window
            overlap:          Overlap fraction between consecutive windows
            normalize_method: 'zscore' or 'minmax'
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.window_size = window_size
        self.overlap = overlap
        self.normalize_method = normalize_method

        # Design bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = signal.butter(filter_order, [low, high], btype='band')

    def remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """Remove DC component (mean) from each axis."""
        return data - np.mean(data, axis=0)

    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to remove frequencies outside
        the band of interest.

        Uses zero-phase filtering (filtfilt) to avoid phase distortion.
        """
        filtered = np.zeros_like(data)
        for axis in range(data.shape[1]):
            filtered[:, axis] = signal.filtfilt(self.b, self.a, data[:, axis])
        return filtered

    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handle missing or NaN values using linear interpolation.

        Args:
            data: Input array, possibly containing NaN values

        Returns:
            Array with NaN values replaced by interpolated values
        """
        if not np.any(np.isnan(data)):
            return data

        result = data.copy()
        for axis in range(data.shape[1]):
            col = data[:, axis]
            nans = np.isnan(col)
            if np.any(nans):
                # Interpolate NaN values
                valid_idx = np.where(~nans)[0]
                nan_idx = np.where(nans)[0]
                if len(valid_idx) > 1:
                    interp_func = interp1d(valid_idx, col[valid_idx],
                                          kind='linear', fill_value='extrapolate')
                    result[nan_idx, axis] = interp_func(nan_idx)
                else:
                    result[nan_idx, axis] = 0.0

        return result

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the signal using the specified method.

        z-score: (x - μ) / σ  — zero mean, unit variance
        minmax:  (x - min) / (max - min)  — scale to [0, 1]
        """
        if self.normalize_method == 'zscore':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0  # Prevent division by zero
            return (data - mean) / std

        elif self.normalize_method == 'minmax':
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            return (data - min_val) / range_val

        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")

    def segment(self, data: np.ndarray) -> np.ndarray:
        """
        Segment the signal into fixed-length overlapping windows.

        Args:
            data: Input array of shape (N, 3)

        Returns:
            Array of shape (num_segments, window_size, 3)
        """
        step = int(self.window_size * (1.0 - self.overlap))
        n_samples = data.shape[0]
        segments = []

        for start in range(0, n_samples - self.window_size + 1, step):
            end = start + self.window_size
            segments.append(data[start:end])

        if len(segments) == 0:
            # Signal shorter than window — pad with zeros
            padded = np.zeros((self.window_size, data.shape[1]))
            padded[:n_samples] = data
            segments.append(padded)

        return np.array(segments)

    def process_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the complete preprocessing pipeline to a single signal.

        Steps: DC removal → missing values → bandpass → normalize → segment

        Args:
            data: Raw 3-axis vibration data of shape (N, 3)

        Returns:
            Segmented preprocessed data of shape (num_segments, window_size, 3)
        """
        # Step 1: Handle missing values
        data = self.handle_missing_values(data)

        # Step 2: Remove DC offset
        data = self.remove_dc_offset(data)

        # Step 3: Apply bandpass filter
        data = self.apply_bandpass_filter(data)

        # Step 4: Normalize
        data = self.normalize(data)

        # Step 5: Segment into windows
        segments = self.segment(data)

        return segments

    def process_file(self, filepath: str) -> np.ndarray:
        """
        Load and process a single CSV file.

        Args:
            filepath: Path to CSV file with columns [accel_x, accel_y, accel_z]

        Returns:
            Preprocessed segments of shape (num_segments, window_size, 3)
        """
        df = pd.read_csv(filepath)
        data = df[['accel_x', 'accel_y', 'accel_z']].values
        return self.process_signal(data)

    def process_dataset(self, data_dir: str, output_dir: str,
                        metadata_path: str = None) -> pd.DataFrame:
        """
        Process an entire dataset directory.

        Reads raw CSV files, preprocesses them, and saves the segmented
        data as numpy arrays.

        Args:
            data_dir:      Path to raw data directory (e.g., 'data/raw')
            output_dir:    Path to save processed data (e.g., 'data/processed')
            metadata_path: Path to metadata.csv

        Returns:
            DataFrame with processed file metadata
        """
        data_path = Path(data_dir)
        out_path = Path(output_dir)

        # Load metadata if available
        if metadata_path and os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = None

        processed_records = []

        for split in ['train', 'val', 'test']:
            split_dir = data_path / split
            if not split_dir.exists():
                continue

            out_split = out_path / split
            out_split.mkdir(parents=True, exist_ok=True)

            csv_files = sorted(split_dir.glob("*.csv"))
            print(f"  Processing {split}: {len(csv_files)} files...")

            for csv_file in csv_files:
                try:
                    segments = self.process_file(str(csv_file))

                    # Save as numpy array
                    npy_file = out_split / f"{csv_file.stem}.npy"
                    np.save(str(npy_file), segments)

                    # Get metadata for this file
                    record = {
                        'filename': csv_file.name,
                        'split': split,
                        'n_segments': segments.shape[0],
                        'processed_file': str(npy_file),
                    }

                    if metadata is not None:
                        match = metadata[metadata['filename'] == csv_file.name]
                        if len(match) > 0:
                            record['fault_type'] = int(match.iloc[0]['fault_type'])
                            record['fault_name'] = match.iloc[0]['fault_name']
                            record['severity'] = float(match.iloc[0]['severity'])
                            record['rpm'] = float(match.iloc[0]['rpm'])

                    processed_records.append(record)

                except Exception as e:
                    print(f"    WARNING: Failed to process {csv_file.name}: {e}")

        # Save processing summary
        proc_df = pd.DataFrame(processed_records)
        proc_df.to_csv(out_path / "processed_metadata.csv", index=False)

        # Save preprocessing config
        config = {
            'fs': self.fs,
            'lowcut': self.lowcut,
            'highcut': self.highcut,
            'filter_order': self.filter_order,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'normalize_method': self.normalize_method,
        }
        with open(out_path / "preprocessing_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        return proc_df


def run_preprocessing(data_root: str = "data"):
    """Run the complete preprocessing pipeline on generated data."""
    print("=" * 70)
    print("  Vibration Data Preprocessing Pipeline")
    print("=" * 70)

    preprocessor = VibrationPreprocessor(
        fs=12000.0,
        lowcut=10.0,
        highcut=5000.0,
        filter_order=5,
        window_size=1024,
        overlap=0.5,
        normalize_method='zscore'
    )

    print(f"\n  Configuration:")
    print(f"    Sampling freq  : {preprocessor.fs} Hz")
    print(f"    Bandpass       : {preprocessor.lowcut}-{preprocessor.highcut} Hz")
    print(f"    Filter order   : {preprocessor.filter_order}")
    print(f"    Window size    : {preprocessor.window_size} samples")
    print(f"    Overlap        : {preprocessor.overlap:.0%}")
    print(f"    Normalization  : {preprocessor.normalize_method}")

    result = preprocessor.process_dataset(
        data_dir=os.path.join(data_root, "raw"),
        output_dir=os.path.join(data_root, "processed"),
        metadata_path=os.path.join(data_root, "metadata.csv")
    )

    print(f"\n  Preprocessing complete!")
    print(f"    Total files processed: {len(result)}")
    if 'fault_type' in result.columns:
        print(f"    Segments per class:")
        for ft in sorted(result['fault_type'].unique()):
            subset = result[result['fault_type'] == ft]
            total_segs = subset['n_segments'].sum()
            print(f"      Class {int(ft)}: {total_segs} segments from "
                  f"{len(subset)} files")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_preprocessing()
