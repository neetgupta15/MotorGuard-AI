"""
Vibration Feature Extraction Engine
=====================================

Extracts 90+ features from 3-axis vibration data across three domains:
    1. Time Domain  — Statistical features (RMS, Kurtosis, Crest Factor, etc.)
    2. Frequency Domain — Spectral features (FFT peaks, centroid, entropy, etc.)
    3. Time-Frequency Domain — Wavelet Packet Energy decomposition

Each domain extractor produces features per axis (X, Y, Z), resulting in
a comprehensive feature vector suitable for machine learning classification.

Feature Summary:
    Time Domain:      14 features × 3 axes = 42 features
    Frequency Domain: 10 features × 3 axes = 30 features
    Wavelet Domain:    8 features × 3 axes = 24 features
    ─────────────────────────────────────────────────────
    Total:                                   96 features

Reference:
    - Lei, Y. "Intelligent Fault Diagnosis and Remaining Useful Life Prediction"
    - Caesarendra, W. "A Review of Feature Extraction Methods in Vibration-Based
      Condition Monitoring and Its Application for Degradation Trend Estimation"
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from pathlib import Path
from typing import List, Dict, Optional
import os
import json


class TimeFeatures:
    """
    Time-domain statistical feature extraction.

    Extracts 14 statistical features from each axis of the vibration signal.
    These features capture the amplitude distribution, energy content, and
    impulsiveness of the signal — all important indicators of machine health.

    Features:
        1.  Mean                — Average amplitude
        2.  Std                 — Standard deviation
        3.  RMS                 — Root Mean Square (vibration energy)
        4.  Peak                — Maximum absolute amplitude
        5.  Peak-to-Peak        — Range of the signal
        6.  Crest Factor        — Peak / RMS (impulsiveness)
        7.  Shape Factor        — RMS / Mean(|x|)
        8.  Impulse Factor      — Peak / Mean(|x|)
        9.  Clearance Factor    — Peak / (Mean(√|x|))²
        10. Kurtosis            — 4th moment (peakedness)
        11. Skewness            — 3rd moment (asymmetry)
        12. Variance            — Signal power spread
        13. Energy              — Sum of squared amplitudes
        14. Zero Crossing Rate  — Frequency content indicator
    """

    FEATURE_NAMES = [
        'mean', 'std', 'rms', 'peak', 'peak_to_peak', 'crest_factor',
        'shape_factor', 'impulse_factor', 'clearance_factor',
        'kurtosis', 'skewness', 'variance', 'energy', 'zero_crossing_rate'
    ]

    @staticmethod
    def extract(signal_data: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features from a single-axis signal.

        Args:
            signal_data: 1D array of vibration samples

        Returns:
            1D array of 14 features
        """
        x = signal_data.astype(np.float64)
        n = len(x)

        # Basic statistics
        mean_val = np.mean(x)
        std_val = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        peak_to_peak = np.max(x) - np.min(x)

        # Derived factors
        mean_abs = np.mean(np.abs(x))
        mean_sqrt_abs = np.mean(np.sqrt(np.abs(x)))

        crest_factor = peak / rms if rms > 0 else 0
        shape_factor = rms / mean_abs if mean_abs > 0 else 0
        impulse_factor = peak / mean_abs if mean_abs > 0 else 0
        clearance_factor = peak / (mean_sqrt_abs ** 2) if mean_sqrt_abs > 0 else 0

        # Higher-order statistics
        kurt = stats.kurtosis(x, fisher=True)  # Excess kurtosis
        skew = stats.skew(x)
        variance = np.var(x)
        energy = np.sum(x ** 2) / n

        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
        zcr = zero_crossings / n

        return np.array([
            mean_val, std_val, rms, peak, peak_to_peak, crest_factor,
            shape_factor, impulse_factor, clearance_factor,
            kurt, skew, variance, energy, zcr
        ])


class FrequencyFeatures:
    """
    Frequency-domain feature extraction using FFT analysis.

    Extracts 10 spectral features that capture the frequency content
    and distribution of vibration energy across the spectrum.

    Features:
        1.  Spectral Centroid   — "Center of mass" of the spectrum
        2.  Spectral Spread     — Bandwidth around the centroid
        3.  Spectral Entropy    — Randomness of spectral distribution
        4.  Spectral Flatness   — Tonality vs. noise-like character
        5.  Dominant Frequency  — Frequency with highest amplitude
        6.  Dominant Amplitude  — Amplitude at the dominant frequency
        7.  Mean Frequency      — Average frequency weighted by power
        8.  Band Energy Low     — Energy in 10-500 Hz band
        9.  Band Energy Mid     — Energy in 500-2000 Hz band
        10. Band Energy High    — Energy in 2000-5000 Hz band
    """

    FEATURE_NAMES = [
        'spectral_centroid', 'spectral_spread', 'spectral_entropy',
        'spectral_flatness', 'dominant_freq', 'dominant_amplitude',
        'mean_frequency', 'band_energy_low', 'band_energy_mid',
        'band_energy_high'
    ]

    @staticmethod
    def extract(signal_data: np.ndarray, fs: float = 12000.0) -> np.ndarray:
        """
        Extract frequency-domain features from a single-axis signal.

        Args:
            signal_data: 1D array of vibration samples
            fs:          Sampling frequency in Hz

        Returns:
            1D array of 10 features
        """
        x = signal_data.astype(np.float64)
        n = len(x)

        # Compute FFT (single-sided)
        X = fft(x)
        freqs = fftfreq(n, 1.0 / fs)

        # Take only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        magnitude = np.abs(X[pos_mask]) / n
        power = magnitude ** 2

        # Avoid division by zero
        total_power = np.sum(power)
        if total_power == 0:
            return np.zeros(10)

        # Normalized power spectral density
        psd_norm = power / total_power

        # 1. Spectral Centroid — weighted mean frequency
        spectral_centroid = np.sum(freqs_pos * psd_norm)

        # 2. Spectral Spread — standard deviation around centroid
        spectral_spread = np.sqrt(
            np.sum(((freqs_pos - spectral_centroid) ** 2) * psd_norm))

        # 3. Spectral Entropy — Shannon entropy of PSD
        psd_safe = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_safe * np.log2(psd_safe))

        # 4. Spectral Flatness — geometric mean / arithmetic mean
        log_power = np.log(power[power > 0] + 1e-12)
        geom_mean = np.exp(np.mean(log_power))
        arith_mean = np.mean(power)
        spectral_flatness = geom_mean / arith_mean if arith_mean > 0 else 0

        # 5-6. Dominant frequency and its amplitude
        dom_idx = np.argmax(magnitude)
        dominant_freq = freqs_pos[dom_idx]
        dominant_amplitude = magnitude[dom_idx]

        # 7. Mean frequency (weighted by power)
        mean_frequency = np.sum(freqs_pos * power) / total_power

        # 8-10. Band energies
        def band_energy(f_low, f_high):
            mask = (freqs_pos >= f_low) & (freqs_pos < f_high)
            return np.sum(power[mask]) / total_power if total_power > 0 else 0

        band_low = band_energy(10, 500)
        band_mid = band_energy(500, 2000)
        band_high = band_energy(2000, 5000)

        return np.array([
            spectral_centroid, spectral_spread, spectral_entropy,
            spectral_flatness, dominant_freq, dominant_amplitude,
            mean_frequency, band_low, band_mid, band_high
        ])


class WaveletFeatures:
    """
    Time-frequency domain feature extraction using Wavelet Packet Decomposition.

    Decomposes the signal into 8 sub-bands using a 3-level wavelet packet
    tree, and computes the energy in each sub-band. This captures both
    time and frequency information, making it effective for detecting
    transient fault signatures.

    Wavelet: Daubechies-4 (db4) — good for vibration signals
    Decomposition Level: 3 → 2³ = 8 sub-bands

    Features:
        1-8. Normalized energy in each of the 8 wavelet packet sub-bands
    """

    FEATURE_NAMES = [
        'wp_energy_band_0', 'wp_energy_band_1', 'wp_energy_band_2',
        'wp_energy_band_3', 'wp_energy_band_4', 'wp_energy_band_5',
        'wp_energy_band_6', 'wp_energy_band_7'
    ]

    @staticmethod
    def extract(signal_data: np.ndarray, wavelet: str = 'db4',
                level: int = 3) -> np.ndarray:
        """
        Extract wavelet packet energy features.

        Args:
            signal_data: 1D array of vibration samples
            wavelet:     Wavelet family ('db4', 'sym5', etc.)
            level:       Decomposition level

        Returns:
            1D array of 8 features (energy per sub-band)
        """
        x = signal_data.astype(np.float64)

        # Wavelet Packet Decomposition
        wp = pywt.WaveletPacket(data=x, wavelet=wavelet, maxlevel=level)

        # Get leaf nodes (sub-bands at the deepest level)
        nodes = [node.path for node in wp.get_level(level, 'freq')]
        n_bands = len(nodes)

        # Calculate energy in each sub-band
        energies = np.zeros(n_bands)
        for i, node_path in enumerate(nodes):
            coeffs = wp[node_path].data
            energies[i] = np.sum(coeffs ** 2)

        # Normalize energies
        total_energy = np.sum(energies)
        if total_energy > 0:
            energies /= total_energy

        # Ensure we return exactly 8 features
        if len(energies) < 8:
            energies = np.pad(energies, (0, 8 - len(energies)))
        elif len(energies) > 8:
            energies = energies[:8]

        return energies


class VibrationFeatureExtractor:
    """
    Complete feature extraction pipeline combining all three domains.

    Processes 3-axis vibration segments and produces a comprehensive
    feature vector of 96 features (32 per axis).

    Usage:
        >>> extractor = VibrationFeatureExtractor(fs=12000)
        >>> features = extractor.extract_from_segment(segment)
        >>> print(len(features))  # 96
    """

    AXIS_NAMES = ['x', 'y', 'z']

    def __init__(self, fs: float = 12000.0):
        """
        Initialize the feature extractor.

        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.time_extractor = TimeFeatures()
        self.freq_extractor = FrequencyFeatures()
        self.wavelet_extractor = WaveletFeatures()

    def get_feature_names(self) -> List[str]:
        """Get ordered list of all feature names."""
        names = []
        for axis in self.AXIS_NAMES:
            for feat in TimeFeatures.FEATURE_NAMES:
                names.append(f"{axis}_{feat}")
            for feat in FrequencyFeatures.FEATURE_NAMES:
                names.append(f"{axis}_{feat}")
            for feat in WaveletFeatures.FEATURE_NAMES:
                names.append(f"{axis}_{feat}")
        return names

    def extract_from_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single segment.

        Args:
            segment: Array of shape (window_size, 3)

        Returns:
            1D array of 96 features
        """
        all_features = []

        for axis_idx, axis_name in enumerate(self.AXIS_NAMES):
            axis_data = segment[:, axis_idx]

            # Time-domain features (14)
            time_feats = self.time_extractor.extract(axis_data)
            all_features.append(time_feats)

            # Frequency-domain features (10)
            freq_feats = self.freq_extractor.extract(axis_data, self.fs)
            all_features.append(freq_feats)

            # Wavelet features (8)
            wavelet_feats = self.wavelet_extractor.extract(axis_data)
            all_features.append(wavelet_feats)

        return np.concatenate(all_features)

    def extract_from_file(self, npy_path: str) -> np.ndarray:
        """
        Extract features from all segments in a preprocessed .npy file.

        Args:
            npy_path: Path to .npy file with shape (n_segments, window_size, 3)

        Returns:
            Array of shape (n_segments, 96)
        """
        segments = np.load(npy_path)
        features = []

        for i in range(segments.shape[0]):
            feat = self.extract_from_segment(segments[i])
            features.append(feat)

        return np.array(features)

    def extract_dataset(self, processed_dir: str, output_dir: str,
                        metadata_path: str = None) -> pd.DataFrame:
        """
        Extract features from the entire preprocessed dataset.

        Args:
            processed_dir: Path to preprocessed data directory
            output_dir:    Path to save feature files
            metadata_path: Path to processed_metadata.csv

        Returns:
            DataFrame with features and labels
        """
        proc_path = Path(processed_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        else:
            metadata = None

        feature_names = self.get_feature_names()
        all_records = []

        for split in ['train', 'val', 'test']:
            split_dir = proc_path / split
            if not split_dir.exists():
                continue

            npy_files = sorted(split_dir.glob("*.npy"))
            print(f"  Extracting features from {split}: {len(npy_files)} files...")

            split_features = []
            split_labels = []

            for npy_file in npy_files:
                features = self.extract_from_file(str(npy_file))

                # Get label from metadata
                label = -1
                severity = 0.0
                rpm = 0.0
                if metadata is not None:
                    csv_name = npy_file.stem + ".csv"
                    match = metadata[metadata['filename'] == csv_name]
                    if len(match) > 0:
                        label = int(match.iloc[0]['fault_type'])
                        severity = float(match.iloc[0].get('severity', 0))
                        rpm = float(match.iloc[0].get('rpm', 0))

                for seg_idx in range(features.shape[0]):
                    record = {
                        'source_file': npy_file.stem,
                        'segment_idx': seg_idx,
                        'split': split,
                        'fault_type': label,
                        'severity': severity,
                        'rpm': rpm,
                    }
                    for fi, fname in enumerate(feature_names):
                        record[fname] = features[seg_idx, fi]
                    all_records.append(record)

                split_features.append(features)
                split_labels.extend([label] * features.shape[0])

            # Save split-level numpy arrays
            if split_features:
                X = np.vstack(split_features)
                y = np.array(split_labels)
                np.save(str(out_path / f"X_{split}.npy"), X)
                np.save(str(out_path / f"y_{split}.npy"), y)
                print(f"    Saved {X.shape[0]} feature vectors ({X.shape[1]} features)")

        # Save combined feature DataFrame
        feature_df = pd.DataFrame(all_records)
        feature_df.to_csv(out_path / "features.csv", index=False)

        # Save feature names
        with open(out_path / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)

        print(f"\n  Feature extraction complete!")
        print(f"    Total features per sample: {len(feature_names)}")
        print(f"    Total records: {len(feature_df)}")

        return feature_df


def run_feature_extraction(data_root: str = "data"):
    """Run the complete feature extraction pipeline."""
    print("=" * 70)
    print("  Vibration Feature Extraction Pipeline")
    print("=" * 70)

    extractor = VibrationFeatureExtractor(fs=12000.0)

    print(f"\n  Feature dimensions:")
    print(f"    Time-domain:   {len(TimeFeatures.FEATURE_NAMES)} × 3 axes = "
          f"{len(TimeFeatures.FEATURE_NAMES) * 3}")
    print(f"    Frequency:     {len(FrequencyFeatures.FEATURE_NAMES)} × 3 axes = "
          f"{len(FrequencyFeatures.FEATURE_NAMES) * 3}")
    print(f"    Wavelet:       {len(WaveletFeatures.FEATURE_NAMES)} × 3 axes = "
          f"{len(WaveletFeatures.FEATURE_NAMES) * 3}")
    print(f"    Total:         {len(extractor.get_feature_names())} features")

    result = extractor.extract_dataset(
        processed_dir=os.path.join(data_root, "processed"),
        output_dir=os.path.join(data_root, "features"),
        metadata_path=os.path.join(data_root, "metadata.csv")
    )

    print("=" * 70)
    return result


if __name__ == "__main__":
    run_feature_extraction()
