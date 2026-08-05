"""
Preset Samples Generator for Dashboard
========================================

Generates 50+ diverse motor sound and vibration presets across all 5 fault classes
with varying RPMs, severities, and operating conditions.
Outputs a structured `dashboard/preset_samples.json` file for the dashboard UI.
"""

import json
import numpy as np
from pathlib import Path
from src.data_generation.motor_simulator import MotorVibrationSimulator
from src.features.feature_extractor import VibrationFeatureExtractor

def compute_fft(signal, fs=400, n_bins=64):
    """Simple FFT computation for preset data."""
    N = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) / N
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    
    # Downsample to n_bins for dashboard display
    idx = np.linspace(0, len(fft_vals) - 1, n_bins, dtype=int)
    return freqs[idx].tolist(), fft_vals[idx].tolist()

def generate_preset_library(output_file: str = "dashboard/preset_samples.json"):
    sim = MotorVibrationSimulator(seed=42)
    extractor = VibrationFeatureExtractor(fs=12000)
    
    # Define 50+ configurations (10 per class)
    rpms = [1200, 1500, 1800, 2400, 3000, 3600]
    fault_configs = [
        # Normal samples (10)
        {"class": 0, "name": "Standard Normal Operating State", "rpm": 1800, "sev": 0.0, "desc": "Optimal operating conditions. Low baseline vibration across all axes."},
        {"class": 0, "name": "High-Speed Nominal Run", "rpm": 3600, "sev": 0.0, "desc": "Clean operation at max rated RPM with minimal harmonic noise."},
        {"class": 0, "name": "Low-Speed Smooth Baseline", "rpm": 1200, "sev": 0.0, "desc": "Quiet low-speed operation with stable shaft rotation."},
        {"class": 0, "name": "Nominal Load Steady Run", "rpm": 1500, "sev": 0.0, "desc": "Full load operation with normal electromagnetic hum."},
        {"class": 0, "name": "Medium-Speed Quiet Operation", "rpm": 2400, "sev": 0.0, "desc": "Smooth shaft rotation with minor structural resonance."},
        {"class": 0, "name": "Light Load Baseline", "rpm": 1800, "sev": 0.0, "desc": "Unloaded test bench baseline profile."},
        {"class": 0, "name": "VFD Driven Normal Operation", "rpm": 2100, "sev": 0.0, "desc": "Normal operation under VFD switching control."},
        {"class": 0, "name": "Thermally Stable Run", "rpm": 3000, "sev": 0.0, "desc": "Steady state thermal equilibrium operation."},
        {"class": 0, "name": "Re-lubricated Bearing Baseline", "rpm": 1800, "sev": 0.0, "desc": "Post-maintenance freshly greased bearing run."},
        {"class": 0, "name": "Cold Start Nominal Baseline", "rpm": 1500, "sev": 0.0, "desc": "Initial startup baseline after ambient warm-up."},

        # Bearing Fault samples (10)
        {"class": 1, "name": "Incipient Outer Race Flaw (BPFO)", "rpm": 1800, "sev": 0.25, "desc": "Early stage micro-pitting on outer race generating periodic high-frequency impacts."},
        {"class": 1, "name": "Moderate Inner Race Defect (BPFI)", "rpm": 1800, "sev": 0.50, "desc": "Moderate spalling on inner raceway modulated by shaft rotation frequency."},
        {"class": 1, "name": "Severe Ball Element Damage (BSF)", "rpm": 2400, "sev": 0.85, "desc": "Advanced rolling element spalling causing severe impaction and elevated kurtosis."},
        {"class": 1, "name": "Cage Wear Degradation (FTF)", "rpm": 1200, "sev": 0.40, "desc": "Fundamental train frequency chatter from cage wear and loose ball retainer."},
        {"class": 1, "name": "High-Speed Severe Bearing Impact", "rpm": 3600, "sev": 0.90, "desc": "Critical bearing failure warning! High energy transient impact spikes."},
        {"class": 1, "name": "Lack of Lubrication Bearing Wear", "rpm": 1500, "sev": 0.35, "desc": "Dry friction metal contact inducing resonant high-frequency ringing."},
        {"class": 1, "name": "Contaminated Grease Abrasion", "rpm": 2100, "sev": 0.60, "desc": "Particle contamination creating random impulsive transient bursts."},
        {"class": 1, "name": "Combined BPFO + BPFI Degradation", "rpm": 1800, "sev": 0.75, "desc": "Dual race degradation with multiple sideband harmonic interactions."},
        {"class": 1, "name": "Corrosion Pitting Bearing Defect", "rpm": 3000, "sev": 0.45, "desc": "Surface oxidation pits causing high crest factor vibration spikes."},
        {"class": 1, "name": "Catastrophic Bearing Breakdown", "rpm": 1800, "sev": 0.95, "desc": "Immediate shutdown required! Severe mechanical grinding and breakdown."},

        # Rotor Imbalance samples (10)
        {"class": 2, "name": "Slight Mass Unbalance (1x)", "rpm": 1800, "sev": 0.20, "desc": "Minor mass eccentricity producing subtle 1× rotational speed sine wave."},
        {"class": 2, "name": "Moderate Static Rotor Imbalance", "rpm": 1800, "sev": 0.55, "desc": "Clear 1× peak in radial vibration spectrum due to off-center rotor center of mass."},
        {"class": 2, "name": "Severe Dynamic Rotor Imbalance", "rpm": 3000, "sev": 0.80, "desc": "Dominant 1× fundamental frequency proportional to RPM squared."},
        {"class": 2, "name": "Fan Blade Build-up Unbalance", "rpm": 1500, "sev": 0.35, "desc": "Debris accumulation on fan blades creating asymmetric centrifugal force."},
        {"class": 2, "name": "Low-Speed Imbalance Oscillation", "rpm": 1200, "sev": 0.65, "desc": "Low frequency high displacement centrifugal wobble."},
        {"class": 2, "name": "Overhung Rotor Imbalance", "rpm": 2400, "sev": 0.70, "desc": "Coupled axial and radial 1× component from overhung pulley offset."},
        {"class": 2, "name": "Flywheel Mass Asymmetry", "rpm": 3600, "sev": 0.85, "desc": "Extreme 1× radial sinusoidal force threatening housing structural integrity."},
        {"class": 2, "name": "Thermal Bow Imbalance", "rpm": 2100, "sev": 0.45, "desc": "Rotor shaft thermal distortion creating speed-dependent 1× eccentricity."},
        {"class": 2, "name": "Missing Rotor Balance Weight", "rpm": 1800, "sev": 0.75, "desc": "Detached balance weight causing sudden step increase in 1× vibration."},
        {"class": 2, "name": "Resonant Speed Imbalance Run", "rpm": 2700, "sev": 0.90, "desc": "Operation near critical structural resonance amplifying rotor imbalance."},

        # Shaft Misalignment samples (10)
        {"class": 3, "name": "Mild Parallel Coupling Offset", "rpm": 1800, "sev": 0.25, "desc": "Slight parallel offset generating noticeable 2× shaft speed harmonic."},
        {"class": 3, "name": "Moderate Angular Shaft Misalignment", "rpm": 1800, "sev": 0.50, "desc": "Angular misalignment producing strong 1× and 2× axial vibration energy."},
        {"class": 3, "name": "Severe Combination Misalignment", "rpm": 2400, "sev": 0.85, "desc": "Combined angular and parallel misalignment with strong 2× and 3× harmonics."},
        {"class": 3, "name": "Flexible Coupling Wear Misalignment", "rpm": 1500, "sev": 0.40, "desc": "Worn elastomer insert causing non-linear 2× directional forces."},
        {"class": 3, "name": "Thermal Growth Shaft Displacement", "rpm": 3000, "sev": 0.65, "desc": "Differential thermal expansion shifting motor shaft alignment out of tolerance."},
        {"class": 3, "name": "Soft Foot Foundation Strain", "rpm": 1200, "sev": 0.55, "desc": "Frame twisting from unshimmed soft foot footings generating 2× harmonics."},
        {"class": 3, "name": "High-Speed Axial Thrust Misalignment", "rpm": 3600, "sev": 0.75, "desc": "Strong Z-axis axial vibration phase shifted 180 degrees across coupling."},
        {"class": 3, "name": "Bent Shaft Distortion", "rpm": 2100, "sev": 0.80, "desc": "Permanently bowed shaft inducing strong 1× and 2× combined vibration."},
        {"class": 3, "name": "Gearbox Alignment Distortion", "rpm": 1800, "sev": 0.60, "desc": "Shaft misalignment across gear reducer interface with sideband harmonics."},
        {"class": 3, "name": "Loose Shaft Coupling Chatter", "rpm": 2700, "sev": 0.90, "desc": "Severe misalignment combined with mechanical looseness generating rich harmonics."},

        # Electrical Fault samples (10)
        {"class": 4, "name": "Stator Winding Inter-turn Short", "rpm": 1800, "sev": 0.30, "desc": "Mild stator asymmetry generating 2× line frequency (100 Hz) electromagnetic vibration."},
        {"class": 4, "name": "Phase Voltage Unbalance Fault", "rpm": 1800, "sev": 0.50, "desc": "Unbalanced supply voltage creating pulsed magnetic radial pull."},
        {"class": 4, "name": "Broken Rotor Bar Degradation", "rpm": 1500, "sev": 0.75, "desc": "Broken rotor bar producing sidebands at twice slip frequency around 1× shaft speed."},
        {"class": 4, "name": "Air Gap Eccentricity Fault", "rpm": 2400, "sev": 0.65, "desc": "Unequal magnetic gap inducing static and dynamic electromagnetic pull harmonics."},
        {"class": 4, "name": "Loose Stator Laminations Chatter", "rpm": 3000, "sev": 0.40, "desc": "High frequency magnetic hum from loose stator iron core laminations."},
        {"class": 4, "name": "VFD Switching Harmonic Resonance", "rpm": 2100, "sev": 0.55, "desc": "Carrier frequency harmonics from inverter drive PWM distortion."},
        {"class": 4, "name": "Multiple Broken Rotor Bars", "rpm": 1200, "sev": 0.90, "desc": "Severe rotor bar failure causing heavy torque pulsation and sideband modulation."},
        {"class": 4, "name": "Unbalanced Magnetic Pull (UMP)", "rpm": 3600, "sev": 0.70, "desc": "Strong 2× line frequency vibration modulated by rotational speed."},
        {"class": 4, "name": "Shaft Current Bearing Fluting", "rpm": 1800, "sev": 0.60, "desc": "Electrical discharge pitting across bearing raceway surface."},
        {"class": 4, "name": "Rotor Asymmetry Heat Stress", "rpm": 2700, "sev": 0.80, "desc": "Thermally induced rotor bar high resistance creating heavy 2× slip modulation."}
    ]

    presets = []

    for idx, cfg in enumerate(fault_configs, start=1):
        f_class = cfg["class"]
        rpm = cfg["rpm"]
        sev = cfg["sev"]
        
        # Generate physical sample (1024 points @ 12000Hz)
        data, _ = sim.generate_sample(
            fault_type=f_class,
            duration=1024 / 12000.0,
            fs=12000.0,
            rpm=rpm,
            severity=sev if f_class != 0 else None
        )
        
        # Extract features
        feats = extractor.extract_from_segment(data)
        feats = np.nan_to_num(feats, nan=0.0)
        
        # Downsample waveform for fast JSON transmission (200 points)
        step = max(1, len(data) // 200)
        x_wave = data[::step, 0].tolist()[:200]
        y_wave = data[::step, 1].tolist()[:200]
        z_wave = data[::step, 2].tolist()[:200]
        
        # Compute FFT on X axis
        freqs, mags = compute_fft(data[:, 0], fs=400, n_bins=64)
        
        # Calculate RMS for each axis
        rms_x = float(np.sqrt(np.mean(data[:, 0]**2)))
        rms_y = float(np.sqrt(np.mean(data[:, 1]**2)))
        rms_z = float(np.sqrt(np.mean(data[:, 2]**2)))
        
        # Health score calculation
        health = 0.96 - sev * 0.75 if f_class != 0 else 0.95 + np.random.uniform(0.01, 0.04)
        health = float(np.clip(health, 0.1, 1.0))
        
        # Probabilities simulation
        probs = [0.02] * 5
        if f_class == 0:
            probs[0] = 0.92
        else:
            probs[f_class] = 0.65 + sev * 0.30
            probs[0] = max(0.02, 1.0 - probs[f_class])
        sum_p = sum(probs)
        probs = [float(p / sum_p) for p in probs]
        
        presets.append({
            "id": f"P-{idx:02d}",
            "title": cfg["name"],
            "faultClass": f_class,
            "faultClassName": MotorVibrationSimulator.FAULT_NAMES[f_class],
            "severity": sev,
            "rpm": rpm,
            "healthScore": health,
            "description": cfg["desc"],
            "waveform": {"x": x_wave, "y": y_wave, "z": z_wave},
            "fft": {"freqs": [f"{f:.0f}" for f in freqs], "magnitudes": mags},
            "rms": {"x": rms_x, "y": rms_y, "z": rms_z},
            "features": {
                "rms": rms_x,
                "peak": float(np.max(np.abs(data[:, 0]))),
                "kurtosis": float(feats[3]),
                "crest": float(np.max(np.abs(data[:, 0])) / (rms_x + 1e-6)),
                "energy": float(np.mean(data[:, 0]**2)),
                "zcr": float(np.sum(np.diff(np.signbit(data[:, 0]))) / len(data)),
                "shape": float(rms_x / (np.mean(np.abs(data[:, 0])) + 1e-6)),
                "skewness": float(feats[4])
            },
            "probabilities": probs,
            "confidence": max(probs)
        })

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)

    print(f"[OK] Generated {len(presets)} preset samples in {output_file}")

if __name__ == "__main__":
    generate_preset_library()
