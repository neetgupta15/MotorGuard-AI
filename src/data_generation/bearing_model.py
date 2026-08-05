"""
Bearing Model for Fault Frequency Calculation and Defect Simulation
==================================================================

This module implements physics-based bearing fault models that calculate
characteristic defect frequencies (BPFO, BPFI, BSF, FTF) and generate
realistic impulse trains for simulating bearing faults in vibration data.

Theory:
    When a localized defect exists on a bearing surface, it generates periodic
    impulses as rolling elements strike the defect. The frequency of these
    impulses depends on the bearing geometry and shaft rotational speed.

Reference:
    - Harris, T.A. "Rolling Bearing Analysis", 5th Edition
    - SKF Group, "Bearing Damage and Failure Analysis"
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BearingGeometry:
    """
    Defines the physical geometry of a rolling element bearing.

    Attributes:
        n_balls:        Number of rolling elements (balls or rollers)
        ball_diameter:  Diameter of rolling element [mm]
        pitch_diameter: Pitch circle diameter [mm]
        contact_angle:  Contact angle [degrees]
    """
    n_balls: int = 9
    ball_diameter: float = 7.94       # mm (typical 6205 bearing)
    pitch_diameter: float = 38.5      # mm
    contact_angle: float = 0.0        # degrees (deep groove = 0)


class BearingModel:
    """
    Physics-based bearing fault model.

    Calculates characteristic defect frequencies and generates synthetic
    vibration signals for inner race, outer race, ball, and cage faults.

    Example:
        >>> bearing = BearingModel(BearingGeometry(n_balls=9))
        >>> freqs = bearing.calculate_fault_frequencies(shaft_rpm=1800)
        >>> print(f"BPFO = {freqs['BPFO']:.2f} Hz")
    """

    def __init__(self, geometry: BearingGeometry = None):
        """
        Initialize bearing model with specified geometry.

        Args:
            geometry: BearingGeometry dataclass. Uses default 6205 bearing if None.
        """
        self.geometry = geometry or BearingGeometry()

    def calculate_fault_frequencies(self, shaft_rpm: float) -> dict:
        """
        Calculate the four characteristic bearing defect frequencies.

        Args:
            shaft_rpm: Shaft rotational speed in RPM

        Returns:
            dict with keys: 'BPFO', 'BPFI', 'BSF', 'FTF', 'shaft_freq'

        Formulas:
            BPFO = (Z/2) × fr × (1 - d/Dp × cos(φ))
            BPFI = (Z/2) × fr × (1 + d/Dp × cos(φ))
            BSF  = (Dp/2d) × fr × (1 - (d/Dp × cos(φ))²)
            FTF  = (1/2) × fr × (1 - d/Dp × cos(φ))
        """
        g = self.geometry
        fr = shaft_rpm / 60.0  # Convert RPM to Hz
        phi_rad = np.radians(g.contact_angle)
        ratio = g.ball_diameter / g.pitch_diameter

        bpfo = (g.n_balls / 2.0) * fr * (1.0 - ratio * np.cos(phi_rad))
        bpfi = (g.n_balls / 2.0) * fr * (1.0 + ratio * np.cos(phi_rad))
        bsf = (g.pitch_diameter / (2.0 * g.ball_diameter)) * fr * \
              (1.0 - (ratio * np.cos(phi_rad)) ** 2)
        ftf = 0.5 * fr * (1.0 - ratio * np.cos(phi_rad))

        return {
            'BPFO': bpfo,
            'BPFI': bpfi,
            'BSF': bsf,
            'FTF': ftf,
            'shaft_freq': fr
        }

    def generate_outer_race_fault(self, duration: float, fs: float,
                                   shaft_rpm: float, severity: float = 0.5,
                                   resonance_freq: float = 3000.0) -> np.ndarray:
        """
        Generate vibration signal for outer race bearing fault.

        Outer race faults produce impulses at BPFO frequency. The impulse
        excites the bearing's natural frequency, producing amplitude-modulated
        high-frequency bursts.

        Args:
            duration:       Signal duration in seconds
            fs:             Sampling frequency in Hz
            shaft_rpm:      Shaft speed in RPM
            severity:       Fault severity (0.0 = incipient, 1.0 = severe)
            resonance_freq: Bearing resonance frequency in Hz

        Returns:
            np.ndarray of vibration signal
        """
        freqs = self.calculate_fault_frequencies(shaft_rpm)
        t = np.arange(0, duration, 1.0 / fs)
        bpfo = freqs['BPFO']

        # Generate impulse train at BPFO frequency
        impulse_period = 1.0 / bpfo
        impulses = np.zeros_like(t)

        for i in range(int(duration * bpfo) + 1):
            impulse_time = i * impulse_period
            # Each impulse is a decaying exponential burst
            mask = (t >= impulse_time) & (t < impulse_time + 0.005)
            t_local = t[mask] - impulse_time
            decay_rate = 800.0 + 400.0 * severity
            amplitude = severity * (0.5 + 0.5 * np.random.random())
            impulses[mask] += amplitude * np.exp(-decay_rate * t_local) * \
                             np.sin(2.0 * np.pi * resonance_freq * t_local)

        # Add slight randomness to simulate real-world jitter
        jitter = np.random.normal(0, 0.002 * severity, impulses.shape)
        impulses += jitter * impulses

        return impulses

    def generate_inner_race_fault(self, duration: float, fs: float,
                                   shaft_rpm: float, severity: float = 0.5,
                                   resonance_freq: float = 3500.0) -> np.ndarray:
        """
        Generate vibration signal for inner race bearing fault.

        Inner race faults produce impulses at BPFI frequency, amplitude-
        modulated at the shaft frequency because the defect rotates with
        the shaft (moving in and out of the load zone).

        Args:
            duration:       Signal duration in seconds
            fs:             Sampling frequency in Hz
            shaft_rpm:      Shaft speed in RPM
            severity:       Fault severity (0.0 = incipient, 1.0 = severe)
            resonance_freq: Bearing resonance frequency in Hz

        Returns:
            np.ndarray of vibration signal
        """
        freqs = self.calculate_fault_frequencies(shaft_rpm)
        t = np.arange(0, duration, 1.0 / fs)
        bpfi = freqs['BPFI']
        fr = freqs['shaft_freq']

        # Generate impulse train at BPFI frequency
        impulse_period = 1.0 / bpfi
        impulses = np.zeros_like(t)

        for i in range(int(duration * bpfi) + 1):
            impulse_time = i * impulse_period
            mask = (t >= impulse_time) & (t < impulse_time + 0.004)
            t_local = t[mask] - impulse_time
            decay_rate = 900.0 + 500.0 * severity
            amplitude = severity * (0.5 + 0.5 * np.random.random())
            impulses[mask] += amplitude * np.exp(-decay_rate * t_local) * \
                             np.sin(2.0 * np.pi * resonance_freq * t_local)

        # Amplitude modulation at shaft frequency (load zone effect)
        modulation = 1.0 + 0.6 * severity * np.cos(2.0 * np.pi * fr * t)
        impulses *= modulation

        return impulses

    def generate_ball_fault(self, duration: float, fs: float,
                            shaft_rpm: float, severity: float = 0.5,
                            resonance_freq: float = 4000.0) -> np.ndarray:
        """
        Generate vibration signal for rolling element (ball) fault.

        Ball faults produce impulses at 2× BSF (ball contacts both races
        per revolution). The signal is modulated at the cage frequency (FTF).

        Args:
            duration:       Signal duration in seconds
            fs:             Sampling frequency in Hz
            shaft_rpm:      Shaft speed in RPM
            severity:       Fault severity (0.0 = incipient, 1.0 = severe)
            resonance_freq: Bearing resonance frequency in Hz

        Returns:
            np.ndarray of vibration signal
        """
        freqs = self.calculate_fault_frequencies(shaft_rpm)
        t = np.arange(0, duration, 1.0 / fs)
        bsf2 = 2.0 * freqs['BSF']  # 2× BSF
        ftf = freqs['FTF']

        # Generate impulse train at 2×BSF
        impulse_period = 1.0 / bsf2
        impulses = np.zeros_like(t)

        for i in range(int(duration * bsf2) + 1):
            impulse_time = i * impulse_period
            mask = (t >= impulse_time) & (t < impulse_time + 0.003)
            t_local = t[mask] - impulse_time
            decay_rate = 1000.0 + 500.0 * severity
            amplitude = severity * 0.7 * (0.5 + 0.5 * np.random.random())
            impulses[mask] += amplitude * np.exp(-decay_rate * t_local) * \
                             np.sin(2.0 * np.pi * resonance_freq * t_local)

        # Modulate at cage frequency
        modulation = 1.0 + 0.5 * severity * np.cos(2.0 * np.pi * ftf * t)
        impulses *= modulation

        return impulses


if __name__ == "__main__":
    # Quick test and demonstration
    bearing = BearingModel()
    freqs = bearing.calculate_fault_frequencies(shaft_rpm=1800)

    print("=" * 60)
    print("Bearing Fault Frequency Analysis")
    print(f"Bearing: 6205 equivalent ({bearing.geometry.n_balls} balls)")
    print(f"Shaft Speed: 1800 RPM ({freqs['shaft_freq']:.1f} Hz)")
    print("=" * 60)
    print(f"  BPFO (Outer Race) : {freqs['BPFO']:.2f} Hz")
    print(f"  BPFI (Inner Race) : {freqs['BPFI']:.2f} Hz")
    print(f"  BSF  (Ball Spin)  : {freqs['BSF']:.2f} Hz")
    print(f"  FTF  (Cage)       : {freqs['FTF']:.2f} Hz")
    print("=" * 60)

    # Generate sample fault signals
    fs = 12000  # 12 kHz sampling
    duration = 0.5
    for fault_name, gen_func in [
        ("Outer Race", bearing.generate_outer_race_fault),
        ("Inner Race", bearing.generate_inner_race_fault),
        ("Ball", bearing.generate_ball_fault)
    ]:
        sig = gen_func(duration, fs, 1800, severity=0.7)
        print(f"  {fault_name} fault signal: {len(sig)} samples, "
              f"RMS={np.sqrt(np.mean(sig**2)):.4f}")
