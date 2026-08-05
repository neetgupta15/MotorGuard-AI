"""
Electric Motor Vibration Simulator
===================================

Physics-based simulation of electric motor vibrations under normal and
faulty operating conditions. Generates realistic 3-axis accelerometer
data for training fault detection machine learning models.

Fault Types Modeled:
    1. Normal Operation — baseline vibration at shaft harmonics
    2. Bearing Faults — impulse trains at characteristic bearing frequencies
    3. Rotor Imbalance — dominant 1× vibration proportional to ω²
    4. Shaft Misalignment — strong 2× component + axial vibration
    5. Electrical Faults — 2× line frequency sidebands + slot harmonics

Motor Model:
    A generic DC/BLDC motor with configurable parameters including
    shaft speed, number of poles, stator slots, bearing geometry,
    and mechanical coupling characteristics.

Reference:
    - Randall, R.B. "Vibration-based Condition Monitoring"
    - IEC 60034-14: Vibration standards for rotating machines
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from .bearing_model import BearingModel, BearingGeometry


@dataclass
class MotorParameters:
    """
    Physical parameters of an electric motor.

    Attributes:
        nominal_rpm:    Rated shaft speed [RPM]
        num_poles:      Number of magnetic poles
        num_slots:      Number of stator slots
        rotor_mass:     Rotor mass [kg]
        shaft_length:   Shaft length [mm]
        line_freq:      Supply line frequency [Hz] (50 or 60)
        bearing:        Bearing geometry specification
        base_vibration: Baseline vibration amplitude [g]
    """
    nominal_rpm: float = 1800.0
    num_poles: int = 4
    num_slots: int = 24
    rotor_mass: float = 2.5           # kg
    shaft_length: float = 200.0       # mm
    line_freq: float = 50.0           # Hz (mains frequency)
    bearing: BearingGeometry = field(default_factory=BearingGeometry)
    base_vibration: float = 0.05      # g (baseline amplitude)


class MotorVibrationSimulator:
    """
    Generates synthetic 3-axis vibration data for electric motors.

    This simulator creates realistic accelerometer signals by combining:
    - Shaft rotation harmonics (1×, 2×, 3× shaft frequency)
    - Bearing characteristic vibrations
    - Structural resonances
    - Background noise (Gaussian + colored)
    - Fault-specific signatures

    The output mimics data from a triaxial accelerometer (e.g., LIS3DH)
    mounted on the motor housing.

    Usage:
        >>> sim = MotorVibrationSimulator()
        >>> data = sim.generate_normal(duration=1.0, fs=12000)
        >>> print(data.shape)  # (12000, 3) for X, Y, Z axes
    """

    # Fault class labels
    FAULT_NORMAL = 0
    FAULT_BEARING = 1
    FAULT_IMBALANCE = 2
    FAULT_MISALIGNMENT = 3
    FAULT_ELECTRICAL = 4

    FAULT_NAMES = {
        0: "Normal",
        1: "Bearing Fault",
        2: "Rotor Imbalance",
        3: "Shaft Misalignment",
        4: "Electrical Fault"
    }

    def __init__(self, params: MotorParameters = None, seed: int = None):
        """
        Initialize the motor vibration simulator.

        Args:
            params: Motor physical parameters. Uses defaults if None.
            seed:   Random seed for reproducibility. None for random.
        """
        self.params = params or MotorParameters()
        self.bearing_model = BearingModel(self.params.bearing)
        self.rng = np.random.RandomState(seed)

    def _shaft_frequency(self, rpm: float = None) -> float:
        """Get shaft rotational frequency in Hz."""
        rpm = rpm or self.params.nominal_rpm
        return rpm / 60.0

    def _generate_base_vibration(self, t: np.ndarray, rpm: float = None,
                                  variation: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate baseline vibration present in all motor conditions.

        Includes:
        - 1×, 2×, 3× shaft frequency harmonics (radial)
        - Residual imbalance (small 1× component)
        - Background noise floor

        Args:
            t:          Time vector
            rpm:        Shaft speed in RPM
            variation:  Random variation factor for natural variability

        Returns:
            Tuple of (x_vibration, y_vibration, z_vibration)
        """
        f_shaft = self._shaft_frequency(rpm)
        base = self.params.base_vibration

        # --- Radial vibrations (X and Y axes) ---
        # 1× shaft frequency — always present due to residual imbalance
        amp_1x = base * (1.0 + variation * self.rng.randn())
        phase_1x = self.rng.uniform(0, 2 * np.pi)
        vib_1x = amp_1x * np.sin(2 * np.pi * f_shaft * t + phase_1x)

        # 2× shaft frequency — coupling/alignment residuals
        amp_2x = base * 0.3 * (1.0 + variation * self.rng.randn())
        phase_2x = self.rng.uniform(0, 2 * np.pi)
        vib_2x = amp_2x * np.sin(2 * np.pi * 2 * f_shaft * t + phase_2x)

        # 3× shaft frequency — very small
        amp_3x = base * 0.1 * (1.0 + variation * self.rng.randn())
        phase_3x = self.rng.uniform(0, 2 * np.pi)
        vib_3x = amp_3x * np.sin(2 * np.pi * 3 * f_shaft * t + phase_3x)

        # Combine radial components
        radial = vib_1x + vib_2x + vib_3x

        # X and Y are radial, with 90° phase difference
        x_vib = radial
        y_vib = amp_1x * np.sin(2 * np.pi * f_shaft * t + phase_1x + np.pi / 2) + \
                amp_2x * 0.8 * np.sin(2 * np.pi * 2 * f_shaft * t + phase_2x + np.pi / 4)

        # Z axis (axial) — typically much smaller for healthy motor
        z_vib = base * 0.15 * np.sin(2 * np.pi * f_shaft * t + self.rng.uniform(0, 2 * np.pi))

        # Add Gaussian noise floor
        noise_level = base * 0.2
        x_vib += self.rng.normal(0, noise_level, len(t))
        y_vib += self.rng.normal(0, noise_level, len(t))
        z_vib += self.rng.normal(0, noise_level * 0.5, len(t))

        return x_vib, y_vib, z_vib

    def generate_normal(self, duration: float = 1.0, fs: float = 12000.0,
                        rpm: float = None) -> np.ndarray:
        """
        Generate vibration data for a healthy motor.

        Args:
            duration: Signal duration in seconds
            fs:       Sampling frequency in Hz
            rpm:      Shaft speed (uses nominal if None)

        Returns:
            np.ndarray of shape (N, 3) — [X, Y, Z] accelerometer data in g
        """
        rpm = rpm or self.params.nominal_rpm
        t = np.arange(0, duration, 1.0 / fs)
        x, y, z = self._generate_base_vibration(t, rpm)

        # Add very slight colored noise (low-frequency drift)
        drift_freq = self.rng.uniform(0.5, 3.0)
        drift_amp = self.params.base_vibration * 0.05
        x += drift_amp * np.sin(2 * np.pi * drift_freq * t)
        y += drift_amp * np.sin(2 * np.pi * drift_freq * t + np.pi / 3)

        return np.column_stack([x, y, z])

    def generate_bearing_fault(self, duration: float = 1.0, fs: float = 12000.0,
                                rpm: float = None, severity: float = 0.5,
                                fault_location: str = 'outer') -> np.ndarray:
        """
        Generate vibration data with bearing fault.

        Bearing faults add characteristic impulse trains at defect frequencies
        superimposed on the baseline vibration.

        Args:
            duration:       Signal duration in seconds
            fs:             Sampling frequency in Hz
            rpm:            Shaft speed in RPM
            severity:       Fault severity [0.0 - 1.0]
            fault_location: 'outer', 'inner', or 'ball'

        Returns:
            np.ndarray of shape (N, 3) — [X, Y, Z] accelerometer data in g
        """
        rpm = rpm or self.params.nominal_rpm
        t = np.arange(0, duration, 1.0 / fs)

        # Start with baseline vibration
        x, y, z = self._generate_base_vibration(t, rpm)

        # Add bearing fault signature
        if fault_location == 'outer':
            fault_sig = self.bearing_model.generate_outer_race_fault(
                duration, fs, rpm, severity)
        elif fault_location == 'inner':
            fault_sig = self.bearing_model.generate_inner_race_fault(
                duration, fs, rpm, severity)
        elif fault_location == 'ball':
            fault_sig = self.bearing_model.generate_ball_fault(
                duration, fs, rpm, severity)
        else:
            raise ValueError(f"Unknown fault location: {fault_location}")

        # Bearing faults primarily affect radial directions
        x += fault_sig * (0.8 + 0.2 * self.rng.random())
        y += fault_sig * (0.6 + 0.2 * self.rng.random())
        # Slight axial component
        z += fault_sig * (0.15 + 0.1 * self.rng.random())

        # Increase overall noise slightly (bearing damage creates broadband noise)
        broadband = self.rng.normal(0, 0.02 * severity, len(t))
        x += broadband
        y += broadband * 0.8

        return np.column_stack([x, y, z])

    def generate_rotor_imbalance(self, duration: float = 1.0, fs: float = 12000.0,
                                  rpm: float = None, severity: float = 0.5) -> np.ndarray:
        """
        Generate vibration data with rotor imbalance fault.

        Rotor imbalance produces a dominant 1× shaft frequency component.
        The amplitude increases proportionally to ω² (speed squared).
        The vibration is primarily radial with ~90° phase difference
        between horizontal and vertical.

        Args:
            duration:   Signal duration in seconds
            fs:         Sampling frequency in Hz
            rpm:        Shaft speed in RPM
            severity:   Imbalance severity [0.0 - 1.0]
                       (0.1 = slight, 0.5 = moderate, 1.0 = severe)

        Returns:
            np.ndarray of shape (N, 3) — [X, Y, Z] accelerometer data in g
        """
        rpm = rpm or self.params.nominal_rpm
        t = np.arange(0, duration, 1.0 / fs)

        # Baseline vibration
        x, y, z = self._generate_base_vibration(t, rpm)

        f_shaft = self._shaft_frequency(rpm)
        omega = 2 * np.pi * f_shaft

        # Imbalance force ∝ m × e × ω²
        # Normalized imbalance amplitude
        imbalance_amp = severity * 0.8 * (omega / (2 * np.pi * 30)) ** 2
        imbalance_amp = min(imbalance_amp, severity * 2.0)  # Cap amplitude

        phase = self.rng.uniform(0, 2 * np.pi)

        # Strong 1× component in radial directions
        x += imbalance_amp * np.sin(2 * np.pi * f_shaft * t + phase)
        y += imbalance_amp * np.sin(2 * np.pi * f_shaft * t + phase + np.pi / 2)
        # Axial component is very small for pure imbalance
        z += imbalance_amp * 0.05 * np.sin(2 * np.pi * f_shaft * t + phase)

        # Slight increase in harmonics due to nonlinear bearing response
        x += imbalance_amp * 0.1 * np.sin(2 * np.pi * 2 * f_shaft * t)

        return np.column_stack([x, y, z])

    def generate_shaft_misalignment(self, duration: float = 1.0, fs: float = 12000.0,
                                     rpm: float = None, severity: float = 0.5,
                                     misalign_type: str = 'angular') -> np.ndarray:
        """
        Generate vibration data with shaft misalignment fault.

        Misalignment produces strong 2× shaft frequency components.
        - Angular misalignment: strong axial 1× and 2×
        - Parallel misalignment: strong radial 2×

        Args:
            duration:       Signal duration in seconds
            fs:             Sampling frequency in Hz
            rpm:            Shaft speed in RPM
            severity:       Misalignment severity [0.0 - 1.0]
            misalign_type:  'angular' or 'parallel'

        Returns:
            np.ndarray of shape (N, 3) — [X, Y, Z] accelerometer data in g
        """
        rpm = rpm or self.params.nominal_rpm
        t = np.arange(0, duration, 1.0 / fs)

        # Baseline vibration
        x, y, z = self._generate_base_vibration(t, rpm)

        f_shaft = self._shaft_frequency(rpm)
        phase = self.rng.uniform(0, 2 * np.pi)

        if misalign_type == 'angular':
            # Angular misalignment — strong axial vibration at 1× and 2×
            misalign_amp = severity * 0.6

            # Strong axial 1× and 2×
            z += misalign_amp * 0.8 * np.sin(2 * np.pi * f_shaft * t + phase)
            z += misalign_amp * 1.0 * np.sin(2 * np.pi * 2 * f_shaft * t + phase)

            # Moderate radial 2×
            x += misalign_amp * 0.4 * np.sin(2 * np.pi * 2 * f_shaft * t + phase)
            y += misalign_amp * 0.3 * np.sin(2 * np.pi * 2 * f_shaft * t + phase + np.pi / 6)

            # Small 3× component
            z += misalign_amp * 0.3 * np.sin(2 * np.pi * 3 * f_shaft * t)

        elif misalign_type == 'parallel':
            # Parallel misalignment — strong radial 2× with phase opposition
            misalign_amp = severity * 0.7

            # Strong radial 2×
            x += misalign_amp * 1.0 * np.sin(2 * np.pi * 2 * f_shaft * t + phase)
            y += misalign_amp * 0.9 * np.sin(2 * np.pi * 2 * f_shaft * t + phase + np.pi)

            # Radial 1× also elevated
            x += misalign_amp * 0.5 * np.sin(2 * np.pi * f_shaft * t + phase)

            # Moderate axial
            z += misalign_amp * 0.3 * np.sin(2 * np.pi * 2 * f_shaft * t)

            # Higher harmonics (3×, 4×)
            x += misalign_amp * 0.2 * np.sin(2 * np.pi * 3 * f_shaft * t)
            x += misalign_amp * 0.1 * np.sin(2 * np.pi * 4 * f_shaft * t)

        return np.column_stack([x, y, z])

    def generate_electrical_fault(self, duration: float = 1.0, fs: float = 12000.0,
                                   rpm: float = None, severity: float = 0.5,
                                   fault_type: str = 'stator') -> np.ndarray:
        """
        Generate vibration data with electrical fault.

        Electrical faults produce vibrations at:
        - 2× line frequency (stator faults, supply issues)
        - Rotor bar pass frequency and sidebands
        - Stator slot pass frequency

        Args:
            duration:   Signal duration in seconds
            fs:         Sampling frequency in Hz
            rpm:        Shaft speed in RPM
            severity:   Fault severity [0.0 - 1.0]
            fault_type: 'stator' (winding fault) or 'rotor' (broken bar)

        Returns:
            np.ndarray of shape (N, 3) — [X, Y, Z] accelerometer data in g
        """
        rpm = rpm or self.params.nominal_rpm
        t = np.arange(0, duration, 1.0 / fs)

        # Baseline vibration
        x, y, z = self._generate_base_vibration(t, rpm)

        f_shaft = self._shaft_frequency(rpm)
        f_line = self.params.line_freq
        f_2line = 2.0 * f_line  # 100 Hz (for 50 Hz supply)

        if fault_type == 'stator':
            # Stator winding faults — dominant 2× line frequency
            elec_amp = severity * 0.5

            # Strong 2× line frequency vibration
            x += elec_amp * np.sin(2 * np.pi * f_2line * t)
            y += elec_amp * 0.8 * np.sin(2 * np.pi * f_2line * t + np.pi / 4)

            # Stator slot passing frequency = num_slots × f_shaft
            f_slot = self.params.num_slots * f_shaft
            x += elec_amp * 0.3 * np.sin(2 * np.pi * f_slot * t)

            # Sidebands around 2× line frequency
            x += elec_amp * 0.2 * np.sin(2 * np.pi * (f_2line + f_shaft) * t)
            x += elec_amp * 0.2 * np.sin(2 * np.pi * (f_2line - f_shaft) * t)

        elif fault_type == 'rotor':
            # Broken rotor bar — slip frequency sidebands
            # Slip frequency
            sync_speed = 120 * f_line / self.params.num_poles
            slip = (sync_speed - rpm) / sync_speed
            f_slip = slip * f_line

            elec_amp = severity * 0.4

            # Sidebands around 1× at ±2×f_slip
            x += elec_amp * np.sin(2 * np.pi * (f_shaft + 2 * f_slip) * t)
            x += elec_amp * np.sin(2 * np.pi * (f_shaft - 2 * f_slip) * t)

            # 2× line frequency component
            x += elec_amp * 0.6 * np.sin(2 * np.pi * f_2line * t)
            y += elec_amp * 0.5 * np.sin(2 * np.pi * f_2line * t)

            # Rotor slot harmonics
            for k in range(1, 3):
                x += elec_amp * 0.15 / k * np.sin(
                    2 * np.pi * k * self.params.num_slots * f_shaft * t)

        return np.column_stack([x, y, z])

    def generate_sample(self, fault_type: int, duration: float = 1.0,
                        fs: float = 12000.0, rpm: float = None,
                        severity: float = None) -> Tuple[np.ndarray, dict]:
        """
        Generate a single vibration sample with metadata.

        This is the main entry point for dataset generation.

        Args:
            fault_type: One of FAULT_NORMAL(0), FAULT_BEARING(1),
                       FAULT_IMBALANCE(2), FAULT_MISALIGNMENT(3),
                       FAULT_ELECTRICAL(4)
            duration:   Signal duration in seconds
            fs:         Sampling frequency in Hz
            rpm:        Shaft speed (randomized if None)
            severity:   Fault severity (randomized if None)

        Returns:
            Tuple of (data, metadata) where:
                data: np.ndarray of shape (N, 3) — [X, Y, Z]
                metadata: dict with generation parameters
        """
        # Randomize RPM if not specified (±15% around nominal)
        if rpm is None:
            rpm = self.params.nominal_rpm * (1.0 + 0.15 * self.rng.uniform(-1, 1))

        # Randomize severity if not specified
        if severity is None and fault_type != self.FAULT_NORMAL:
            severity = self.rng.uniform(0.2, 1.0)

        metadata = {
            'fault_type': fault_type,
            'fault_name': self.FAULT_NAMES[fault_type],
            'rpm': rpm,
            'severity': severity if fault_type != self.FAULT_NORMAL else 0.0,
            'duration': duration,
            'fs': fs,
            'n_samples': int(duration * fs)
        }

        # Generate based on fault type
        if fault_type == self.FAULT_NORMAL:
            data = self.generate_normal(duration, fs, rpm)

        elif fault_type == self.FAULT_BEARING:
            # Randomly pick bearing fault location
            location = self.rng.choice(['outer', 'inner', 'ball'])
            metadata['sub_type'] = f'bearing_{location}'
            data = self.generate_bearing_fault(duration, fs, rpm, severity, location)

        elif fault_type == self.FAULT_IMBALANCE:
            metadata['sub_type'] = 'rotor_imbalance'
            data = self.generate_rotor_imbalance(duration, fs, rpm, severity)

        elif fault_type == self.FAULT_MISALIGNMENT:
            align_type = self.rng.choice(['angular', 'parallel'])
            metadata['sub_type'] = f'misalignment_{align_type}'
            data = self.generate_shaft_misalignment(duration, fs, rpm, severity, align_type)

        elif fault_type == self.FAULT_ELECTRICAL:
            elec_type = self.rng.choice(['stator', 'rotor'])
            metadata['sub_type'] = f'electrical_{elec_type}'
            data = self.generate_electrical_fault(duration, fs, rpm, severity, elec_type)

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        return data, metadata


if __name__ == "__main__":
    # Demonstration of the simulator
    print("=" * 70)
    print("  Electric Motor Vibration Simulator — Demo")
    print("=" * 70)

    sim = MotorVibrationSimulator(seed=42)

    for fault_id, fault_name in MotorVibrationSimulator.FAULT_NAMES.items():
        data, meta = sim.generate_sample(fault_id, duration=1.0, fs=12000)
        rms_x = np.sqrt(np.mean(data[:, 0] ** 2))
        rms_y = np.sqrt(np.mean(data[:, 1] ** 2))
        rms_z = np.sqrt(np.mean(data[:, 2] ** 2))
        print(f"\n  [{fault_id}] {fault_name}")
        print(f"      RPM: {meta['rpm']:.0f}, Severity: {meta['severity']:.2f}")
        print(f"      RMS — X: {rms_x:.4f}g  Y: {rms_y:.4f}g  Z: {rms_z:.4f}g")
        if 'sub_type' in meta:
            print(f"      Sub-type: {meta['sub_type']}")

    print("\n" + "=" * 70)
    print("  Simulation complete. All fault types generated successfully.")
    print("=" * 70)
