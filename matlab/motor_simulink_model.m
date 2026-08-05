%% =========================================================================
%  Motor Fault Detection — Simulink Motor Model Reference Script
%  =========================================================================
%  This MATLAB script demonstrates how to programmatically create a Simulink
%  model for electric motor vibration simulation with fault injection.
%
%  Requirements:
%    - MATLAB R2022a or later
%    - Simulink
%    - Simscape / Simscape Electrical
%    - Simscape Multibody (optional, for advanced vibration)
%
%  Usage:
%    >> motor_simulink_model
%  =========================================================================

%% Clear workspace
clear; clc; close all;

fprintf('==========================================================\n');
fprintf('  Motor Fault Detection — Simulink Model Generator\n');
fprintf('==========================================================\n\n');

%% ─── Motor Parameters ─────────────────────────────────────────────────
% DC Motor specifications (typical household motor)
motor_params = struct();
motor_params.nominal_voltage = 12;          % V
motor_params.armature_resistance = 1.0;     % Ohm
motor_params.armature_inductance = 0.5e-3;  % H
motor_params.back_emf_constant = 0.01;      % V/(rad/s)
motor_params.torque_constant = 0.01;        % N·m/A
motor_params.rotor_inertia = 5e-5;          % kg·m²
motor_params.viscous_damping = 1e-5;        % N·m·s/rad
motor_params.nominal_speed = 1800;          % RPM

%% ─── Bearing Parameters ───────────────────────────────────────────────
bearing_params = struct();
bearing_params.n_balls = 9;
bearing_params.ball_diameter = 7.94e-3;     % m
bearing_params.pitch_diameter = 38.5e-3;    % m
bearing_params.contact_angle = 0;           % degrees

% Calculate characteristic frequencies
fr = motor_params.nominal_speed / 60;       % Shaft frequency (Hz)
d_Dp = bearing_params.ball_diameter / bearing_params.pitch_diameter;

bearing_params.BPFO = (bearing_params.n_balls/2) * fr * (1 - d_Dp);
bearing_params.BPFI = (bearing_params.n_balls/2) * fr * (1 + d_Dp);
bearing_params.BSF  = (bearing_params.pitch_diameter/(2*bearing_params.ball_diameter)) * fr * (1 - d_Dp^2);
bearing_params.FTF  = 0.5 * fr * (1 - d_Dp);

fprintf('  Motor Parameters:\n');
fprintf('    Nominal Speed: %d RPM (%.1f Hz)\n', motor_params.nominal_speed, fr);
fprintf('    Voltage: %d V\n\n', motor_params.nominal_voltage);

fprintf('  Bearing Fault Frequencies:\n');
fprintf('    BPFO (Outer Race): %.2f Hz\n', bearing_params.BPFO);
fprintf('    BPFI (Inner Race): %.2f Hz\n', bearing_params.BPFI);
fprintf('    BSF  (Ball Spin):  %.2f Hz\n', bearing_params.BSF);
fprintf('    FTF  (Cage):       %.2f Hz\n\n', bearing_params.FTF);

%% ─── Simulation Parameters ───────────────────────────────────────────
sim_params = struct();
sim_params.duration = 2.0;        % seconds
sim_params.sample_rate = 12000;   % Hz
sim_params.dt = 1/sim_params.sample_rate;

t = 0:sim_params.dt:(sim_params.duration - sim_params.dt);
N = length(t);

fprintf('  Simulation:\n');
fprintf('    Duration: %.1f s\n', sim_params.duration);
fprintf('    Sample Rate: %d Hz\n', sim_params.sample_rate);
fprintf('    Total Samples: %d\n\n', N);

%% ─── Generate Vibration Signals ──────────────────────────────────────

fprintf('  Generating vibration signals...\n\n');

% ─── 1. Normal Operation ─────────────────────────────────────────────
normal_x = 0.05 * sin(2*pi*fr*t) + ...          % 1x shaft
           0.015 * sin(2*pi*2*fr*t + 0.3) + ...  % 2x
           0.005 * sin(2*pi*3*fr*t + 0.7) + ...  % 3x
           0.01 * randn(1, N);                    % noise

normal_y = 0.05 * sin(2*pi*fr*t + pi/2) + ...
           0.012 * sin(2*pi*2*fr*t + 0.8) + ...
           0.01 * randn(1, N);

normal_z = 0.008 * sin(2*pi*fr*t + 0.5) + ...
           0.005 * randn(1, N);

fprintf('    [1/5] Normal operation signal generated\n');

% ─── 2. Bearing Fault (Outer Race) ──────────────────────────────────
severity_bearing = 0.6;
bearing_impulse = zeros(1, N);
impulse_period = round(sim_params.sample_rate / bearing_params.BPFO);

for i = 1:impulse_period:N
    idx_end = min(i + 60, N);
    t_local = (0:(idx_end-i)) * sim_params.dt;
    bearing_impulse(i:idx_end) = severity_bearing * ...
        exp(-800*t_local) .* sin(2*pi*3000*t_local);
end

bearing_x = normal_x + bearing_impulse * 0.8 + 0.02*severity_bearing*randn(1,N);
bearing_y = normal_y + bearing_impulse * 0.6 + 0.015*severity_bearing*randn(1,N);
bearing_z = normal_z + bearing_impulse * 0.15;

fprintf('    [2/5] Bearing fault signal generated (BPFO = %.1f Hz)\n', bearing_params.BPFO);

% ─── 3. Rotor Imbalance ─────────────────────────────────────────────
severity_imbalance = 0.7;
omega = 2*pi*fr;
imbalance_amp = severity_imbalance * 0.6;

imbalance_x = normal_x + imbalance_amp * sin(2*pi*fr*t + 1.2);
imbalance_y = normal_y + imbalance_amp * sin(2*pi*fr*t + 1.2 + pi/2);
imbalance_z = normal_z + imbalance_amp * 0.05 * sin(2*pi*fr*t);

fprintf('    [3/5] Rotor imbalance signal generated\n');

% ─── 4. Shaft Misalignment ──────────────────────────────────────────
severity_misalign = 0.6;
misalign_amp = severity_misalign * 0.6;

misalign_x = normal_x + misalign_amp * 0.4 * sin(2*pi*2*fr*t + 0.5);
misalign_y = normal_y + misalign_amp * 0.3 * sin(2*pi*2*fr*t + 0.5 + pi/6);
misalign_z = normal_z + ...
             misalign_amp * 0.8 * sin(2*pi*fr*t + 0.5) + ...
             misalign_amp * 1.0 * sin(2*pi*2*fr*t + 0.5) + ...
             misalign_amp * 0.3 * sin(2*pi*3*fr*t);

fprintf('    [4/5] Shaft misalignment signal generated\n');

% ─── 5. Electrical Fault ────────────────────────────────────────────
severity_elec = 0.5;
f_line = 50;    % Mains frequency (Hz)
f_2line = 100;  % 2× line frequency

elec_amp = severity_elec * 0.5;
f_slot = 24 * fr;  % Stator slot frequency

elec_x = normal_x + elec_amp * sin(2*pi*f_2line*t) + ...
         elec_amp * 0.3 * sin(2*pi*f_slot*t) + ...
         elec_amp * 0.2 * sin(2*pi*(f_2line+fr)*t) + ...
         elec_amp * 0.2 * sin(2*pi*(f_2line-fr)*t);

elec_y = normal_y + elec_amp * 0.8 * sin(2*pi*f_2line*t + pi/4);
elec_z = normal_z;

fprintf('    [5/5] Electrical fault signal generated\n\n');

%% ─── Visualization ──────────────────────────────────────────────────

fprintf('  Generating plots...\n');

figure('Position', [100 100 1400 900], 'Color', 'w');

signals = {normal_x, bearing_x, imbalance_x, misalign_x, elec_x};
titles = {'Normal', 'Bearing Fault', 'Rotor Imbalance', ...
          'Shaft Misalignment', 'Electrical Fault'};

for i = 1:5
    % Time domain
    subplot(5, 2, 2*i-1);
    plot(t(1:1200), signals{i}(1:1200), 'b', 'LineWidth', 0.5);
    title([titles{i} ' — Time Domain'], 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Acceleration (g)');
    grid on; xlim([0 t(1200)]);

    % Frequency domain
    subplot(5, 2, 2*i);
    Y = abs(fft(signals{i})) / N;
    f_axis = (0:N-1) * sim_params.sample_rate / N;
    plot(f_axis(1:N/2), Y(1:N/2), 'r', 'LineWidth', 0.5);
    title([titles{i} ' — FFT Spectrum'], 'FontWeight', 'bold');
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    grid on; xlim([0 500]);
end

sgtitle('Motor Vibration Analysis — All Fault Scenarios', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'results/vibration_analysis.png');
fprintf('  Plot saved to: results/vibration_analysis.png\n');

%% ─── Export Data ────────────────────────────────────────────────────

fprintf('\n  Exporting data...\n');

% Save signals to MAT file
save('data/matlab_signals.mat', ...
    'normal_x', 'normal_y', 'normal_z', ...
    'bearing_x', 'bearing_y', 'bearing_z', ...
    'imbalance_x', 'imbalance_y', 'imbalance_z', ...
    'misalign_x', 'misalign_y', 'misalign_z', ...
    'elec_x', 'elec_y', 'elec_z', ...
    't', 'motor_params', 'bearing_params', 'sim_params');

fprintf('  Data saved to: data/matlab_signals.mat\n');

%% ─── Simulink Model Creation Notes ─────────────────────────────────
fprintf('\n==========================================================\n');
fprintf('  SIMULINK MODEL CREATION GUIDE\n');
fprintf('==========================================================\n');
fprintf('  To create the Simulink model:\n\n');
fprintf('  1. Open Simulink: >> simulink\n');
fprintf('  2. Create new model: >> new_system(''MotorFaultModel'')\n');
fprintf('  3. Add blocks:\n');
fprintf('     - Simscape Electrical > DC Motor\n');
fprintf('     - Simscape > Rotational Inertia (rotor)\n');
fprintf('     - Simscape > Ideal Rotational Motion Sensor\n');
fprintf('     - Signal Processing > Accelerometer (simulated)\n\n');
fprintf('  4. For bearing faults:\n');
fprintf('     - Use Simscape > Rotational Damper (Variable)\n');
fprintf('     - Modulate damping at BPFO frequency\n\n');
fprintf('  5. For rotor imbalance:\n');
fprintf('     - Add sinusoidal torque at shaft frequency\n');
fprintf('     - Amplitude proportional to omega^2\n\n');
fprintf('  6. For shaft misalignment:\n');
fprintf('     - Use Simscape > Flexible Shaft block\n');
fprintf('     - Set angular offset parameter\n\n');
fprintf('  7. For electrical faults:\n');
fprintf('     - Modify armature resistance (open circuit)\n');
fprintf('     - Add 2× line frequency noise source\n\n');
fprintf('  8. Use Signal Builder for fault scenario sequencing\n');
fprintf('  9. Export data using To Workspace blocks\n');
fprintf('==========================================================\n');
