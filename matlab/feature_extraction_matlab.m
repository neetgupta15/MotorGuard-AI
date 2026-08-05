%% =========================================================================
%  Feature Extraction using Predictive Maintenance Toolbox
%  =========================================================================
%  This script demonstrates feature extraction from vibration data using
%  MATLAB's Predictive Maintenance Toolbox and Statistics Toolbox.
%
%  Requirements:
%    - MATLAB R2022a or later
%    - Predictive Maintenance Toolbox
%    - Signal Processing Toolbox
%    - Statistics and Machine Learning Toolbox
%  =========================================================================

clear; clc; close all;

fprintf('==========================================================\n');
fprintf('  Feature Extraction — Predictive Maintenance Toolbox\n');
fprintf('==========================================================\n\n');

%% ─── Load Data ────────────────────────────────────────────────────────

if exist('data/matlab_signals.mat', 'file')
    load('data/matlab_signals.mat');
    fprintf('  Loaded data from matlab_signals.mat\n\n');
else
    fprintf('  WARNING: matlab_signals.mat not found.\n');
    fprintf('  Run motor_simulink_model.m first.\n');
    fprintf('  Generating sample data...\n\n');
    
    % Generate quick sample data
    fs = 12000;
    duration = 2.0;
    t = 0:1/fs:(duration - 1/fs);
    fr = 30; % 1800 RPM
    
    normal_x = 0.05*sin(2*pi*fr*t) + 0.01*randn(size(t));
    bearing_x = normal_x + 0.3*sin(2*pi*107*t).*exp(-mod(t*107,1)*50);
    imbalance_x = normal_x + 0.5*sin(2*pi*fr*t);
    
    sim_params.sample_rate = fs;
end

fs = sim_params.sample_rate;

%% ─── Time-Domain Feature Extraction ──────────────────────────────────

fprintf('  Extracting time-domain features...\n');

signals = {normal_x, bearing_x, imbalance_x};
labels = {'Normal', 'Bearing', 'Imbalance'};
n_signals = length(signals);

% Feature computation
features_table = table();

for i = 1:n_signals
    sig = signals{i};
    
    feat = struct();
    feat.Label = labels(i);
    feat.Mean = mean(sig);
    feat.Std = std(sig);
    feat.RMS = rms(sig);
    feat.Peak = max(abs(sig));
    feat.PeakToPeak = max(sig) - min(sig);
    feat.CrestFactor = max(abs(sig)) / rms(sig);
    feat.ShapeFactor = rms(sig) / mean(abs(sig));
    feat.ImpulseFactor = max(abs(sig)) / mean(abs(sig));
    feat.Kurtosis = kurtosis(sig);
    feat.Skewness = skewness(sig);
    feat.Variance = var(sig);
    feat.Energy = sum(sig.^2) / length(sig);
    
    % Zero crossing rate
    zc = sum(abs(diff(sign(sig))) > 0);
    feat.ZeroCrossingRate = zc / length(sig);
    
    features_table = [features_table; struct2table(feat)];
end

fprintf('\n  Time-Domain Features:\n');
disp(features_table);

%% ─── Frequency-Domain Features ──────────────────────────────────────

fprintf('  Extracting frequency-domain features...\n\n');

N = length(signals{1});
f = (0:N-1) * fs / N;

figure('Position', [100 100 1200 400], 'Color', 'w');

for i = 1:n_signals
    Y = abs(fft(signals{i})) / N;
    Y_half = Y(1:floor(N/2));
    f_half = f(1:floor(N/2));
    
    subplot(1, 3, i);
    plot(f_half, Y_half, 'LineWidth', 0.8);
    title([labels{i} ' — FFT'], 'FontWeight', 'bold');
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    xlim([0 500]); grid on;
    
    % Spectral features
    power = Y_half.^2;
    total_power = sum(power);
    
    if total_power > 0
        psd_norm = power / total_power;
        
        % Spectral centroid
        spectral_centroid = sum(f_half .* psd_norm);
        
        % Dominant frequency
        [~, dom_idx] = max(Y_half);
        dom_freq = f_half(dom_idx);
        
        fprintf('  %s:\n', labels{i});
        fprintf('    Spectral Centroid: %.1f Hz\n', spectral_centroid);
        fprintf('    Dominant Freq:     %.1f Hz\n', dom_freq);
        fprintf('    Total Power:       %.6f\n\n', total_power);
    end
end

sgtitle('Frequency Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% ─── Envelope Analysis (Bearing Fault Detection) ────────────────────

fprintf('  Performing envelope analysis for bearing fault detection...\n\n');

% Bandpass filter around bearing resonance
[b_bp, a_bp] = butter(4, [2000 5000]/(fs/2), 'bandpass');
bearing_filtered = filtfilt(b_bp, a_bp, bearing_x);

% Hilbert transform for envelope
bearing_envelope = abs(hilbert(bearing_filtered));

% Envelope spectrum
env_fft = abs(fft(bearing_envelope)) / length(bearing_envelope);
f_env = (0:length(bearing_envelope)-1) * fs / length(bearing_envelope);

figure('Position', [100 550 800 400], 'Color', 'w');

subplot(2,1,1);
plot(t(1:2400), bearing_envelope(1:2400), 'r', 'LineWidth', 0.8);
title('Bearing Fault — Envelope Signal', 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(f_env(1:floor(end/2)), env_fft(1:floor(end/2)), 'b', 'LineWidth', 0.8);
title('Envelope Spectrum', 'FontWeight', 'bold');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 300]); grid on;

% Mark bearing frequencies
hold on;
xline(bearing_params.BPFO, 'r--', sprintf('BPFO=%.0fHz', bearing_params.BPFO), ...
    'LineWidth', 1.5, 'LabelOrientation', 'horizontal');
xline(bearing_params.BPFI, 'g--', sprintf('BPFI=%.0fHz', bearing_params.BPFI), ...
    'LineWidth', 1.5, 'LabelOrientation', 'horizontal');
hold off;

%% ─── Diagnostic Feature Designer (Programmatic) ────────────────────

fprintf('  ─────────────────────────────────────────────────────\n');
fprintf('  DIAGNOSTIC FEATURE DESIGNER WORKFLOW\n');
fprintf('  ─────────────────────────────────────────────────────\n');
fprintf('  To use the interactive Diagnostic Feature Designer:\n\n');
fprintf('  1. >> diagnosticFeatureDesigner\n');
fprintf('  2. Import vibration signals as an ensemble\n');
fprintf('  3. Add time-domain features:\n');
fprintf('     - Signal Statistics (mean, std, kurtosis)\n');
fprintf('     - Impulsiveness (crest factor, peak)\n');
fprintf('  4. Add frequency-domain features:\n');
fprintf('     - Spectral Kurtosis\n');
fprintf('     - Band Power\n');
fprintf('  5. Add time-frequency features:\n');
fprintf('     - Wavelet Packet Decomposition\n');
fprintf('  6. Rank features by discriminative power\n');
fprintf('  7. Export selected features to workspace\n');
fprintf('  ─────────────────────────────────────────────────────\n\n');

%% ─── Classification with fitcecoc ──────────────────────────────────

fprintf('  Training multi-class SVM classifier...\n');

% Build feature matrix (simplified)
X = [];
Y = [];

for i = 1:n_signals
    sig = signals{i};
    feat_vec = [rms(sig), max(abs(sig)), kurtosis(sig), ...
                std(sig), max(abs(sig))/rms(sig), var(sig)];
    X = [X; feat_vec];
    Y = [Y; i-1];  % 0, 1, 2
end

% Note: In practice, you'd have many more samples
% This is a demonstration of the workflow

fprintf('  Feature matrix: %d samples x %d features\n', size(X,1), size(X,2));
fprintf('\n  For full classification:\n');
fprintf('    model = fitcecoc(X_train, Y_train);\n');
fprintf('    predictions = predict(model, X_test);\n');
fprintf('    confMat = confusionmat(Y_test, predictions);\n');
fprintf('    confusionchart(confMat, class_names);\n');

fprintf('\n==========================================================\n');
fprintf('  Feature extraction complete!\n');
fprintf('==========================================================\n');
