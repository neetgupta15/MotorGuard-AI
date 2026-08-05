/**
 * ══════════════════════════════════════════════════════════════════════
 * Motor Fault Detection — Real-Time Dashboard Engine
 * ══════════════════════════════════════════════════════════════════════
 *
 * Supported Modes:
 * 1. Live Stream Simulation — Continuous physics-based motor simulation
 * 2. 50+ Stored Sound & Vibration Presets — Catalog of pre-analyzed fault profiles
 * 3. Live Microphone Detection — Device microphone acoustic & vibration analysis
 *
 * Special Features:
 * - 30-Second Minimum Scaled Alert Visibility Engine with active timers
 * - Web Audio API live real-time FFT & feature extraction
 * - 50 stored preset sound & vibration dataset loader
 * ══════════════════════════════════════════════════════════════════════
 */

// ─── Configuration ────────────────────────────────────────────────────
const CONFIG = {
    updateInterval: 100,       // ms between simulation updates
    waveformPoints: 200,       // Points in waveform display
    fftSize: 256,              // FFT window size
    samplingFreq: 400,         // Hz (matching LIS3DH ODR)
    trendMaxPoints: 60,        // Health trend history length
    maxAlerts: 20,             // Maximum stored alerts
    maxTimeline: 80,           // Timeline blocks
    alertDurationMs: 60000,    // 1 minute (60s) scaled visible duration for alerts
};

const FAULT_CLASSES = {
    0: { name: 'Normal', color: '#22c55e', css: 'normal' },
    1: { name: 'Bearing Fault', color: '#ef4444', css: 'bearing' },
    2: { name: 'Rotor Imbalance', color: '#3b82f6', css: 'imbalance' },
    3: { name: 'Shaft Misalignment', color: '#f59e0b', css: 'misalignment' },
    4: { name: 'Electrical Fault', color: '#a855f7', css: 'electrical' },
};

// ─── Application State ────────────────────────────────────────────────
let state = {
    mode: 'simulated',          // 'simulated' | 'preset' | 'mic'
    windowCount: 0,
    startTime: Date.now(),
    currentFault: 0,
    confidence: 0.95,
    probabilities: [0.95, 0.02, 0.01, 0.01, 0.01],
    alerts: [],                 // Objects: { id, time, faultClass, className, text, expiryTime }
    timeline: [],
    trendData: [],
    waveformData: { x: [], y: [], z: [] },
    fftData: { freqs: [], magnitudes: [] },
    features: {},
    simTimer: null,

    // Presets state
    presets: [],
    currentPresetIndex: 0,
    autoPlayTimer: null,
    activeFilter: 'all',

    // Microphone state
    audioCtx: null,
    analyserNode: null,
    micStream: null,
    isMicRecording: false,
    micAnimFrame: null,
    micGain: 4.0,
    micGate: 0.005,

    // Recording & Stream state
    mediaRecorder: null,
    recordedChunks: [],
    recordedBlob: null,
    recTimer: null,
    recSeconds: 0,
    simPaused: false,
};

// ─── Stream Pause / Resume Toggle ────────────────────────────────────
function initStreamToggle() {
    const btnToggle = document.getElementById('btnToggleSimStream');
    if (btnToggle) {
        btnToggle.addEventListener('click', () => {
            state.simPaused = !state.simPaused;
            if (state.simPaused) {
                btnToggle.textContent = '▶ Resume Stream';
                btnToggle.classList.add('paused');
            } else {
                btnToggle.textContent = '⏸️ Pause Stream';
                btnToggle.classList.remove('paused');
            }
        });
    }
}

// ══════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initClassProbabilities();
    initFeatureGrid();
    initAxisButtons();
    initModeSelector();
    initPresetsEngine();
    initMicEngine();
    initStreamToggle();
    initAlertTimerLoop();
    startSimulation();
});

// ─── Initialize Charts ───────────────────────────────────────────────
function initCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 150 },
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(148,163,184,0.06)' },
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
            },
            y: {
                grid: { color: 'rgba(148,163,184,0.06)' },
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
            }
        }
    };

    // Waveform Chart
    const wfCtx = document.getElementById('waveformChart').getContext('2d');
    waveformChart = new Chart(wfCtx, {
        type: 'line',
        data: {
            labels: Array.from({ length: CONFIG.waveformPoints }, (_, i) => i),
            datasets: [
                { label: 'X-axis / Mic', data: [], borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
                { label: 'Y-axis', data: [], borderColor: '#22c55e', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
                { label: 'Z-axis', data: [], borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: { ...chartDefaults.scales.x, display: false },
                y: { ...chartDefaults.scales.y, title: { display: true, text: 'Amplitude / Acceleration (g)', color: '#64748b' }, suggestedMin: -1, suggestedMax: 1 }
            },
            plugins: { ...chartDefaults.plugins, legend: { position: 'top', labels: { ...chartDefaults.plugins.legend.labels, boxWidth: 12, padding: 10 } } }
        }
    });

    // FFT Chart
    const fftCtx = document.getElementById('fftChart').getContext('2d');
    fftChart = new Chart(fftCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Magnitude Spectrum',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.3,
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: { ...chartDefaults.scales.x, title: { display: true, text: 'Frequency (Hz)', color: '#64748b' } },
                y: { ...chartDefaults.scales.y, title: { display: true, text: 'Magnitude', color: '#64748b' }, min: 0 }
            },
            plugins: { ...chartDefaults.plugins, legend: { display: false } }
        }
    });

    // Trend Chart
    const trendCtx = document.getElementById('trendChart').getContext('2d');
    trendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Health Score', data: [], borderColor: '#22c55e', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(34, 197, 94, 0.08)', tension: 0.3 },
                { label: 'RMS Level', data: [], borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, borderDash: [4, 4], tension: 0.3 }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: { ...chartDefaults.scales.x, display: false },
                y: { ...chartDefaults.scales.y, min: 0, max: 1.1, title: { display: true, text: 'Score', color: '#64748b' } }
            }
        }
    });
}

// ─── Class Probabilities UI ──────────────────────────────────────────
function initClassProbabilities() {
    const container = document.getElementById('classProbabilities');
    container.innerHTML = Object.entries(FAULT_CLASSES).map(([id, cls]) => `
        <div class="prob-row">
            <span class="prob-label">${cls.name.replace(' Fault', '')}</span>
            <div class="prob-bar-wrap">
                <div class="prob-bar-inner" id="probBar${id}" style="width: 0%; background: ${cls.color}"></div>
            </div>
            <span class="prob-value" id="probVal${id}">0%</span>
        </div>
    `).join('');
}

// ─── Feature Grid UI ─────────────────────────────────────────────────
function initFeatureGrid() {
    const features = [
        { key: 'rms', label: 'RMS' },
        { key: 'peak', label: 'Peak' },
        { key: 'kurtosis', label: 'Kurtosis' },
        { key: 'crest', label: 'Crest Factor' },
        { key: 'energy', label: 'Energy' },
        { key: 'zcr', label: 'Zero X Rate' },
        { key: 'shape', label: 'Shape Factor' },
        { key: 'skewness', label: 'Skewness' },
    ];

    const grid = document.getElementById('featuresGrid');
    grid.innerHTML = features.map(f => `
        <div class="feature-item">
            <span class="feature-name">${f.label}</span>
            <span class="feature-value" id="feat_${f.key}">0.000</span>
        </div>
    `).join('');
}

// ─── Axis Filter Buttons ─────────────────────────────────────────────
function initAxisButtons() {
    document.querySelectorAll('.chart-btn[data-axis]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.chart-btn[data-axis]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const axis = btn.dataset.axis;
            waveformChart.data.datasets[0].hidden = axis !== 'all' && axis !== 'x';
            waveformChart.data.datasets[1].hidden = axis !== 'all' && axis !== 'y';
            waveformChart.data.datasets[2].hidden = axis !== 'all' && axis !== 'z';
            waveformChart.update('none');
        });
    });
}

// ══════════════════════════════════════════════════════════════════════
// MODE SWITCHER ENGINE
// ══════════════════════════════════════════════════════════════════════

function initModeSelector() {
    const btnSim = document.getElementById('btnModeSim');
    const btnPreset = document.getElementById('btnModePreset');
    const btnMic = document.getElementById('btnModeMic');
    const statusText = document.getElementById('statusModeText');
    const presetPanel = document.getElementById('presetPanel');
    const micPanel = document.getElementById('micPanel');

    function switchMode(newMode) {
        state.mode = newMode;

        // Reset mode buttons
        btnSim.classList.remove('active');
        btnPreset.classList.remove('active');
        btnMic.classList.remove('active');

        // Hide panels by default
        presetPanel.classList.add('hidden');
        micPanel.classList.add('hidden');

        // Stop microphone if switching away
        if (newMode !== 'mic' && state.isMicRecording) {
            stopMicRecording();
        }

        // Stop auto-play timer if switching away
        if (newMode !== 'preset' && state.autoPlayTimer) {
            clearInterval(state.autoPlayTimer);
            state.autoPlayTimer = null;
            document.getElementById('btnAutoPlayPreset').textContent = '▶ Auto-Cycle (3s)';
        }

        if (newMode === 'simulated') {
            btnSim.classList.add('active');
            statusText.textContent = 'Simulated Data';
            if (!state.simTimer) startSimulation();
        } else if (newMode === 'preset') {
            btnPreset.classList.add('active');
            statusText.textContent = '50+ Preset Library';
            presetPanel.classList.remove('hidden');
            if (state.presets.length > 0) {
                applyPreset(state.currentPresetIndex);
            }
        } else if (newMode === 'mic') {
            btnMic.classList.add('active');
            statusText.textContent = 'Live Mic Input';
            micPanel.classList.remove('hidden');
        }
    }

    btnSim.addEventListener('click', () => switchMode('simulated'));
    btnPreset.addEventListener('click', () => switchMode('preset'));
    btnMic.addEventListener('click', () => switchMode('mic'));
}

// ══════════════════════════════════════════════════════════════════════
// 50+ PRESETS ENGINE
// ══════════════════════════════════════════════════════════════════════

function initPresetsEngine() {
    // Try fetching generated preset_samples.json, otherwise generate 50 procedurally
    fetch('preset_samples.json')
        .then(res => res.json())
        .then(data => {
            state.presets = data;
            setupPresetUI();
        })
        .catch(() => {
            console.log('Generating 50 dynamic presets client-side fallback...');
            state.presets = generateFallbackPresets();
            setupPresetUI();
        });
}

function generateFallbackPresets() {
    const presets = [];
    const classes = [0, 1, 2, 3, 4];
    const classNames = ['Normal', 'Bearing Fault', 'Rotor Imbalance', 'Shaft Misalignment', 'Electrical Fault'];
    const rpms = [1200, 1500, 1800, 2400, 3000, 3600];

    let idCount = 1;
    for (let c of classes) {
        for (let i = 0; i < 10; i++) {
            const rpm = rpms[i % rpms.length];
            const sev = c === 0 ? 0 : 0.2 + (i / 10) * 0.75;
            const health = c === 0 ? 0.96 : Math.max(0.15, 1 - sev * 0.8);
            const title = `${classNames[c]} Preset #${i + 1} (${rpm} RPM)`;
            
            // Waveform
            const x = [], y = [], z = [];
            const dt = 1 / 400;
            const freq = rpm / 60;
            for (let k = 0; k < CONFIG.waveformPoints; k++) {
                const t = k * dt;
                let valX = 0.05 * Math.sin(2 * Math.PI * freq * t);
                let valY = 0.05 * Math.cos(2 * Math.PI * freq * t);
                let valZ = 0.02 * Math.sin(2 * Math.PI * freq * t);

                if (c === 1) { valX += sev * 0.8 * (Math.random() > 0.9 ? 1.5 : 0.1) * Math.sin(200 * t); }
                if (c === 2) { valX += sev * 0.6 * Math.sin(2 * Math.PI * freq * t); valY += sev * 0.6 * Math.sin(2 * Math.PI * freq * t + Math.PI/2); }
                if (c === 3) { valX += sev * 0.5 * Math.sin(4 * Math.PI * freq * t); valZ += sev * 0.5 * Math.sin(2 * Math.PI * freq * t); }
                if (c === 4) { valX += sev * 0.4 * Math.sin(2 * Math.PI * 100 * t); }

                valX += 0.01 * (Math.random() - 0.5);
                valY += 0.01 * (Math.random() - 0.5);
                valZ += 0.01 * (Math.random() - 0.5);

                x.push(valX); y.push(valY); z.push(valZ);
            }

            const rmsX = Math.sqrt(x.reduce((a, b) => a + b*b, 0) / x.length);
            const rmsY = Math.sqrt(y.reduce((a, b) => a + b*b, 0) / y.length);
            const rmsZ = Math.sqrt(z.reduce((a, b) => a + b*b, 0) / z.length);
            const peak = Math.max(...x.map(Math.abs));

            // FFT
            const freqs = Array.from({length: 64}, (_, k) => (k * 3.125).toFixed(0));
            const mags = freqs.map((f, k) => (k === Math.floor(freq / 3.125) ? 0.2 + sev : 0.01 + Math.random() * 0.02));

            // Probs
            const probs = [0.02, 0.02, 0.02, 0.02, 0.02];
            probs[c] = 0.7 + sev * 0.25;
            probs[0] = Math.max(0.02, 1.0 - probs[c]);
            const pSum = probs.reduce((a,b)=>a+b, 0);

            presets.push({
                id: `P-${String(idCount++).padStart(2, '0')}`,
                title,
                faultClass: c,
                faultClassName: classNames[c],
                severity: sev,
                rpm,
                healthScore: health,
                description: `Synthetic motor vibration profile representing ${classNames[c]} at ${rpm} RPM with severity level ${(sev*100).toFixed(0)}%.`,
                waveform: { x, y, z },
                fft: { freqs, magnitudes: mags },
                rms: { x: rmsX, y: rmsY, z: rmsZ },
                features: {
                    rms: rmsX, peak, kurtosis: c === 1 ? 5.5 + sev * 3 : 2.8 + Math.random() * 0.5,
                    crest: rmsX > 0 ? peak / rmsX : 0, energy: rmsX * rmsX,
                    zcr: 0.15 + Math.random() * 0.1, shape: 1.25, skewness: (Math.random() - 0.5) * 0.5
                },
                probabilities: probs.map(p => p / pSum),
                confidence: Math.max(...probs) / pSum
            });
        }
    }
    return presets;
}

// ══════════════════════════════════════════════════════════════════════
// MOTOR AUDIO SYNTHESIZER ENGINE
// ══════════════════════════════════════════════════════════════════════

class MotorAudioSynth {
    constructor() {
        this.ctx = null;
        this.currentSourceNodes = [];
        this.isPlaying = false;
        this.masterVolume = 0.5;
    }

    init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    setVolume(vol) {
        this.masterVolume = Math.max(0, Math.min(1, vol));
    }

    stop() {
        this.currentSourceNodes.forEach(n => {
            try { n.stop(); n.disconnect(); } catch (e) {}
        });
        this.currentSourceNodes = [];
        this.isPlaying = false;

        const btn = document.getElementById('btnPlaySound');
        if (btn) {
            btn.classList.remove('playing');
            btn.textContent = '🔊 Play Sound';
        }
    }

    playPreset(preset, durationSec = 60) {
        this.stop();
        this.init();

        const sampleRate = this.ctx.sampleRate;
        const totalSamples = sampleRate * durationSec;
        const buffer = this.ctx.createBuffer(1, totalSamples, sampleRate);
        const data = buffer.getChannelData(0);

        const rpm = preset.rpm || 1800;
        const shaftFreq = rpm / 60; // e.g. 30 Hz
        const faultClass = preset.faultClass || 0;
        const sev = preset.severity || 0;
        const wave = (preset.waveform && preset.waveform.x) ? preset.waveform.x : [];

        for (let i = 0; i < totalSamples; i++) {
            const t = i / sampleRate;
            let sample = 0;

            // 1. Motor shaft fundamental hum + harmonics
            sample += 0.25 * Math.sin(2 * Math.PI * shaftFreq * t);
            sample += 0.12 * Math.sin(2 * Math.PI * (2 * shaftFreq) * t);
            sample += 0.05 * Math.sin(2 * Math.PI * (3 * shaftFreq) * t);

            // 2. Electrical 100 Hz hum
            sample += 0.08 * Math.sin(2 * Math.PI * 100 * t);

            // 3. Background airflow noise
            sample += 0.02 * (Math.random() - 0.5);

            // 4. Fault-specific acoustic characteristics
            if (faultClass === 1) { // Bearing Fault: Periodic high-frequency impact clicks
                const bpfo = shaftFreq * 3.56;
                const period = 1 / bpfo;
                const phase = (t % period) / period;
                if (phase < 0.05) {
                    const burstEnv = Math.exp(-phase * 120);
                    sample += sev * 0.7 * burstEnv * Math.sin(2 * Math.PI * 2800 * t);
                }
            } else if (faultClass === 2) { // Rotor Imbalance: Deep 1x low frequency heavy thumping
                sample += sev * 0.6 * Math.sin(2 * Math.PI * shaftFreq * t);
            } else if (faultClass === 3) { // Shaft Misalignment: Strong 2x harmonic buzz
                sample += sev * 0.5 * Math.sin(2 * Math.PI * (2 * shaftFreq) * t);
                sample += sev * 0.3 * Math.sin(2 * Math.PI * (4 * shaftFreq) * t);
            } else if (faultClass === 4) { // Electrical Fault: Harsh 100Hz/200Hz buzzing
                sample += sev * 0.5 * (Math.sin(2 * Math.PI * 100 * t) > 0 ? 0.3 : -0.3);
                sample += sev * 0.3 * Math.sin(2 * Math.PI * 300 * t);
            }

            if (wave.length > 0) {
                const waveIdx = Math.floor((i % wave.length));
                sample += wave[waveIdx] * 0.3;
            }

            data[i] = Math.max(-1, Math.min(1, sample * this.masterVolume));
        }

        const source = this.ctx.createBufferSource();
        source.buffer = buffer;

        const gainNode = this.ctx.createGain();
        gainNode.gain.value = this.masterVolume;

        source.connect(gainNode);
        gainNode.connect(this.ctx.destination);

        source.start();
        this.currentSourceNodes.push(source);
        this.isPlaying = true;

        const btn = document.getElementById('btnPlaySound');
        if (btn) {
            btn.classList.add('playing');
            btn.textContent = '⏹️ Stop Sound';
        }

        source.onended = () => {
            this.isPlaying = false;
            if (btn) {
                btn.classList.remove('playing');
                btn.textContent = '🔊 Play Sound';
            }
        };
    }
}

const audioSynth = new MotorAudioSynth();

function setupPresetUI() {
    const select = document.getElementById('presetSelect');
    const badge = document.getElementById('presetCountBadge');
    badge.textContent = `${state.presets.length} Loaded`;

    populatePresetDropdown('all');

    // Filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activeFilter = btn.dataset.class;
            populatePresetDropdown(state.activeFilter);
        });
    });

    // Select change
    select.addEventListener('change', (e) => {
        const idx = parseInt(e.target.value, 10);
        if (!isNaN(idx)) {
            state.currentPresetIndex = idx;
            applyPreset(idx);
        }
    });

    // Prev / Next buttons
    document.getElementById('btnPrevPreset').addEventListener('click', () => {
        if (state.presets.length === 0) return;
        state.currentPresetIndex = (state.currentPresetIndex - 1 + state.presets.length) % state.presets.length;
        select.value = state.currentPresetIndex;
        applyPreset(state.currentPresetIndex);
    });

    document.getElementById('btnNextPreset').addEventListener('click', () => {
        if (state.presets.length === 0) return;
        state.currentPresetIndex = (state.currentPresetIndex + 1) % state.presets.length;
        select.value = state.currentPresetIndex;
        applyPreset(state.currentPresetIndex);
    });

    // Auto-play button
    const btnAuto = document.getElementById('btnAutoPlayPreset');
    btnAuto.addEventListener('click', () => {
        if (state.autoPlayTimer) {
            clearInterval(state.autoPlayTimer);
            state.autoPlayTimer = null;
            btnAuto.textContent = '▶ Auto-Cycle (3s)';
            audioSynth.stop();
        } else {
            btnAuto.textContent = '⏸ Pause Cycle';
            state.autoPlayTimer = setInterval(() => {
                state.currentPresetIndex = (state.currentPresetIndex + 1) % state.presets.length;
                select.value = state.currentPresetIndex;
                applyPreset(state.currentPresetIndex);
            }, 3000);
        }
    });

    // Audio Play / Stop Button
    const btnPlaySound = document.getElementById('btnPlaySound');
    if (btnPlaySound) {
        btnPlaySound.addEventListener('click', () => {
            if (audioSynth.isPlaying) {
                audioSynth.stop();
            } else if (state.presets[state.currentPresetIndex]) {
                audioSynth.playPreset(state.presets[state.currentPresetIndex]);
            }
        });
    }

    // Volume slider
    const volSlider = document.getElementById('soundVolume');
    if (volSlider) {
        volSlider.addEventListener('input', (e) => {
            audioSynth.setVolume(parseFloat(e.target.value));
        });
    }
}

function populatePresetDropdown(filterClass) {
    const select = document.getElementById('presetSelect');
    select.innerHTML = state.presets
        .map((p, idx) => ({ p, idx }))
        .filter(({ p }) => filterClass === 'all' || p.faultClass.toString() === filterClass)
        .map(({ p, idx }) => `<option value="${idx}">[${p.id}] ${p.title} — ${p.faultClassName} (${(p.confidence * 100).toFixed(0)}%)</option>`)
        .join('');

    if (select.options.length > 0) {
        select.selectedIndex = 0;
        state.currentPresetIndex = parseInt(select.value, 10);
    }
}

function applyPreset(index) {
    const p = state.presets[index];
    if (!p) return;

    state.windowCount++;
    state.currentFault = p.faultClass;
    state.confidence = p.confidence;
    state.probabilities = p.probabilities;
    state.waveformData = p.waveform;
    state.fftData = p.fft;
    state.features = p.features;

    // Check auto-sound checkbox
    const chkAuto = document.getElementById('chkAutoSound');
    if (chkAuto && chkAuto.checked && state.mode === 'preset') {
        audioSynth.playPreset(p);
    }

    // Update banner
    document.getElementById('presetClassBadge').textContent = `Class: ${p.faultClassName}`;
    document.getElementById('presetClassBadge').style.background = FAULT_CLASSES[p.faultClass].color + '22';
    document.getElementById('presetClassBadge').style.color = FAULT_CLASSES[p.faultClass].color;
    document.getElementById('presetRpmBadge').textContent = `${p.rpm} RPM`;
    document.getElementById('presetSevBadge').textContent = `Severity: ${(p.severity * 100).toFixed(0)}%`;
    document.getElementById('presetDesc').textContent = p.description;

    // Trend & Timeline
    state.trendData.push(p.healthScore);
    if (state.trendData.length > CONFIG.trendMaxPoints) state.trendData.shift();

    state.timeline.push(p.faultClass);
    if (state.timeline.length > CONFIG.maxTimeline) state.timeline.shift();

    // Trigger scaled 30s alert if fault detected
    if (p.faultClass !== 0) {
        addScaledAlert(p.faultClass, p.confidence, p.title);
    }

    updateUI(p.waveform, p.features, p.probabilities, p.faultClass, p.confidence, p.healthScore);
}

// ══════════════════════════════════════════════════════════════════════
// SCALED 30-SECOND ALERT VISIBILITY ENGINE
// ══════════════════════════════════════════════════════════════════════

function addScaledAlert(faultClass, confidence, customTitle = null) {
    const now = Date.now();
    const timeStr = new Date(now).toTimeString().slice(0, 8);
    const cls = FAULT_CLASSES[faultClass];
    const alertId = 'alert_' + now + '_' + Math.floor(Math.random() * 1000);

    const titleText = customTitle || `${cls.name} detected (${(confidence * 100).toFixed(0)}%)`;

    // Deduplicate recent identical alerts within 3 seconds
    const existing = state.alerts.find(a => a.faultClass === faultClass && (now - a.createdAt) < 3000);
    if (existing) {
        existing.expiryTime = now + CONFIG.alertDurationMs; // Refresh 30s timer
        return;
    }

    state.alerts.unshift({
        id: alertId,
        time: timeStr,
        faultClass,
        className: cls.css,
        text: titleText,
        createdAt: now,
        expiryTime: now + CONFIG.alertDurationMs, // Visible for AT LEAST 30 seconds
    });

    if (state.alerts.length > CONFIG.maxAlerts) state.alerts.pop();
    renderAlertsList();
}

function initAlertTimerLoop() {
    // Check and update alert countdowns every 200ms
    setInterval(() => {
        const now = Date.now();
        // Remove expired alerts (older than 30 seconds)
        const initialCount = state.alerts.length;
        state.alerts = state.alerts.filter(a => now < a.expiryTime);

        if (state.alerts.length > 0 || initialCount !== state.alerts.length) {
            renderAlertsList();
        }
    }, 200);
}

function renderAlertsList() {
    const alertsList = document.getElementById('alertsList');
    const alertBadge = document.getElementById('alertBadge');
    const now = Date.now();

    if (state.alerts.length === 0) {
        alertsList.innerHTML = '<div class="alert-empty">No active alerts</div>';
        alertBadge.textContent = '0';
        alertBadge.classList.add('hidden');
        return;
    }

    alertBadge.textContent = state.alerts.length;
    alertBadge.classList.remove('hidden');

    const faultIcons = { 0: '✅', 1: '⚙️', 2: '⚖️', 3: '🔀', 4: '⚡' };

    alertsList.innerHTML = state.alerts.slice(0, 8).map(a => {
        const remainingMs = Math.max(0, a.expiryTime - now);
        const remainingSec = Math.ceil(remainingMs / 1000);
        const progressPct = (remainingMs / CONFIG.alertDurationMs) * 100;
        const icon = faultIcons[a.faultClass] || '⚠️';

        return `
            <div class="alert-item alert-${a.className}" id="${a.id}">
                <div class="alert-item-header">
                    <div class="alert-time-box">
                        <span class="alert-icon">${icon}</span>
                        <span class="alert-time">${a.time}</span>
                    </div>
                    <span class="alert-countdown-badge">⏳ ${remainingSec}s active</span>
                </div>
                <div class="alert-text">${a.text}</div>
                <div class="alert-progress-bar-wrap">
                    <div class="alert-progress-bar alert-bar-${a.className}" style="width: ${progressPct}%"></div>
                </div>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════════════════════════════
// LIVE MICROPHONE DETECTION ENGINE
// ══════════════════════════════════════════════════════════════════════

function initMicEngine() {
    const btnToggle = document.getElementById('btnMicToggle');
    const gainSlider = document.getElementById('micGainSlider');
    const gateSlider = document.getElementById('micNoiseGate');

    gainSlider.addEventListener('input', (e) => {
        state.micGain = parseFloat(e.target.value);
        document.getElementById('micGainVal').textContent = `${state.micGain.toFixed(1)}x`;
    });

    gateSlider.addEventListener('input', (e) => {
        state.micGate = parseFloat(e.target.value);
        document.getElementById('micGateVal').textContent = state.micGate.toFixed(3);
    });

    btnToggle.addEventListener('click', () => {
        if (state.isMicRecording) {
            stopMicRecording();
        } else {
            startMicRecording();
        }
    });
}

function startMicRecording() {
    const btnToggle = document.getElementById('btnMicToggle');
    const statusPill = document.getElementById('micStatusPill');
    const statusText = document.getElementById('micStatusText');
    const recTimerTag = document.getElementById('recTimerTag');
    const btnPlayRecorded = document.getElementById('btnPlayRecorded');
    const btnDownloadRecorded = document.getElementById('btnDownloadRecorded');
    const recordedAudioPlayer = document.getElementById('recordedAudioPlayer');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Microphone access is not supported by your browser.');
        return;
    }

    // Hide old playback controls during new recording
    if (btnPlayRecorded) btnPlayRecorded.classList.add('hidden');
    if (btnDownloadRecorded) btnDownloadRecorded.classList.add('hidden');
    if (recordedAudioPlayer) recordedAudioPlayer.classList.add('hidden');

    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
        .then(stream => {
            state.micStream = stream;
            state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = state.audioCtx.createMediaStreamSource(stream);
            state.analyserNode = state.audioCtx.createAnalyser();
            state.analyserNode.fftSize = 512;
            source.connect(state.analyserNode);

            // Initialize MediaRecorder for playback & download
            state.recordedChunks = [];
            try {
                state.mediaRecorder = new MediaRecorder(stream);
                state.mediaRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) state.recordedChunks.push(e.data);
                };
                state.mediaRecorder.start(100);
            } catch (e) {
                console.warn('MediaRecorder error:', e);
            }

            state.isMicRecording = true;
            btnToggle.classList.add('recording');
            document.getElementById('micBtnIcon').textContent = '⏹️';
            document.getElementById('micBtnLabel').textContent = 'Stop Recording';
            statusPill.classList.add('recording');
            statusText.textContent = 'Live Audio Recording...';

            // Start live recording duration timer
            state.recSeconds = 0;
            if (recTimerTag) {
                recTimerTag.classList.remove('hidden');
                recTimerTag.textContent = '🔴 00:00';
            }
            state.recTimer = setInterval(() => {
                state.recSeconds++;
                const m = String(Math.floor(state.recSeconds / 60)).padStart(2, '0');
                const s = String(state.recSeconds % 60).padStart(2, '0');
                if (recTimerTag) recTimerTag.textContent = `🔴 ${m}:${s}`;
            }, 1000);

            processMicFrame();
        })
        .catch(err => {
            console.error('Microphone access denied or error:', err);
            alert('Could not access microphone: ' + err.message);
        });
}

function stopMicRecording() {
    const btnToggle = document.getElementById('btnMicToggle');
    const statusPill = document.getElementById('micStatusPill');
    const statusText = document.getElementById('micStatusText');
    const recTimerTag = document.getElementById('recTimerTag');
    const btnPlayRecorded = document.getElementById('btnPlayRecorded');
    const btnDownloadRecorded = document.getElementById('btnDownloadRecorded');
    const recordedAudioPlayer = document.getElementById('recordedAudioPlayer');

    if (state.recTimer) {
        clearInterval(state.recTimer);
        state.recTimer = null;
    }

    if (state.micAnimFrame) {
        cancelAnimationFrame(state.micAnimFrame);
        state.micAnimFrame = null;
    }

    // Stop MediaRecorder and build playback URL
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
        state.mediaRecorder.onstop = () => {
            if (state.recordedChunks.length > 0) {
                state.recordedBlob = new Blob(state.recordedChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(state.recordedBlob);

                if (recordedAudioPlayer) {
                    recordedAudioPlayer.src = audioUrl;
                    recordedAudioPlayer.classList.remove('hidden');
                }

                if (btnPlayRecorded) {
                    btnPlayRecorded.classList.remove('hidden');
                    btnPlayRecorded.onclick = () => {
                        if (recordedAudioPlayer) recordedAudioPlayer.play();
                    };
                }

                if (btnDownloadRecorded) {
                    btnDownloadRecorded.classList.remove('hidden');
                    btnDownloadRecorded.onclick = () => {
                        const a = document.createElement('a');
                        a.href = audioUrl;
                        a.download = `motor_recording_${Date.now()}.webm`;
                        a.click();
                    };
                }
            }
        };
    }

    if (state.micStream) {
        state.micStream.getTracks().forEach(track => track.stop());
        state.micStream = null;
    }

    if (state.audioCtx) {
        state.audioCtx.close();
        state.audioCtx = null;
    }

    state.isMicRecording = false;
    btnToggle.classList.remove('recording');
    document.getElementById('micBtnIcon').textContent = '🎤';
    document.getElementById('micBtnLabel').textContent = 'Start Recording';
    statusPill.classList.remove('recording');
    statusText.textContent = 'Recording Saved';
    if (recTimerTag) recTimerTag.classList.add('hidden');
}

function processMicFrame() {
    if (!state.isMicRecording || !state.analyserNode) return;

    const bufferLength = state.analyserNode.frequencyBinCount;
    const timeDomain = new Float32Array(bufferLength);
    const freqDomain = new Float32Array(bufferLength);

    state.analyserNode.getFloatTimeDomainData(timeDomain);
    state.analyserNode.getFloatFrequencyData(freqDomain);

    // Apply gain and gate
    const xWave = [], yWave = [], zWave = [];
    let sumSq = 0;
    let peak = 0;

    for (let i = 0; i < CONFIG.waveformPoints && i < bufferLength; i++) {
        let val = timeDomain[i] * state.micGain;
        if (Math.abs(val) < state.micGate) val = 0;

        xWave.push(val);
        yWave.push(val * 0.7);
        zWave.push(val * 0.3);

        sumSq += val * val;
        if (Math.abs(val) > peak) peak = Math.abs(val);
    }

    const rms = Math.sqrt(sumSq / xWave.length);
    const crest = rms > 0 ? peak / rms : 0;

    // Kurtosis estimate
    let sum4 = 0;
    for (let v of xWave) sum4 += (v / (rms || 1)) ** 4;
    const kurtosis = xWave.length > 0 ? sum4 / xWave.length - 3 : 0;

    // FFT magnitudes
    const freqs = [], mags = [];
    const nyquist = (state.audioCtx ? state.audioCtx.sampleRate : 44100) / 2;
    for (let i = 0; i < 64 && i < bufferLength; i++) {
        freqs.push((i * (nyquist / bufferLength)).toFixed(0));
        const db = freqDomain[i];
        const lin = Math.max(0, (db + 100) / 100);
        mags.push(lin);
    }

    // Heuristic fault classification based on live acoustic properties
    let predictedClass = 0;
    let probs = [0.85, 0.04, 0.04, 0.04, 0.03];

    if (kurtosis > 4.5 || crest > 5.0) {
        predictedClass = 1; // Bearing fault (impulsive audio clicks)
        probs = [0.10, 0.75, 0.05, 0.05, 0.05];
    } else if (rms > 0.3) {
        predictedClass = 2; // Imbalance (heavy loud rumble)
        probs = [0.10, 0.05, 0.75, 0.05, 0.05];
    } else if (rms > 0.15 && crest > 3.0) {
        predictedClass = 3; // Misalignment
        probs = [0.10, 0.05, 0.05, 0.75, 0.05];
    } else if (rms > 0.12 && kurtosis > 2.5) {
        predictedClass = 4; // Electrical fault
        probs = [0.10, 0.05, 0.05, 0.05, 0.75];
    }

    const confidence = Math.max(...probs);
    const healthScore = predictedClass === 0 ? 0.95 : Math.max(0.2, 1.0 - rms * 2.0);

    state.windowCount++;
    state.currentFault = predictedClass;
    state.confidence = confidence;
    state.probabilities = probs;
    state.waveformData = { x: xWave, y: yWave, z: zWave };
    state.fftData = { freqs, magnitudes: mags };
    state.features = { rms, peak, kurtosis, crest, energy: rms * rms, zcr: 0.2, shape: 1.2, skewness: 0.1 };

    state.trendData.push(healthScore);
    if (state.trendData.length > CONFIG.trendMaxPoints) state.trendData.shift();

    state.timeline.push(predictedClass);
    if (state.timeline.length > CONFIG.maxTimeline) state.timeline.shift();

    if (predictedClass !== 0 && confidence > 0.6) {
        addScaledAlert(predictedClass, confidence, `Live Mic: ${FAULT_CLASSES[predictedClass].name} (Audio Level: ${rms.toFixed(2)})`);
    }

    updateUI(state.waveformData, state.features, probs, predictedClass, confidence, healthScore);

    state.micAnimFrame = requestAnimationFrame(processMicFrame);
}

// ══════════════════════════════════════════════════════════════════════
// SIMULATION ENGINE (LIVE STREAM MODE)
// ══════════════════════════════════════════════════════════════════════

class VibrationSimulator {
    constructor() {
        this.t = 0;
        this.shaftFreq = 30; // Hz (1800 RPM)
    }

    generate(faultType, severity = 0.5) {
        const dt = 1 / CONFIG.samplingFreq;
        const samples = { x: [], y: [], z: [] };

        for (let i = 0; i < CONFIG.waveformPoints; i++) {
            this.t += dt;
            let x = 0, y = 0, z = 0;

            const baseAmp = 0.05;
            x += baseAmp * Math.sin(2 * Math.PI * this.shaftFreq * this.t);
            y += baseAmp * Math.sin(2 * Math.PI * this.shaftFreq * this.t + Math.PI / 2);
            z += baseAmp * 0.15 * Math.sin(2 * Math.PI * this.shaftFreq * this.t);

            x += 0.01 * (Math.random() - 0.5);
            y += 0.01 * (Math.random() - 0.5);
            z += 0.005 * (Math.random() - 0.5);

            switch (faultType) {
                case 1: {
                    const bpfo = this.shaftFreq * 3.56;
                    const impulsePhase = (this.t * bpfo) % 1;
                    if (impulsePhase < 0.02) {
                        const burst = severity * 0.8 * Math.exp(-impulsePhase * 200) * Math.sin(2 * Math.PI * 3000 * this.t);
                        x += burst; y += burst * 0.7;
                    }
                    break;
                }
                case 2:
                    x += severity * 0.6 * Math.sin(2 * Math.PI * this.shaftFreq * this.t + 0.5);
                    y += severity * 0.6 * Math.sin(2 * Math.PI * this.shaftFreq * this.t + Math.PI / 2 + 0.5);
                    break;
                case 3:
                    x += severity * 0.4 * Math.sin(2 * Math.PI * 2 * this.shaftFreq * this.t);
                    z += severity * 0.5 * Math.sin(2 * Math.PI * this.shaftFreq * this.t);
                    break;
                case 4:
                    x += severity * 0.35 * Math.sin(2 * Math.PI * 100 * this.t);
                    y += severity * 0.3 * Math.sin(2 * Math.PI * 100 * this.t + Math.PI / 4);
                    break;
            }

            samples.x.push(x); samples.y.push(y); samples.z.push(z);
        }

        return samples;
    }
}

function startSimulation() {
    const simulator = new VibrationSimulator();
    let tick = 0;

    state.simTimer = setInterval(() => {
        if (state.mode !== 'simulated' || state.simPaused) return;

        tick++;
        const progress = (tick % 300) / 300;
        let faultType = 0, severity = 0;

        if (progress < 0.3) { faultType = 0; severity = 0; }
        else if (progress < 0.45) { faultType = 1; severity = (progress - 0.3) / 0.15 * 0.7; }
        else if (progress < 0.55) { faultType = 2; severity = 0.6; }
        else if (progress < 0.65) { faultType = 0; severity = 0; }
        else if (progress < 0.78) { faultType = 3; severity = 0.5; }
        else if (progress < 0.90) { faultType = 4; severity = 0.4; }
        else { faultType = 1; severity = 0.9; }

        const samples = simulator.generate(faultType, severity);
        const features = computeFeatures(samples.x);
        const probs = simulateClassification(faultType, severity, features);
        const confidence = Math.max(...probs);
        const predictedClass = probs.indexOf(confidence);

        state.windowCount++;
        state.currentFault = predictedClass;
        state.confidence = confidence;
        state.probabilities = probs;
        state.waveformData = samples;
        state.features = features;

        if (tick % 5 === 0) {
            state.fftData = computeFFT(samples.x);
        }

        const healthScore = faultType === 0 ? 0.95 + Math.random() * 0.05 : Math.max(0.1, 1.0 - severity * 0.8 + Math.random() * 0.1);
        state.trendData.push(healthScore);
        if (state.trendData.length > CONFIG.trendMaxPoints) state.trendData.shift();

        state.timeline.push(predictedClass);
        if (state.timeline.length > CONFIG.maxTimeline) state.timeline.shift();

        if (predictedClass !== 0 && confidence > 0.5) {
            addScaledAlert(predictedClass, confidence);
        }

        updateUI(samples, features, probs, predictedClass, confidence, healthScore);

    }, CONFIG.updateInterval);

    setInterval(updateUptime, 1000);
}

function simulateClassification(trueFault, severity, features) {
    const probs = [0.05, 0.05, 0.05, 0.05, 0.05];
    if (trueFault === 0) {
        probs[0] = 0.85 + Math.random() * 0.12;
    } else {
        probs[trueFault] = 0.5 + severity * 0.35 + Math.random() * 0.1;
        probs[0] = Math.max(0.02, 1.0 - severity) * 0.3;
    }
    const sum = probs.reduce((a, b) => a + b, 0);
    return probs.map(p => p / sum);
}

function computeFFT(signal) {
    const N = signal.length;
    const nextPow2 = Math.pow(2, Math.ceil(Math.log2(N)));
    const padded = new Float64Array(nextPow2);
    for (let i = 0; i < N; i++) padded[i] = signal[i];

    const magnitudes = [];
    const freqStep = CONFIG.samplingFreq / nextPow2;
    const maxBin = Math.floor(nextPow2 / 2);

    for (let k = 0; k < maxBin; k++) {
        let re = 0, im = 0;
        for (let n = 0; n < nextPow2; n++) {
            const angle = -2 * Math.PI * k * n / nextPow2;
            re += padded[n] * Math.cos(angle);
            im += padded[n] * Math.sin(angle);
        }
        magnitudes.push(Math.sqrt(re * re + im * im) / nextPow2);
    }

    const freqs = Array.from({ length: maxBin }, (_, i) => (i * freqStep).toFixed(0));
    return { freqs, magnitudes };
}

function computeFeatures(signal) {
    const N = signal.length;
    const mean = signal.reduce((a, b) => a + b, 0) / N;
    const variance = signal.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
    const std = Math.sqrt(variance);
    const rms = Math.sqrt(signal.reduce((a, b) => a + b * b, 0) / N);
    const peak = Math.max(...signal.map(Math.abs));
    const energy = signal.reduce((a, b) => a + b * b, 0) / N;

    let crossings = 0;
    for (let i = 1; i < N; i++) {
        if (signal[i] * signal[i - 1] < 0) crossings++;
    }

    const meanAbs = signal.reduce((a, b) => a + Math.abs(b), 0) / N;
    const crest = rms > 0 ? peak / rms : 0;
    const shape = meanAbs > 0 ? rms / meanAbs : 0;

    let sum4 = 0, sum3 = 0;
    for (let i = 0; i < N; i++) {
        sum4 += ((signal[i] - mean) / (std || 1)) ** 4;
        sum3 += ((signal[i] - mean) / (std || 1)) ** 3;
    }

    return { rms, peak, kurtosis: sum4 / N - 3, crest, energy, zcr: crossings / N, shape, skewness: sum3 / N };
}

// ══════════════════════════════════════════════════════════════════════
// UI RENDER UPDATE
// ══════════════════════════════════════════════════════════════════════

function updateUI(samples, features, probs, predictedClass, confidence, healthScore) {
    document.getElementById('windowCount').textContent = state.windowCount;

    const healthPercent = Math.round(healthScore * 100);
    const circumference = 339.292;
    const offset = circumference * (1 - healthScore);
    const progressEl = document.getElementById('healthProgress');
    progressEl.style.strokeDashoffset = offset;

    if (healthScore > 0.8) progressEl.style.stroke = '#22c55e';
    else if (healthScore > 0.5) progressEl.style.stroke = '#f59e0b';
    else progressEl.style.stroke = '#ef4444';

    document.getElementById('healthPercent').textContent = healthPercent + '%';
    document.getElementById('healthLabel').textContent = healthScore > 0.8 ? 'Healthy' : healthScore > 0.5 ? 'Warning' : 'Critical';

    const healthCard = document.getElementById('healthCard');
    healthCard.className = 'card status-card ' + (healthScore > 0.8 ? 'status-healthy' : healthScore > 0.5 ? 'status-warning' : 'status-danger');

    const rmsX = computeRMS(samples.x || []);
    const rmsY = computeRMS(samples.y || []);
    const rmsZ = computeRMS(samples.z || []);
    document.getElementById('rmsX').textContent = rmsX.toFixed(3);
    document.getElementById('rmsY').textContent = rmsY.toFixed(3);
    document.getElementById('rmsZ').textContent = rmsZ.toFixed(3);

    const maxRms = Math.max(rmsX, rmsY, rmsZ, 0.01);
    document.getElementById('rmsBarX').style.width = (rmsX / maxRms * 80 + 5) + '%';
    document.getElementById('rmsBarY').style.width = (rmsY / maxRms * 80 + 5) + '%';
    document.getElementById('rmsBarZ').style.width = (rmsZ / maxRms * 80 + 5) + '%';

    const faultEl = document.getElementById('faultClass');
    faultEl.textContent = FAULT_CLASSES[predictedClass].name;
    faultEl.style.color = FAULT_CLASSES[predictedClass].color;
    faultEl.className = 'fault-class' + (predictedClass !== 0 ? ' fault-active' : '');

    document.getElementById('confidenceFill').style.width = (confidence * 100) + '%';
    document.getElementById('confidenceValue').textContent = (confidence * 100).toFixed(0) + '%';

    probs.forEach((p, i) => {
        const bar = document.getElementById('probBar' + i);
        const val = document.getElementById('probVal' + i);
        if (bar) bar.style.width = (p * 100) + '%';
        if (val) val.textContent = (p * 100).toFixed(0) + '%';
    });

    waveformChart.data.datasets[0].data = samples.x || [];
    waveformChart.data.datasets[1].data = samples.y || [];
    waveformChart.data.datasets[2].data = samples.z || [];
    waveformChart.update('none');

    if (state.fftData.freqs) {
        fftChart.data.labels = state.fftData.freqs;
        fftChart.data.datasets[0].data = state.fftData.magnitudes;
        fftChart.update('none');
    }

    trendChart.data.labels = state.trendData.map((_, i) => i);
    trendChart.data.datasets[0].data = state.trendData;
    trendChart.data.datasets[1].data = state.trendData.map((_, i) => {
        const idx = Math.max(0, state.timeline.length - state.trendData.length + i);
        return state.timeline[idx] === 0 ? 0.1 : 0.3 + Math.random() * 0.3;
    });
    trendChart.update('none');

    const featureEls = {
        rms: features.rms || 0, peak: features.peak || 0, kurtosis: features.kurtosis || 0,
        crest: features.crest || 0, energy: features.energy || 0, zcr: features.zcr || 0,
        shape: features.shape || 0, skewness: features.skewness || 0
    };
    Object.entries(featureEls).forEach(([key, val]) => {
        const el = document.getElementById('feat_' + key);
        if (el) el.textContent = (val || 0).toFixed(3);
    });

    const timeline = document.getElementById('timeline');
    timeline.innerHTML = state.timeline.map(cls =>
        `<div class="timeline-block ${FAULT_CLASSES[cls].css}" title="${FAULT_CLASSES[cls].name}"></div>`
    ).join('');
}

function computeRMS(arr) {
    if (!arr || arr.length === 0) return 0;
    return Math.sqrt(arr.reduce((sum, v) => sum + v * v, 0) / arr.length);
}

function updateUptime() {
    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('uptime').textContent = `${h}:${m}:${s}`;
}
