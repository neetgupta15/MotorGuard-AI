/**
 * ============================================================================
 * Fault Detector — Embedded Feature Extraction & Classification
 * ============================================================================
 *
 * On-device implementation of the fault detection pipeline:
 *   1. Circular buffer management for streaming accelerometer data
 *   2. Time-domain feature extraction (14 features × 3 axes = 42 features)
 *   3. Feature normalization using pre-computed scaler parameters
 *   4. Simple threshold-based classification (placeholder for TFLite)
 *
 * Optimization Notes:
 *   - All computations use single-precision float (Cortex-M7 has FPU)
 *   - Feature extraction is optimized for single-pass computation
 *   - No dynamic memory allocation — all buffers are statically allocated
 *
 * ============================================================================
 */

#include "fault_detector.h"
#include <string.h>
#include <float.h>

/* ─── Static Buffers ───────────────────────────────────────────────────── */

static AccelBuffer_t accel_buffer;
static FeatureVector_t current_features;

/* ─── Fault Class Names ────────────────────────────────────────────────── */

static const char* fault_names[N_CLASSES] = {
    "Normal",
    "Bearing Fault",
    "Rotor Imbalance",
    "Shaft Misalignment",
    "Electrical Fault"
};

/* ─── Pre-computed Scaler Parameters ───────────────────────────────────── */
/* These values should be replaced with actual trained scaler parameters
 * exported from Python using export_model.py → scaler_params.h
 *
 * For now, using placeholder values for demonstration.
 * In production, #include "scaler_params.h" and use those values. */

/* Placeholder scaler — identity transform until real parameters are loaded */
static float scaler_mean[FD_N_FEATURES];   /* Initialized to 0 in Init */
static float scaler_scale[FD_N_FEATURES];  /* Initialized to 1 in Init */

/* ─── Simple Decision Thresholds ───────────────────────────────────────── */
/* These thresholds are used for a rule-based classifier as a fallback.
 * For production deployment, replace with TFLite Micro inference.
 *
 * Key indicators:
 *   - High RMS → Imbalance or severe fault
 *   - High Kurtosis → Bearing fault (impulsive signal)
 *   - High 2× ratio → Misalignment
 *   - Unusual patterns → Electrical fault */

#define THRESHOLD_RMS_HIGH       0.3f    /* g - elevated vibration */
#define THRESHOLD_RMS_SEVERE     0.6f    /* g - severe vibration */
#define THRESHOLD_KURTOSIS       5.0f    /* Excess kurtosis for bearing */
#define THRESHOLD_CREST_FACTOR   4.5f    /* Crest factor for bearing */
#define THRESHOLD_IMBALANCE_1X   0.4f    /* Strong 1× component */

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

void FaultDetector_Init(void)
{
    /* Clear accelerometer buffer */
    memset(&accel_buffer, 0, sizeof(AccelBuffer_t));
    accel_buffer.write_idx = 0;
    accel_buffer.count = 0;
    accel_buffer.window_full = false;

    /* Clear feature vector */
    memset(&current_features, 0, sizeof(FeatureVector_t));

    /* Initialize scaler to identity transform */
    for (int i = 0; i < FD_N_FEATURES; i++) {
        scaler_mean[i] = 0.0f;
        scaler_scale[i] = 1.0f;
    }
}

/* ============================================================================
 * SAMPLE BUFFER MANAGEMENT
 * ============================================================================ */

bool FaultDetector_AddSample(float x, float y, float z)
{
    uint16_t idx = accel_buffer.write_idx;

    accel_buffer.x[idx] = x;
    accel_buffer.y[idx] = y;
    accel_buffer.z[idx] = z;

    accel_buffer.write_idx = (idx + 1) % FD_WINDOW_SIZE;

    if (accel_buffer.count < FD_WINDOW_SIZE) {
        accel_buffer.count++;
    }

    /* Window is full when we've collected WINDOW_SIZE samples */
    if (accel_buffer.count >= FD_WINDOW_SIZE) {
        accel_buffer.window_full = true;
        accel_buffer.count = 0;  /* Reset for next window */
        return true;
    }

    return false;
}

/* ============================================================================
 * STATISTICAL UTILITY FUNCTIONS
 * ============================================================================ */

float compute_rms(const float* data, uint16_t length)
{
    float sum_sq = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum_sq += data[i] * data[i];
    }
    return sqrtf(sum_sq / (float)length);
}

/**
 * Compute mean of a float array.
 */
static float compute_mean(const float* data, uint16_t length)
{
    float sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum += data[i];
    }
    return sum / (float)length;
}

/**
 * Compute standard deviation of a float array.
 */
static float compute_std(const float* data, uint16_t length, float mean)
{
    float sum_sq = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sqrtf(sum_sq / (float)length);
}

/**
 * Compute peak (maximum absolute value) of a float array.
 */
static float compute_peak(const float* data, uint16_t length)
{
    float peak = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    return peak;
}

/**
 * Compute peak-to-peak range.
 */
static float compute_peak_to_peak(const float* data, uint16_t length)
{
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (uint16_t i = 0; i < length; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val - min_val;
}

/**
 * Compute mean of absolute values.
 */
static float compute_mean_abs(const float* data, uint16_t length)
{
    float sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum += fabsf(data[i]);
    }
    return sum / (float)length;
}

/**
 * Compute mean of square roots of absolute values.
 */
static float compute_mean_sqrt_abs(const float* data, uint16_t length)
{
    float sum = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum += sqrtf(fabsf(data[i]));
    }
    return sum / (float)length;
}

float compute_kurtosis(const float* data, uint16_t length)
{
    float mean = compute_mean(data, length);
    float std = compute_std(data, length, mean);

    if (std < 1e-10f) return 0.0f;

    float sum_4th = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float z = (data[i] - mean) / std;
        float z2 = z * z;
        sum_4th += z2 * z2;
    }

    /* Excess kurtosis (subtract 3 for normal distribution baseline) */
    return (sum_4th / (float)length) - 3.0f;
}

float compute_skewness(const float* data, uint16_t length)
{
    float mean = compute_mean(data, length);
    float std = compute_std(data, length, mean);

    if (std < 1e-10f) return 0.0f;

    float sum_3rd = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        float z = (data[i] - mean) / std;
        sum_3rd += z * z * z;
    }

    return sum_3rd / (float)length;
}

/**
 * Compute zero crossing rate.
 */
static float compute_zcr(const float* data, uint16_t length)
{
    uint16_t crossings = 0;
    for (uint16_t i = 1; i < length; i++) {
        if ((data[i] >= 0.0f && data[i-1] < 0.0f) ||
            (data[i] < 0.0f && data[i-1] >= 0.0f)) {
            crossings++;
        }
    }
    return (float)crossings / (float)length;
}

/**
 * Compute energy (sum of squared values normalized by length).
 */
static float compute_energy(const float* data, uint16_t length)
{
    float sum_sq = 0.0f;
    for (uint16_t i = 0; i < length; i++) {
        sum_sq += data[i] * data[i];
    }
    return sum_sq / (float)length;
}

/* ============================================================================
 * FEATURE EXTRACTION — Single Axis
 * ============================================================================
 *
 * Extracts 14 time-domain features from one axis of vibration data:
 *   [0]  Mean
 *   [1]  Std
 *   [2]  RMS
 *   [3]  Peak
 *   [4]  Peak-to-Peak
 *   [5]  Crest Factor
 *   [6]  Shape Factor
 *   [7]  Impulse Factor
 *   [8]  Clearance Factor
 *   [9]  Kurtosis
 *   [10] Skewness
 *   [11] Variance
 *   [12] Energy
 *   [13] Zero Crossing Rate
 */

static void extract_axis_features(const float* data, uint16_t length,
                                   float* features)
{
    /* Basic statistics */
    float mean_val = compute_mean(data, length);
    float std_val = compute_std(data, length, mean_val);
    float rms = compute_rms(data, length);
    float peak = compute_peak(data, length);
    float p2p = compute_peak_to_peak(data, length);
    float mean_abs = compute_mean_abs(data, length);
    float mean_sqrt = compute_mean_sqrt_abs(data, length);

    /* Derived factors */
    float crest_factor = (rms > 1e-10f) ? (peak / rms) : 0.0f;
    float shape_factor = (mean_abs > 1e-10f) ? (rms / mean_abs) : 0.0f;
    float impulse_factor = (mean_abs > 1e-10f) ? (peak / mean_abs) : 0.0f;
    float clearance_factor = (mean_sqrt > 1e-10f) ?
                             (peak / (mean_sqrt * mean_sqrt)) : 0.0f;

    /* Higher-order statistics */
    float kurt = compute_kurtosis(data, length);
    float skew = compute_skewness(data, length);
    float variance = std_val * std_val;
    float energy = compute_energy(data, length);
    float zcr = compute_zcr(data, length);

    /* Store features */
    features[0]  = mean_val;
    features[1]  = std_val;
    features[2]  = rms;
    features[3]  = peak;
    features[4]  = p2p;
    features[5]  = crest_factor;
    features[6]  = shape_factor;
    features[7]  = impulse_factor;
    features[8]  = clearance_factor;
    features[9]  = kurt;
    features[10] = skew;
    features[11] = variance;
    features[12] = energy;
    features[13] = zcr;
}

/* ============================================================================
 * FEATURE EXTRACTION — Full 3-Axis Window
 * ============================================================================ */

void FaultDetector_ExtractFeatures(FeatureVector_t* fv)
{
    /* Extract features for each axis */
    extract_axis_features(accel_buffer.x, FD_WINDOW_SIZE,
                          &fv->features[0 * FD_N_TIME_FEATURES]);  /* X: [0..13] */

    extract_axis_features(accel_buffer.y, FD_WINDOW_SIZE,
                          &fv->features[1 * FD_N_TIME_FEATURES]);  /* Y: [14..27] */

    extract_axis_features(accel_buffer.z, FD_WINDOW_SIZE,
                          &fv->features[2 * FD_N_TIME_FEATURES]);  /* Z: [28..41] */

    /* Apply feature scaling: scaled = (x - mean) / scale */
    for (int i = 0; i < FD_N_FEATURES; i++) {
        fv->features_scaled[i] =
            (fv->features[i] - scaler_mean[i]) / scaler_scale[i];
    }
}

/* ============================================================================
 * FAULT CLASSIFICATION
 * ============================================================================
 *
 * This implements a rule-based classifier as a deployment baseline.
 *
 * For production:
 *   Replace this function body with TFLite Micro inference:
 *   1. Copy features_scaled to TFLite input tensor
 *   2. Call interpreter->Invoke()
 *   3. Read output tensor probabilities
 *
 * The rule-based approach uses key vibration indicators:
 *   - RMS level → overall vibration severity
 *   - Kurtosis → impulsiveness (bearing faults)
 *   - Crest Factor → peak/RMS ratio (bearing faults)
 *   - Axial RMS → misalignment indicator
 *   - Frequency content clues via ZCR
 * ============================================================================ */

void FaultDetector_Classify(const FeatureVector_t* fv, DetectionResult_t* result)
{
    /* Extract key features for decision logic */
    /* X-axis features */
    float x_rms = fv->features[2];
    float x_peak = fv->features[3];
    float x_crest = fv->features[5];
    float x_kurtosis = fv->features[9];
    float x_energy = fv->features[12];

    /* Y-axis features */
    float y_rms = fv->features[FD_N_TIME_FEATURES + 2];
    float y_kurtosis = fv->features[FD_N_TIME_FEATURES + 9];

    /* Z-axis features (axial) */
    float z_rms = fv->features[2 * FD_N_TIME_FEATURES + 2];
    float z_energy = fv->features[2 * FD_N_TIME_FEATURES + 12];

    /* Derived metrics */
    float radial_rms = sqrtf(x_rms * x_rms + y_rms * y_rms);
    float axial_ratio = (radial_rms > 1e-6f) ? (z_rms / radial_rms) : 0.0f;
    float avg_kurtosis = (x_kurtosis + y_kurtosis) / 2.0f;

    /* Initialize class probabilities */
    float probs[N_CLASSES] = {0.0f};

    /* ─── Rule-Based Classification Logic ──────────────────────────── */

    /* Normal: Low RMS, low kurtosis, balanced energy */
    if (radial_rms < THRESHOLD_RMS_HIGH) {
        probs[FAULT_NORMAL] = 0.8f;
    }

    /* Bearing Fault: High kurtosis (impulsive), elevated crest factor */
    if (avg_kurtosis > THRESHOLD_KURTOSIS || x_crest > THRESHOLD_CREST_FACTOR) {
        probs[FAULT_BEARING] = 0.3f + 0.4f * (avg_kurtosis / 10.0f);
        if (probs[FAULT_BEARING] > 0.95f) probs[FAULT_BEARING] = 0.95f;
    }

    /* Rotor Imbalance: High RMS with dominant 1× (detected via high radial RMS) */
    if (radial_rms > THRESHOLD_IMBALANCE_1X && avg_kurtosis < 4.0f) {
        probs[FAULT_IMBALANCE] = 0.3f + 0.5f * (radial_rms / 1.0f);
        if (probs[FAULT_IMBALANCE] > 0.95f) probs[FAULT_IMBALANCE] = 0.95f;
    }

    /* Shaft Misalignment: High axial ratio (strong Z-axis relative to radial) */
    if (axial_ratio > 0.3f && radial_rms > THRESHOLD_RMS_HIGH * 0.8f) {
        probs[FAULT_MISALIGNMENT] = 0.3f + 0.5f * axial_ratio;
        if (probs[FAULT_MISALIGNMENT] > 0.95f) probs[FAULT_MISALIGNMENT] = 0.95f;
    }

    /* Electrical Fault: Moderate RMS with specific ZCR patterns */
    float x_zcr = fv->features[13];
    if (radial_rms > THRESHOLD_RMS_HIGH * 0.6f && x_zcr > 0.3f &&
        avg_kurtosis < 4.0f && axial_ratio < 0.25f) {
        probs[FAULT_ELECTRICAL] = 0.3f + 0.3f * x_zcr;
        if (probs[FAULT_ELECTRICAL] > 0.95f) probs[FAULT_ELECTRICAL] = 0.95f;
    }

    /* ─── Normalize Probabilities ──────────────────────────────────── */
    float sum_probs = 0.0f;
    for (int i = 0; i < N_CLASSES; i++) {
        sum_probs += probs[i];
    }

    if (sum_probs > 0.0f) {
        for (int i = 0; i < N_CLASSES; i++) {
            probs[i] /= sum_probs;
        }
    } else {
        probs[FAULT_NORMAL] = 1.0f;  /* Default to normal if no evidence */
    }

    /* ─── Find Predicted Class ─────────────────────────────────────── */
    float max_prob = 0.0f;
    FaultClass_t predicted = FAULT_NORMAL;

    for (int i = 0; i < N_CLASSES; i++) {
        result->class_probabilities[i] = probs[i];
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = (FaultClass_t)i;
        }
    }

    result->predicted_class = predicted;
    result->confidence = max_prob;
}

/* ============================================================================
 * COMPLETE DETECTION PIPELINE
 * ============================================================================ */

bool FaultDetector_RunDetection(DetectionResult_t* result)
{
    /* Measure inference time using DWT cycle counter */
    uint32_t start_cycles = DWT->CYCCNT;

    /* Step 1: Extract features */
    FaultDetector_ExtractFeatures(&current_features);

    /* Step 2: Classify */
    FaultDetector_Classify(&current_features, result);

    /* Calculate inference time in microseconds */
    uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;
    result->inference_time_us = elapsed_cycles / (SystemCoreClock / 1000000);

    /* Return true if a fault is detected with sufficient confidence */
    bool fault_detected = (result->predicted_class != FAULT_NORMAL) &&
                          (result->confidence >= FAULT_THRESHOLD);

    return fault_detected;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

const char* FaultDetector_GetFaultName(FaultClass_t fault_class)
{
    if (fault_class < N_CLASSES) {
        return fault_names[fault_class];
    }
    return "Unknown";
}
