/**
 * ============================================================================
 * Fault Detector — Embedded Feature Extraction & Classification Header
 * ============================================================================
 *
 * Implements on-device feature extraction and fault classification for
 * real-time vibration monitoring on STM32 Cortex-M7.
 *
 * Pipeline:
 *   1. Collect 1024 samples (3-axis) at 400 Hz = 2.56s window
 *   2. Extract time-domain features (RMS, Kurtosis, Crest Factor, etc.)
 *   3. Normalize features using pre-computed scaler parameters
 *   4. Run ML inference → fault classification
 *   5. Trigger alert if fault detected with confidence > threshold
 *
 * Memory Budget (estimated):
 *   - Sample buffer: 1024 × 3 × 4 bytes = 12 KB
 *   - Feature vector: 96 × 4 bytes = 384 bytes
 *   - Model data: ~50–200 KB (depends on model)
 * ============================================================================
 */

#ifndef FAULT_DETECTOR_H
#define FAULT_DETECTOR_H

#include "main.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* ─── Configuration ────────────────────────────────────────────────────── */

#define FD_WINDOW_SIZE      1024    /* Samples per analysis window */
#define FD_N_AXES           3       /* X, Y, Z axes */
#define FD_N_TIME_FEATURES  14      /* Time-domain features per axis */
#define FD_N_FEATURES       (FD_N_TIME_FEATURES * FD_N_AXES)  /* 42 for time only */

/* Use simplified feature set for embedded (time-domain only = 42 features)
 * Full 96-feature set requires FFT and wavelet, which is computationally
 * expensive on MCU. Time-domain features alone provide ~90% accuracy. */

/* ─── Data Types ───────────────────────────────────────────────────────── */

/**
 * Circular buffer for streaming accelerometer data.
 */
typedef struct {
    float x[FD_WINDOW_SIZE];
    float y[FD_WINDOW_SIZE];
    float z[FD_WINDOW_SIZE];
    uint16_t write_idx;
    uint16_t count;
    bool window_full;
} AccelBuffer_t;

/**
 * Feature vector extracted from one analysis window.
 */
typedef struct {
    float features[FD_N_FEATURES];
    float features_scaled[FD_N_FEATURES];
} FeatureVector_t;

/**
 * Fault detection result.
 */
typedef struct {
    FaultClass_t predicted_class;
    float confidence;
    float class_probabilities[N_CLASSES];
    uint32_t inference_time_us;
} DetectionResult_t;

/* ─── Function Prototypes ──────────────────────────────────────────────── */

/**
 * Initialize the fault detector module.
 * Clears buffers and prepares for data collection.
 */
void FaultDetector_Init(void);

/**
 * Add a new accelerometer sample to the buffer.
 *
 * @param x  X-axis acceleration in g
 * @param y  Y-axis acceleration in g
 * @param z  Z-axis acceleration in g
 * @return true if analysis window is now full
 */
bool FaultDetector_AddSample(float x, float y, float z);

/**
 * Extract features from the current analysis window.
 * Call this when FaultDetector_AddSample returns true.
 *
 * @param fv  Pointer to FeatureVector structure to fill
 */
void FaultDetector_ExtractFeatures(FeatureVector_t* fv);

/**
 * Run fault classification on extracted features.
 *
 * @param fv      Pointer to feature vector (input)
 * @param result  Pointer to detection result (output)
 */
void FaultDetector_Classify(const FeatureVector_t* fv, DetectionResult_t* result);

/**
 * Complete detection pipeline: extract features + classify.
 *
 * @param result  Pointer to detection result (output)
 * @return true if a fault was detected
 */
bool FaultDetector_RunDetection(DetectionResult_t* result);

/**
 * Get the name string for a fault class.
 *
 * @param fault_class  Fault class enum value
 * @return Pointer to null-terminated string
 */
const char* FaultDetector_GetFaultName(FaultClass_t fault_class);

/* ─── Utility Functions ────────────────────────────────────────────────── */

/**
 * Compute RMS of a float array.
 */
float compute_rms(const float* data, uint16_t length);

/**
 * Compute kurtosis of a float array (excess kurtosis).
 */
float compute_kurtosis(const float* data, uint16_t length);

/**
 * Compute skewness of a float array.
 */
float compute_skewness(const float* data, uint16_t length);

#endif /* FAULT_DETECTOR_H */
