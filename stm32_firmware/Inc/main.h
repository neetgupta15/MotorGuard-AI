/**
 * ============================================================================
 * Motor Fault Detection System — STM32 NUCLEO-H743ZI2 Firmware
 * ============================================================================
 *
 * Main application header file.
 *
 * Target Board: STM32 NUCLEO-H743ZI2 (ARM Cortex-M7 @ 480 MHz)
 * Sensor:       LIS3DH 3-axis Accelerometer (I2C)
 *
 * Description:
 *   Real-time vibration monitoring and fault detection for electric motors.
 *   Reads 3-axis accelerometer data, extracts features, runs ML inference,
 *   and triggers alerts (LED + buzzer) when faults are detected.
 *
 * Pin Mapping:
 *   PB8  - I2C1_SCL  (LIS3DH SCL)
 *   PB9  - I2C1_SDA  (LIS3DH SDA)
 *   PB0  - LED Green  (Normal status)
 *   PB7  - LED Blue   (Processing indicator)
 *   PB14 - LED Red    (Fault alert)
 *   PA0  - Buzzer     (Fault alarm)
 *
 * Author: Motor Fault Detection Project
 * ============================================================================
 */

#ifndef MAIN_H
#define MAIN_H

#include "stm32h7xx_hal.h"
#include <stdint.h>
#include <stdbool.h>

/* ─── System Configuration ─────────────────────────────────────────────── */

#define SYSTEM_CLOCK_MHZ        480     /* Core clock frequency */
#define SAMPLING_FREQ_HZ        400     /* Accelerometer ODR */
#define WINDOW_SIZE             1024    /* Samples per analysis window */
#define N_FEATURES              96      /* Feature vector dimension */
#define N_CLASSES               5       /* Number of fault classes */
#define FAULT_THRESHOLD         0.7f    /* Confidence threshold for alert */

/* ─── Fault Class Definitions ──────────────────────────────────────────── */

typedef enum {
    FAULT_NORMAL        = 0,
    FAULT_BEARING       = 1,
    FAULT_IMBALANCE     = 2,
    FAULT_MISALIGNMENT  = 3,
    FAULT_ELECTRICAL    = 4
} FaultClass_t;

/* ─── System State ─────────────────────────────────────────────────────── */

typedef struct {
    FaultClass_t current_fault;
    float confidence;
    float severity_estimate;
    uint32_t sample_count;
    uint32_t fault_count;
    uint32_t window_count;
    bool fault_active;
    bool data_ready;
} SystemState_t;

/* ─── LED Pin Definitions (NUCLEO-H743ZI2) ─────────────────────────────── */

#define LED_GREEN_PORT      GPIOB
#define LED_GREEN_PIN       GPIO_PIN_0
#define LED_BLUE_PORT       GPIOB
#define LED_BLUE_PIN        GPIO_PIN_7
#define LED_RED_PORT        GPIOB
#define LED_RED_PIN         GPIO_PIN_14

#define BUZZER_PORT         GPIOA
#define BUZZER_PIN          GPIO_PIN_0

/* ─── Function Prototypes ──────────────────────────────────────────────── */

void SystemClock_Config(void);
void GPIO_Init(void);
void I2C1_Init(void);
void Timer_Init(void);
void Error_Handler(void);

void LED_SetStatus(FaultClass_t fault);
void Buzzer_Alert(bool enable);

#endif /* MAIN_H */
