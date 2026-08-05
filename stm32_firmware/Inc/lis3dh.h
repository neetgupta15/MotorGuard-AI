/**
 * ============================================================================
 * LIS3DH Accelerometer Driver — Header
 * ============================================================================
 *
 * I2C driver for the LIS3DH 3-axis accelerometer sensor.
 * Configured for vibration monitoring with high ODR and FIFO support.
 *
 * Key Features:
 *   - I2C communication (100/400 kHz)
 *   - Configurable Output Data Rate (1 Hz to 5.376 kHz)
 *   - Selectable full-scale range (±2g, ±4g, ±8g, ±16g)
 *   - 32-level FIFO buffer
 *   - Interrupt-driven data ready notification
 *
 * Reference: ST AN3308 — LIS3DH Application Note
 * ============================================================================
 */

#ifndef LIS3DH_H
#define LIS3DH_H

#include "stm32h7xx_hal.h"
#include <stdint.h>
#include <stdbool.h>

/* ─── I2C Address ──────────────────────────────────────────────────────── */

/* LIS3DH I2C address: SDO/SA0 pin determines LSB */
#define LIS3DH_ADDR_LOW     (0x18 << 1)   /* SA0 = GND: 0x18 */
#define LIS3DH_ADDR_HIGH    (0x19 << 1)   /* SA0 = VCC: 0x19 */
#define LIS3DH_I2C_ADDR     LIS3DH_ADDR_LOW

/* ─── Register Map ─────────────────────────────────────────────────────── */

#define LIS3DH_REG_STATUS_AUX   0x07
#define LIS3DH_REG_OUT_ADC1_L   0x08
#define LIS3DH_REG_OUT_ADC1_H   0x09
#define LIS3DH_REG_WHO_AM_I     0x0F    /* Expected value: 0x33 */
#define LIS3DH_REG_CTRL_REG0    0x1E
#define LIS3DH_REG_TEMP_CFG     0x1F
#define LIS3DH_REG_CTRL_REG1    0x20    /* ODR, axes enable */
#define LIS3DH_REG_CTRL_REG2    0x21    /* HPF configuration */
#define LIS3DH_REG_CTRL_REG3    0x22    /* Interrupt config */
#define LIS3DH_REG_CTRL_REG4    0x23    /* Full-scale, resolution */
#define LIS3DH_REG_CTRL_REG5    0x24    /* FIFO enable, latch */
#define LIS3DH_REG_CTRL_REG6    0x25    /* INT2 configuration */
#define LIS3DH_REG_REFERENCE    0x26
#define LIS3DH_REG_STATUS       0x27    /* Data ready status */
#define LIS3DH_REG_OUT_X_L      0x28    /* X-axis low byte */
#define LIS3DH_REG_OUT_X_H      0x29
#define LIS3DH_REG_OUT_Y_L      0x2A
#define LIS3DH_REG_OUT_Y_H      0x2B
#define LIS3DH_REG_OUT_Z_L      0x2C
#define LIS3DH_REG_OUT_Z_H      0x2D
#define LIS3DH_REG_FIFO_CTRL    0x2E
#define LIS3DH_REG_FIFO_SRC     0x2F
#define LIS3DH_REG_INT1_CFG     0x30
#define LIS3DH_REG_INT1_SRC     0x31
#define LIS3DH_REG_INT1_THS     0x32
#define LIS3DH_REG_INT1_DUR     0x33

/* ─── WHO_AM_I Expected Value ──────────────────────────────────────────── */
#define LIS3DH_WHO_AM_I_VALUE   0x33

/* ─── ODR (Output Data Rate) Settings ──────────────────────────────────── */

typedef enum {
    LIS3DH_ODR_POWER_DOWN  = 0x00,
    LIS3DH_ODR_1_HZ        = 0x10,
    LIS3DH_ODR_10_HZ       = 0x20,
    LIS3DH_ODR_25_HZ       = 0x30,
    LIS3DH_ODR_50_HZ       = 0x40,
    LIS3DH_ODR_100_HZ      = 0x50,
    LIS3DH_ODR_200_HZ      = 0x60,
    LIS3DH_ODR_400_HZ      = 0x70,    /* Recommended for vibration */
    LIS3DH_ODR_1600_HZ     = 0x80,    /* Low-power only */
    LIS3DH_ODR_5376_HZ     = 0x90     /* Low-power / 1344 Hz normal */
} LIS3DH_ODR_t;

/* ─── Full-Scale Range ─────────────────────────────────────────────────── */

typedef enum {
    LIS3DH_FS_2G   = 0x00,            /* ±2g  (sensitivity: 1 mg/digit) */
    LIS3DH_FS_4G   = 0x10,            /* ±4g  (sensitivity: 2 mg/digit) */
    LIS3DH_FS_8G   = 0x20,            /* ±8g  (sensitivity: 4 mg/digit) */
    LIS3DH_FS_16G  = 0x30             /* ±16g (sensitivity: 12 mg/digit) */
} LIS3DH_FullScale_t;

/* ─── Accelerometer Data Structure ─────────────────────────────────────── */

typedef struct {
    int16_t x_raw;      /* Raw 16-bit X-axis reading */
    int16_t y_raw;      /* Raw 16-bit Y-axis reading */
    int16_t z_raw;      /* Raw 16-bit Z-axis reading */
    float x_g;          /* X-axis in g */
    float y_g;          /* Y-axis in g */
    float z_g;          /* Z-axis in g */
} LIS3DH_Data_t;

/* ─── Driver Configuration ─────────────────────────────────────────────── */

typedef struct {
    I2C_HandleTypeDef* hi2c;           /* I2C peripheral handle */
    uint8_t i2c_addr;                  /* I2C device address */
    LIS3DH_ODR_t odr;                 /* Output data rate */
    LIS3DH_FullScale_t full_scale;    /* Full-scale range */
    float sensitivity;                 /* mg per digit */
    bool fifo_enabled;                 /* FIFO buffer enable */
} LIS3DH_Config_t;

/* ─── Function Prototypes ──────────────────────────────────────────────── */

/**
 * Initialize the LIS3DH sensor.
 *
 * @param config  Pointer to driver configuration
 * @return HAL_OK on success, HAL_ERROR on failure
 */
HAL_StatusTypeDef LIS3DH_Init(LIS3DH_Config_t* config);

/**
 * Verify sensor identity by reading WHO_AM_I register.
 *
 * @param config  Pointer to driver configuration
 * @return true if WHO_AM_I == 0x33
 */
bool LIS3DH_VerifyID(LIS3DH_Config_t* config);

/**
 * Read a single 3-axis acceleration sample.
 *
 * @param config  Pointer to driver configuration
 * @param data    Pointer to data structure to fill
 * @return HAL_OK on success
 */
HAL_StatusTypeDef LIS3DH_ReadAccel(LIS3DH_Config_t* config, LIS3DH_Data_t* data);

/**
 * Check if new data is available.
 *
 * @param config  Pointer to driver configuration
 * @return true if data ready flag is set
 */
bool LIS3DH_DataReady(LIS3DH_Config_t* config);

/**
 * Read multiple samples from FIFO buffer.
 *
 * @param config    Pointer to driver configuration
 * @param buffer    Array of data structures to fill
 * @param max_count Maximum number of samples to read
 * @return Number of samples actually read
 */
uint8_t LIS3DH_ReadFIFO(LIS3DH_Config_t* config, LIS3DH_Data_t* buffer,
                         uint8_t max_count);

/**
 * Set the output data rate.
 *
 * @param config  Pointer to driver configuration
 * @param odr     Desired output data rate
 */
void LIS3DH_SetODR(LIS3DH_Config_t* config, LIS3DH_ODR_t odr);

/**
 * Set the full-scale range.
 *
 * @param config      Pointer to driver configuration
 * @param full_scale  Desired full-scale range
 */
void LIS3DH_SetFullScale(LIS3DH_Config_t* config, LIS3DH_FullScale_t full_scale);

#endif /* LIS3DH_H */
