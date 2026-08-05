/**
 * ============================================================================
 * LIS3DH Accelerometer Driver — Implementation
 * ============================================================================
 *
 * I2C driver for the LIS3DH 3-axis MEMS accelerometer.
 *
 * Initialization Sequence:
 *   1. Verify WHO_AM_I register (0x0F) = 0x33
 *   2. Set CTRL_REG1: ODR=400Hz, enable X/Y/Z axes
 *   3. Set CTRL_REG4: Full-scale ±4g, high-resolution mode
 *   4. Configure FIFO if enabled
 *
 * Data Reading:
 *   - Read 6 bytes from OUT_X_L (0x28) with auto-increment
 *   - Convert raw 16-bit values to g using sensitivity factor
 *   - For ±4g range: sensitivity = 2 mg/digit (12-bit left-justified)
 *
 * ============================================================================
 */

#include "lis3dh.h"
#include <string.h>

/* ─── Internal Helper Functions ────────────────────────────────────────── */

/**
 * Write a single byte to a LIS3DH register.
 */
static HAL_StatusTypeDef LIS3DH_WriteReg(LIS3DH_Config_t* config,
                                          uint8_t reg, uint8_t value)
{
    return HAL_I2C_Mem_Write(config->hi2c, config->i2c_addr,
                             reg, I2C_MEMADD_SIZE_8BIT,
                             &value, 1, HAL_MAX_DELAY);
}

/**
 * Read a single byte from a LIS3DH register.
 */
static HAL_StatusTypeDef LIS3DH_ReadReg(LIS3DH_Config_t* config,
                                         uint8_t reg, uint8_t* value)
{
    return HAL_I2C_Mem_Read(config->hi2c, config->i2c_addr,
                            reg, I2C_MEMADD_SIZE_8BIT,
                            value, 1, HAL_MAX_DELAY);
}

/**
 * Read multiple bytes from consecutive LIS3DH registers.
 * Sets MSB of register address for auto-increment.
 */
static HAL_StatusTypeDef LIS3DH_ReadRegs(LIS3DH_Config_t* config,
                                          uint8_t reg, uint8_t* data,
                                          uint16_t length)
{
    /* Set MSB for auto-increment in multi-byte read */
    uint8_t reg_auto = reg | 0x80;
    return HAL_I2C_Mem_Read(config->hi2c, config->i2c_addr,
                            reg_auto, I2C_MEMADD_SIZE_8BIT,
                            data, length, HAL_MAX_DELAY);
}

/**
 * Get sensitivity value (mg/digit) based on full-scale range.
 */
static float LIS3DH_GetSensitivity(LIS3DH_FullScale_t fs)
{
    switch (fs) {
        case LIS3DH_FS_2G:  return 1.0f;    /* 1 mg/digit at ±2g */
        case LIS3DH_FS_4G:  return 2.0f;    /* 2 mg/digit at ±4g */
        case LIS3DH_FS_8G:  return 4.0f;    /* 4 mg/digit at ±8g */
        case LIS3DH_FS_16G: return 12.0f;   /* 12 mg/digit at ±16g */
        default:            return 1.0f;
    }
}

/* ============================================================================
 * PUBLIC API IMPLEMENTATION
 * ============================================================================ */

HAL_StatusTypeDef LIS3DH_Init(LIS3DH_Config_t* config)
{
    HAL_StatusTypeDef status;

    /* Verify device identity */
    if (!LIS3DH_VerifyID(config)) {
        return HAL_ERROR;
    }

    /* Store sensitivity for data conversion */
    config->sensitivity = LIS3DH_GetSensitivity(config->full_scale);

    /* ─── CTRL_REG1 (0x20) ─────────────────────────────────────────
     * [7:4] ODR    = 0x70 (400 Hz)
     * [3]   LPen   = 0 (Normal mode, not low-power)
     * [2]   Zen    = 1 (Z-axis enabled)
     * [1]   Yen    = 1 (Y-axis enabled)
     * [0]   Xen    = 1 (X-axis enabled)
     */
    uint8_t ctrl_reg1 = config->odr | 0x07;  /* ODR + enable all axes */
    status = LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG1, ctrl_reg1);
    if (status != HAL_OK) return status;

    /* ─── CTRL_REG2 (0x21) ─────────────────────────────────────────
     * [7:6] HPM    = 00 (Normal mode, reset by reading REF)
     * [5:4] HPCF   = 00 (Highest cutoff)
     * [3]   FDS    = 0 (Filter bypassed)
     * [2]   HPCLICK= 0
     * [1]   HP_IA2 = 0
     * [0]   HP_IA1 = 0
     * High-pass filter disabled for vibration monitoring.
     */
    status = LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG2, 0x00);
    if (status != HAL_OK) return status;

    /* ─── CTRL_REG3 (0x22) ─────────────────────────────────────────
     * Enable Data Ready interrupt on INT1 (for optional interrupt-driven reading)
     * [4] I1_DRDY1 = 1
     */
    status = LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG3, 0x10);
    if (status != HAL_OK) return status;

    /* ─── CTRL_REG4 (0x23) ─────────────────────────────────────────
     * [7]   BDU    = 1 (Block Data Update — prevent reading partial samples)
     * [6]   BLE    = 0 (Little-endian)
     * [5:4] FS     = config->full_scale
     * [3]   HR     = 1 (High-resolution mode, 12-bit)
     * [1:0] ST     = 00 (Self-test disabled)
     */
    uint8_t ctrl_reg4 = 0x88 | config->full_scale;  /* BDU=1, HR=1, FS=config */
    status = LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG4, ctrl_reg4);
    if (status != HAL_OK) return status;

    /* ─── CTRL_REG5 (0x24) ─────────────────────────────────────────
     * [6] FIFO_EN = config->fifo_enabled
     * [3] LIR_INT1 = 1 (Latch interrupt on INT1)
     */
    uint8_t ctrl_reg5 = 0x08;  /* Latch interrupt */
    if (config->fifo_enabled) {
        ctrl_reg5 |= 0x40;     /* Enable FIFO */
    }
    status = LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG5, ctrl_reg5);
    if (status != HAL_OK) return status;

    /* ─── FIFO Configuration ──────────────────────────────────────── */
    if (config->fifo_enabled) {
        /* FIFO_CTRL (0x2E): Stream mode, watermark = 25 */
        uint8_t fifo_ctrl = 0x80 | 25;  /* Stream mode + threshold */
        status = LIS3DH_WriteReg(config, LIS3DH_REG_FIFO_CTRL, fifo_ctrl);
        if (status != HAL_OK) return status;
    }

    return HAL_OK;
}

bool LIS3DH_VerifyID(LIS3DH_Config_t* config)
{
    uint8_t who_am_i = 0;
    HAL_StatusTypeDef status;

    status = LIS3DH_ReadReg(config, LIS3DH_REG_WHO_AM_I, &who_am_i);

    if (status != HAL_OK) {
        return false;
    }

    return (who_am_i == LIS3DH_WHO_AM_I_VALUE);
}

HAL_StatusTypeDef LIS3DH_ReadAccel(LIS3DH_Config_t* config, LIS3DH_Data_t* data)
{
    uint8_t raw_data[6];
    HAL_StatusTypeDef status;

    /* Read 6 bytes starting from OUT_X_L (0x28) with auto-increment */
    status = LIS3DH_ReadRegs(config, LIS3DH_REG_OUT_X_L, raw_data, 6);
    if (status != HAL_OK) {
        return status;
    }

    /* Combine bytes into 16-bit signed values (little-endian) */
    data->x_raw = (int16_t)(raw_data[1] << 8 | raw_data[0]);
    data->y_raw = (int16_t)(raw_data[3] << 8 | raw_data[2]);
    data->z_raw = (int16_t)(raw_data[5] << 8 | raw_data[4]);

    /* In high-resolution mode, data is left-justified in 16 bits.
     * The actual data is 12 bits, so shift right by 4. */
    int16_t x_12bit = data->x_raw >> 4;
    int16_t y_12bit = data->y_raw >> 4;
    int16_t z_12bit = data->z_raw >> 4;

    /* Convert to g: value_g = raw_12bit × sensitivity / 1000 */
    data->x_g = (float)x_12bit * config->sensitivity / 1000.0f;
    data->y_g = (float)y_12bit * config->sensitivity / 1000.0f;
    data->z_g = (float)z_12bit * config->sensitivity / 1000.0f;

    return HAL_OK;
}

bool LIS3DH_DataReady(LIS3DH_Config_t* config)
{
    uint8_t status_reg = 0;
    LIS3DH_ReadReg(config, LIS3DH_REG_STATUS, &status_reg);

    /* Bit 3 (ZYXDA) = 1 when new data available on all axes */
    return (status_reg & 0x08) != 0;
}

uint8_t LIS3DH_ReadFIFO(LIS3DH_Config_t* config, LIS3DH_Data_t* buffer,
                         uint8_t max_count)
{
    uint8_t fifo_src = 0;
    uint8_t count;

    /* Read FIFO status to get number of samples */
    LIS3DH_ReadReg(config, LIS3DH_REG_FIFO_SRC, &fifo_src);

    /* Bits [4:0] = number of unread samples */
    count = fifo_src & 0x1F;
    if (count > max_count) {
        count = max_count;
    }

    /* Read each sample from FIFO */
    for (uint8_t i = 0; i < count; i++) {
        LIS3DH_ReadAccel(config, &buffer[i]);
    }

    return count;
}

void LIS3DH_SetODR(LIS3DH_Config_t* config, LIS3DH_ODR_t odr)
{
    uint8_t ctrl_reg1 = odr | 0x07;  /* New ODR + keep axes enabled */
    LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG1, ctrl_reg1);
    config->odr = odr;
}

void LIS3DH_SetFullScale(LIS3DH_Config_t* config, LIS3DH_FullScale_t full_scale)
{
    uint8_t ctrl_reg4 = 0x88 | full_scale;  /* BDU=1, HR=1, new FS */
    LIS3DH_WriteReg(config, LIS3DH_REG_CTRL_REG4, ctrl_reg4);
    config->full_scale = full_scale;
    config->sensitivity = LIS3DH_GetSensitivity(full_scale);
}
