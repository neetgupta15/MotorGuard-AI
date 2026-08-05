/**
 * ============================================================================
 * Motor Fault Detection System — Main Application
 * ============================================================================
 *
 * Target: STM32 NUCLEO-H743ZI2 (Cortex-M7 @ 480 MHz)
 * Sensor: LIS3DH 3-axis Accelerometer (I2C1)
 *
 * Real-time vibration monitoring loop:
 *   1. Read accelerometer data at 400 Hz (timer interrupt)
 *   2. Fill analysis window (1024 samples = 2.56 seconds)
 *   3. Extract time-domain features
 *   4. Run fault classification
 *   5. Update LED status and trigger buzzer for faults
 *   6. Send results via UART for monitoring
 *
 * ============================================================================
 */

#include "main.h"
#include "lis3dh.h"
#include "fault_detector.h"
#include <stdio.h>
#include <string.h>

/* ─── Global Handles ───────────────────────────────────────────────────── */

I2C_HandleTypeDef hi2c1;
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart3;

/* ─── Global State ─────────────────────────────────────────────────────── */

static LIS3DH_Config_t accel_config;
static SystemState_t system_state;
static volatile bool sample_timer_flag = false;

/* ─── UART Print Buffer ────────────────────────────────────────────────── */

static char uart_buf[256];

/* ─── Function Prototypes ──────────────────────────────────────────────── */

static void UART3_Init(void);
static void UART_Print(const char* msg);
static void System_PrintBanner(void);
static void System_ProcessWindow(void);

/* ============================================================================
 * MAIN APPLICATION
 * ============================================================================ */

int main(void)
{
    /* ─── HAL & System Initialization ──────────────────────────────── */
    HAL_Init();
    SystemClock_Config();
    GPIO_Init();
    I2C1_Init();
    UART3_Init();
    Timer_Init();

    /* ─── Print startup banner ─────────────────────────────────────── */
    System_PrintBanner();

    /* ─── Initialize LIS3DH Accelerometer ──────────────────────────── */
    accel_config.hi2c = &hi2c1;
    accel_config.i2c_addr = LIS3DH_I2C_ADDR;
    accel_config.odr = LIS3DH_ODR_400_HZ;
    accel_config.full_scale = LIS3DH_FS_4G;
    accel_config.fifo_enabled = false;

    UART_Print("[INIT] Initializing LIS3DH accelerometer...\r\n");

    if (LIS3DH_Init(&accel_config) != HAL_OK) {
        UART_Print("[ERROR] LIS3DH initialization failed!\r\n");
        LED_SetStatus(FAULT_ELECTRICAL);  /* Red LED for error */
        Error_Handler();
    }

    if (!LIS3DH_VerifyID(&accel_config)) {
        UART_Print("[ERROR] LIS3DH WHO_AM_I verification failed!\r\n");
        Error_Handler();
    }

    UART_Print("[INIT] LIS3DH initialized successfully (400 Hz, +/-4g)\r\n");

    /* ─── Initialize Fault Detector ────────────────────────────────── */
    FaultDetector_Init();
    UART_Print("[INIT] Fault detector initialized\r\n");

    /* ─── Initialize System State ──────────────────────────────────── */
    memset(&system_state, 0, sizeof(SystemState_t));
    system_state.current_fault = FAULT_NORMAL;

    UART_Print("[INIT] System ready. Starting monitoring...\r\n\r\n");
    LED_SetStatus(FAULT_NORMAL);

    /* ─── Start Sampling Timer ─────────────────────────────────────── */
    HAL_TIM_Base_Start_IT(&htim2);

    /* ============================================================== */
    /* MAIN LOOP — Real-Time Vibration Monitoring                     */
    /* ============================================================== */

    while (1)
    {
        /* Wait for timer interrupt (400 Hz) */
        if (sample_timer_flag)
        {
            sample_timer_flag = false;

            /* ─── Read Accelerometer ───────────────────────────────── */
            LIS3DH_Data_t accel_data;
            if (LIS3DH_ReadAccel(&accel_config, &accel_data) == HAL_OK)
            {
                /* Toggle blue LED during data acquisition */
                HAL_GPIO_TogglePin(LED_BLUE_PORT, LED_BLUE_PIN);

                /* Add sample to analysis buffer */
                bool window_full = FaultDetector_AddSample(
                    accel_data.x_g,
                    accel_data.y_g,
                    accel_data.z_g
                );

                system_state.sample_count++;

                /* ─── Process Complete Window ──────────────────────── */
                if (window_full)
                {
                    System_ProcessWindow();
                }
            }
        }

        /* Low-power wait for interrupt */
        __WFI();
    }
}

/* ============================================================================
 * WINDOW PROCESSING — Feature Extraction + Classification
 * ============================================================================ */

static void System_ProcessWindow(void)
{
    DetectionResult_t result;
    bool fault_detected;

    system_state.window_count++;

    /* Run full detection pipeline */
    fault_detected = FaultDetector_RunDetection(&result);

    /* Update system state */
    system_state.current_fault = result.predicted_class;
    system_state.confidence = result.confidence;
    system_state.fault_active = fault_detected;

    if (fault_detected) {
        system_state.fault_count++;
    }

    /* ─── Update LED Indicators ────────────────────────────────────── */
    LED_SetStatus(result.predicted_class);

    /* ─── Trigger Buzzer for Faults ────────────────────────────────── */
    Buzzer_Alert(fault_detected);

    /* ─── Send Results via UART ────────────────────────────────────── */
    snprintf(uart_buf, sizeof(uart_buf),
        "[W%04lu] Class: %s | Confidence: %.1f%% | Inference: %lu us\r\n",
        system_state.window_count,
        FaultDetector_GetFaultName(result.predicted_class),
        result.confidence * 100.0f,
        result.inference_time_us
    );
    UART_Print(uart_buf);

    /* Print class probabilities for monitoring dashboard */
    snprintf(uart_buf, sizeof(uart_buf),
        "  Probabilities: N=%.2f B=%.2f I=%.2f M=%.2f E=%.2f\r\n",
        result.class_probabilities[0],
        result.class_probabilities[1],
        result.class_probabilities[2],
        result.class_probabilities[3],
        result.class_probabilities[4]
    );
    UART_Print(uart_buf);

    if (fault_detected) {
        snprintf(uart_buf, sizeof(uart_buf),
            "  *** FAULT ALERT: %s detected! ***\r\n",
            FaultDetector_GetFaultName(result.predicted_class)
        );
        UART_Print(uart_buf);
    }
}

/* ============================================================================
 * LED STATUS INDICATOR
 * ============================================================================
 *
 * LED Pattern:
 *   Normal:        Green ON, others OFF
 *   Bearing:       Red ON (blinking fast)
 *   Imbalance:     Red + Blue ON
 *   Misalignment:  Red ON (solid)
 *   Electrical:    All LEDs blinking
 */

void LED_SetStatus(FaultClass_t fault)
{
    /* Reset all LEDs */
    HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_BLUE_PORT, LED_BLUE_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_RESET);

    switch (fault) {
        case FAULT_NORMAL:
            HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_SET);
            break;

        case FAULT_BEARING:
            HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
            break;

        case FAULT_IMBALANCE:
            HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(LED_BLUE_PORT, LED_BLUE_PIN, GPIO_PIN_SET);
            break;

        case FAULT_MISALIGNMENT:
            HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_SET);
            break;

        case FAULT_ELECTRICAL:
            HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(LED_BLUE_PORT, LED_BLUE_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_SET);
            break;
    }
}

void Buzzer_Alert(bool enable)
{
    HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN,
                      enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/* ============================================================================
 * TIMER INTERRUPT — 400 Hz Sampling Trigger
 * ============================================================================ */

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM2) {
        sample_timer_flag = true;
    }
}

/* ============================================================================
 * PERIPHERAL INITIALIZATION
 * ============================================================================ */

void SystemClock_Config(void)
{
    /* Configure system clock to 480 MHz using PLL
     * HSE → PLL → SYSCLK = 480 MHz
     * APB1 = 120 MHz, APB2 = 120 MHz
     *
     * Note: Full clock tree configuration should be generated
     * using STM32CubeMX for production firmware. */

    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /* Configure power supply */
    HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

    /* HSE oscillator and PLL configuration */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 1;
    RCC_OscInitStruct.PLL.PLLN = 120;
    RCC_OscInitStruct.PLL.PLLP = 2;
    RCC_OscInitStruct.PLL.PLLQ = 2;
    RCC_OscInitStruct.PLL.PLLR = 2;
    RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
    RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    /* System clock configuration */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                  RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                                  RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
    RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
        Error_Handler();
    }
}

void GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    /* Enable GPIO clocks */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* Configure LED pins: PB0 (Green), PB7 (Blue), PB14 (Red) */
    GPIO_InitStruct.Pin = LED_GREEN_PIN | LED_BLUE_PIN | LED_RED_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* Configure Buzzer pin: PA0 */
    GPIO_InitStruct.Pin = BUZZER_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(BUZZER_PORT, &GPIO_InitStruct);

    /* Initialize all outputs LOW */
    HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_BLUE_PORT, LED_BLUE_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(BUZZER_PORT, BUZZER_PIN, GPIO_PIN_RESET);
}

void I2C1_Init(void)
{
    hi2c1.Instance = I2C1;
    hi2c1.Init.Timing = 0x307075B1;          /* 400 kHz Fast Mode */
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
        Error_Handler();
    }
}

void Timer_Init(void)
{
    /* TIM2: 400 Hz interrupt for accelerometer sampling
     * Timer clock = 120 MHz (APB1 × 2)
     * Prescaler = 1200 - 1 → Counter clock = 100 kHz
     * Period = 250 - 1 → Interrupt freq = 100000/250 = 400 Hz */

    __HAL_RCC_TIM2_CLK_ENABLE();

    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 1200 - 1;
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 250 - 1;
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;

    if (HAL_TIM_Base_Init(&htim2) != HAL_OK) {
        Error_Handler();
    }

    /* Enable TIM2 interrupt */
    HAL_NVIC_SetPriority(TIM2_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(TIM2_IRQn);
}

static void UART3_Init(void)
{
    /* USART3 is connected to ST-Link Virtual COM Port on Nucleo board */
    huart3.Instance = USART3;
    huart3.Init.BaudRate = 115200;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;

    if (HAL_UART_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }
}

static void UART_Print(const char* msg)
{
    HAL_UART_Transmit(&huart3, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
}

static void System_PrintBanner(void)
{
    UART_Print("\r\n");
    UART_Print("============================================================\r\n");
    UART_Print("  Motor Fault Detection System v1.0\r\n");
    UART_Print("  STM32 NUCLEO-H743ZI2 + LIS3DH Accelerometer\r\n");
    UART_Print("============================================================\r\n");
    UART_Print("  Sampling Rate : 400 Hz\r\n");
    UART_Print("  Window Size   : 1024 samples (2.56 sec)\r\n");
    UART_Print("  Features      : 42 (time-domain, 3-axis)\r\n");
    UART_Print("  Classes       : Normal, Bearing, Imbalance,\r\n");
    UART_Print("                  Misalignment, Electrical\r\n");
    UART_Print("============================================================\r\n\r\n");
}

/* ============================================================================
 * ERROR HANDLER
 * ============================================================================ */

void Error_Handler(void)
{
    /* Disable interrupts */
    __disable_irq();

    /* Blink red LED rapidly to indicate error */
    while (1) {
        HAL_GPIO_TogglePin(LED_RED_PORT, LED_RED_PIN);
        for (volatile uint32_t i = 0; i < 500000; i++);
    }
}

/* ============================================================================
 * INTERRUPT HANDLERS
 * ============================================================================ */

void TIM2_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim2);
}

void NMI_Handler(void)       { }
void HardFault_Handler(void) { while (1); }
void MemManage_Handler(void) { while (1); }
void BusFault_Handler(void)  { while (1); }
void UsageFault_Handler(void){ while (1); }
void SVC_Handler(void)       { }
void PendSV_Handler(void)    { }
void SysTick_Handler(void)   { HAL_IncTick(); }
