# Hardware Setup Guide
## STM32 NUCLEO-H743ZI2 + LIS3DH Accelerometer

---

## 1. Components Required

| Component | Description | Quantity |
|-----------|-------------|----------|
| STM32 NUCLEO-H743ZI2 | Development board (Cortex-M7 @ 480 MHz) | 1 |
| LIS3DH Breakout Board | 3-axis MEMS accelerometer (e.g., Adafruit #2809) | 1 |
| DC Motor | 12V DC motor (or any available motor) | 1 |
| Red LED | 5mm, for fault indication | 1 |
| Green LED | 5mm, for normal status | 1 |
| Blue LED | 5mm, for processing indicator | 1 |
| Piezo Buzzer | 5V active buzzer for audible alerts | 1 |
| 220Ω Resistors | LED current limiting | 3 |
| Jumper Wires | Male-to-female, various lengths | ~15 |
| Breadboard | Standard 830-point breadboard | 1 |
| USB Cable | Micro-USB for STM32 programming | 1 |
| Power Supply | 12V for motor (battery or adapter) | 1 |

---

## 2. Pin Mapping

### I2C Connection (LIS3DH → STM32)

```
LIS3DH Breakout          STM32 NUCLEO-H743ZI2
─────────────            ────────────────────
VCC  ───────────────────  3.3V (CN8 Pin 7)
GND  ───────────────────  GND  (CN8 Pin 11)
SDA  ───────────────────  PB9  (CN7 Pin 4) — I2C1_SDA
SCL  ───────────────────  PB8  (CN7 Pin 2) — I2C1_SCL
INT1 ───────────────────  PA1  (CN7 Pin 30) — EXTI (optional)
SDO  ───────────────────  GND  (sets I2C address to 0x18)
```

> **Important**: The LIS3DH operates at 3.3V. Do NOT connect to 5V.

### LED & Buzzer Connections

```
Component               STM32 Pin
─────────               ─────────
Green LED (+) ──[220Ω]── PB0  (CN10 Pin 31)
Blue LED  (+) ──[220Ω]── PB7  (CN7 Pin 21)
Red LED   (+) ──[220Ω]── PB14 (CN10 Pin 28)
Buzzer    (+) ──────────  PA0  (CN10 Pin 29)
All (-)   ──────────────  GND
```

> **Note**: On the NUCLEO-H743ZI2, PB0 (Green), PB7 (Blue), and PB14 (Red)
> are already connected to on-board LEDs. You can use them directly without
> external LEDs.

---

## 3. Wiring Diagram

```
                    ┌──────────────────────┐
                    │  STM32 NUCLEO-H743ZI2│
                    │                      │
    ┌───────┐       │  3.3V ────┐          │
    │LIS3DH │       │           │          │
    │       │  VCC ─┤───────────┘          │
    │       │  GND ─┤── GND               │
    │       │  SDA ─┤── PB9 (I2C1_SDA)    │
    │       │  SCL ─┤── PB8 (I2C1_SCL)    │
    │       │  INT1─┤── PA1 (optional)     │
    │       │  SDO ─┤── GND (addr=0x18)   │
    └───────┘       │                      │
                    │  PB0 ── [220Ω] ── Green LED ── GND
                    │  PB7 ── [220Ω] ── Blue LED  ── GND
                    │  PB14── [220Ω] ── Red LED   ── GND
                    │  PA0 ──────────── Buzzer (+) ── GND
                    │                      │
                    │  USB ← Programming   │
                    └──────────────────────┘

    ┌─────────┐
    │  Motor  │ ←── LIS3DH mounted on motor housing
    │  (12V)  │     using adhesive or mechanical mount
    └─────────┘
```

---

## 4. Sensor Placement

### Optimal Accelerometer Positioning

The LIS3DH accelerometer should be mounted to capture vibrations that
correspond to the training data patterns:

1. **Location**: Mount on the motor housing, near the drive-end bearing
2. **Orientation**: 
   - X-axis: Horizontal (radial)
   - Y-axis: Vertical (radial)
   - Z-axis: Axial (along shaft)
3. **Mounting**: Use rigid adhesive (epoxy or cyanoacrylate) for best
   high-frequency transmission. Avoid flexible mounting.
4. **Surface**: Clean and flat surface, perpendicular to the shaft axis

```
    Motor Cross-Section:
    
         Y (vertical)
         ↑
         │
    ─────┼─────── X (horizontal)
         │
         │        Z → (axial, into page)
    
    [LIS3DH sensor mounted here]
    ├──────────────────────────┤
    │    Motor Housing         │
    │   ┌──────────────┐       │
    │   │   Bearing    │       │
    │   │  ┌────────┐  │       │
    │   │  │ Shaft  │  │       │
    │   │  └────────┘  │       │
    │   └──────────────┘       │
    └──────────────────────────┘
```

---

## 5. Software Setup

### STM32CubeIDE Project Setup

1. **Install STM32CubeIDE** from st.com
2. Create new STM32 project for NUCLEO-H743ZI2
3. In STM32CubeMX:
   - Enable I2C1 (PB8=SCL, PB9=SDA) at 400 kHz
   - Enable TIM2 for 400 Hz interrupt
   - Enable USART3 (for debug output via ST-Link)
   - Configure GPIO: PB0, PB7, PB14 (Output), PA0 (Output)
4. Generate code
5. Copy firmware files:
   - `stm32_firmware/Src/main.c` → `Core/Src/main.c`
   - `stm32_firmware/Src/lis3dh_driver.c` → `Core/Src/`
   - `stm32_firmware/Src/fault_detector.c` → `Core/Src/`
   - `stm32_firmware/Inc/*.h` → `Core/Inc/`

### Flashing the Firmware

```bash
# Using STM32CubeIDE: Build → Run
# Using ST-Link CLI:
ST-LINK_CLI -c SWD -P firmware.bin 0x08000000 -V -Rst
```

### Serial Monitor

Connect to the Nucleo's virtual COM port at **115200 baud** to see
real-time detection output:

```
============================================================
  Motor Fault Detection System v1.0
  STM32 NUCLEO-H743ZI2 + LIS3DH Accelerometer
============================================================
[INIT] LIS3DH initialized successfully (400 Hz, +/-4g)
[INIT] Fault detector initialized
[INIT] System ready. Starting monitoring...

[W0001] Class: Normal | Confidence: 92.3% | Inference: 145 us
[W0002] Class: Normal | Confidence: 95.1% | Inference: 142 us
[W0003] Class: Bearing Fault | Confidence: 87.5% | Inference: 148 us
  *** FAULT ALERT: Bearing Fault detected! ***
```

---

## 6. I2C Communication Protocol

### Register Configuration Sequence

```
1. Read WHO_AM_I (0x0F) → expect 0x33
2. Write CTRL_REG1 (0x20) = 0x77
   [ODR=400Hz, XYZ enabled, Normal mode]
3. Write CTRL_REG4 (0x23) = 0x98
   [BDU=1, FS=±4g, HR=1]
4. Write CTRL_REG3 (0x22) = 0x10
   [DRDY interrupt on INT1]
```

### Data Read Sequence

```
1. Check STATUS (0x27) bit 3 (ZYXDA = data available)
2. Read 6 bytes from 0x28 with auto-increment:
   OUT_X_L (0x28), OUT_X_H (0x29)
   OUT_Y_L (0x2A), OUT_Y_H (0x2B)
   OUT_Z_L (0x2C), OUT_Z_H (0x2D)
3. Combine bytes: raw = (HIGH << 8) | LOW
4. Shift right 4 bits for 12-bit resolution
5. Convert to g: accel_g = raw_12bit × sensitivity / 1000
```

---

## 7. Troubleshooting

| Issue | Solution |
|-------|---------|
| WHO_AM_I returns 0x00 | Check wiring, verify I2C pull-ups (4.7kΩ) |
| No data from sensor | Verify CTRL_REG1 ODR is not power-down |
| Noisy readings | Check mounting rigidity, add decoupling cap |
| I2C timeout | Reduce I2C speed to 100 kHz |
| Model accuracy low | Retrain with sensor data, adjust thresholds |
| Buzzer always on | Increase FAULT_THRESHOLD in main.h |
