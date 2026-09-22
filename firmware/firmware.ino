/*
 * MIMO Differential-Drive Car Firmware
 * Consolidated Arduino IDE sketch (single .ino file)
 *
 * Target board        : ESP32 Dev Module
 * Arduino-ESP32 core  : v2.0.x or v3.x (auto-detected at compile time)
 * Required library    : ArduinoJson by Benoit Blanchon (v6.21.5 or newer)
 *                       Install via: Tools -> Manage Libraries -> search "ArduinoJson"
 * Upload speed        : 921600
 * CPU frequency       : 240 MHz
 * Flash size          : 4MB (or match your board)
 * Partition scheme    : Default 4MB with spiffs (1.2MB APP / 1.5MB SPIFFS)
 * Monitor speed       : 460800 baud  (note: firmware uses 460800, not 115200)
 *
 * WiFi                : Defaults to AP mode "MIMO-Car" / password "mimo-car-setup"
 *                       Edit WIFI_STA_SSID / WIFI_STA_PASSWORD in UserConfig below
 *                       to connect to an existing network instead.
 * TCP control port    : 8888
 * UDP diagnostic port : 9999
 *
 * Pin map (DRV8871 + encoders + I2C):
 *   Right motor IN1/IN2 : GPIO18 / GPIO19
 *   Left  motor IN1/IN2 : GPIO16 / GPIO17
 *   Right encoder A/B   : GPIO32 / GPIO33
 *   Left  encoder A/B   : GPIO25 / GPIO26
 *   I2C SDA/SCL         : GPIO21 / GPIO22  (INA3221 @0x40, MPU6050 @0x68)
 *
 * This file was auto-merged from the original PlatformIO project:
 *   platformio.ini
 *   include/user_config.h
 *   include/control_types.h
 *   include/vehicle_io.h
 *   include/theory_controller.h
 *   src/vehicle_io.cpp
 *   src/theory_controller.cpp
 *   src/main.cpp
 *
 * Build instructions:
 *   1. Install the ESP32 board package in Arduino IDE
 *      (Board Manager -> search "esp32" by Espressif).
 *   2. Install ArduinoJson via Library Manager.
 *   3. Open this .ino file. The sketch folder name must match the file name.
 *   4. Select Tools -> Board -> "ESP32 Dev Module".
 *   5. Set Tools -> Upload Speed -> 921600.
 *   6. Select the correct COM port and click Upload.
 */

// ===== System & library includes =====
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <Wire.h>
#include <driver/gpio.h>
#include <driver/pcnt.h>
#include <esp_arduino_version.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <cstddef>
#include <cstdint>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <lwip/sockets.h>

// ===== include/user_config.h =====
namespace UserConfig {

// DRV8871 control pins.
constexpr int PIN_RIGHT_IN1 = 18;
constexpr int PIN_RIGHT_IN2 = 19;
constexpr int PIN_LEFT_IN1 = 16;
constexpr int PIN_LEFT_IN2 = 17;

// Quadrature encoder pins.
constexpr int PIN_RIGHT_ENC_A = 32;
constexpr int PIN_RIGHT_ENC_B = 33;
constexpr int PIN_LEFT_ENC_A = 25;
constexpr int PIN_LEFT_ENC_B = 26;

// Shared I2C bus.
constexpr int PIN_I2C_SDA = 21;
constexpr int PIN_I2C_SCL = 22;

constexpr int RIGHT_MOTOR_SIGN = 1;
constexpr int LEFT_MOTOR_SIGN = -1;

// PCNT ×4 正交解码（官方 rotary-encoder 配置）的原始计数方向与实际运动
// 相反（实际前进时计数为负），故 SIGN=-1 使速度与 ticks 方向都取正。
constexpr int RIGHT_ENCODER_SIGN = -1;

constexpr int LEFT_ENCODER_SIGN = -1;

constexpr int GYRO_Z_SIGN = 1;

// Measured vehicle geometry and encoder resolution.
constexpr float WHEEL_RADIUS_M = 0.0325f;
constexpr float TRACK_WIDTH_M = 0.2035f;
constexpr int ENCODER_TICKS_PER_WHEEL_REV = 1320;  // 11 PPR x 30 gearbox x 4.

// Electrical configuration.
constexpr uint8_t INA3221_ADDRESS = 0x40;
constexpr uint8_t MPU6050_ADDRESS = 0x68;
constexpr uint8_t INA_CHANNEL_LEFT = 0;   // CH1
constexpr uint8_t INA_CHANNEL_RIGHT = 1;  // CH2
constexpr uint8_t INA_CHANNEL_LOGIC = 2;  // CH3, LM2596 input
constexpr float INA_SHUNT_OHM = 0.100f;
constexpr float NOMINAL_BATTERY_V = 11.1f;
// Diagnostic power-sag limit requested for the current vehicle.  This is a
// hard output ceiling, independent of any runtime configuration sent by PC.
constexpr float MAX_MOTOR_COMMAND_V = 4.0f;

constexpr float MOTOR_V_PER_MS = 14.0f;

// Real-time loop and PWM.
constexpr uint32_t CONTROL_PERIOD_US = 5000;
constexpr uint32_t PWM_FREQUENCY_HZ = 20000;
constexpr uint8_t PWM_RESOLUTION_BITS = 10;

constexpr uint32_t DRV8871_WAKE_PULSE_US = 1000;

constexpr float DEFAULT_MOTOR_DEADZONE_V = 0.5f;

constexpr char WIFI_STA_SSID[] = "";
constexpr char WIFI_STA_PASSWORD[] = "";
constexpr char WIFI_AP_SSID[] = "MIMO-Car";
constexpr char WIFI_AP_PASSWORD[] = "mimo-car-setup";
constexpr uint16_t TCP_PORT = 8888;

}  // namespace UserConfig

// ===== include/control_types.h =====
constexpr int BLOCK_N = 2;
constexpr int SIGMA_N = 6;
constexpr int REGRESSOR_N = BLOCK_N + SIGMA_N;

struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;
};

struct Pose2D {
  float x = 0.0f;
  float y = 0.0f;
  float theta = 0.0f;
};

enum class RunMode : uint8_t {
  IDLE = 0,
  MONITOR = 1,
  MANUAL = 2,
  COLLECT = 3,
  THEORY = 4,
};

enum class FaultCode : uint8_t {
  NONE = 0,
  NETWORK_LOST = 1,
  OVER_CURRENT = 2,
  UNDER_VOLTAGE = 3,
  SENSOR_FAILURE = 4,
  NONFINITE_CONTROL = 5,
  CONTROL_OVERRUN = 6,
  INVALID_WEIGHTS = 7,
  TRACKING_DIVERGENCE = 8,
  IMU_FAILURE = 9,
  RIGHT_ENCODER_FAILURE = 10,
  LEFT_ENCODER_FAILURE = 11,
};

struct RuntimeConfig {
  float dtControl = 0.005f;
  float telemetryPeriod = 0.020f;
  float collectionPeriod = 0.010f;
  float integralWindow = 0.20f;
  float runDuration = 30.0f;
  // 记录保持段：参考在 run_duration 后钳在终端位形，运行再延续本时长，
  // 让终端捕获（预览钳在 T_f 的减速段）在记录内完整出现。0 = 旧行为。
  float holdDuration = 0.0f;
  float uMax = UserConfig::MAX_MOTOR_COMMAND_V;
  float motorDeadzone = UserConfig::DEFAULT_MOTOR_DEADZONE_V;
  float currentLimit = 1.30f;
  float batteryMin = 9.6f;
  float velocityTau = 0.030f;
  float currentTau = 0.040f;
  float derivativeTau = 0.025f;
  float gyroTau = 0.15f;
  // A filtered yaw-rate magnitude at or below this threshold is retained for
  // diagnostics only. It does not update the odometry/control omega state.
  float gyroDeadband = 0.040f;
  float gyroBlend = 0.75f;
  float tau1 = 0.040f;
  float tau2 = 0.015f;
  float r2 = 0.50f;  // 内环自适应学习率（置 0 仍可手动禁用 mhat 在线更新）
  float r3 = 0.50f;  // 当前实车验证默认启用在线补偿
  float delta2 = 0.08f;
  float delta3 = 0.10f;
  // 对比实验：1=完整数据驱动，2=关闭自适应，3=速度-电流级联 PID。
  float comparisonCase = 1.0f;
  float pidVelocityFf = 0.45f;
  float pidVelocityKp = 0.25f;
  float pidVelocityKi = 0.10f;
  float pidVelocityKd = 0.0f;
  float pidCurrentKp = 10.0f;
  float pidCurrentKi = 20.0f;
  float pidCurrentKd = 0.0f;
  float pidVoltageFf = 20.0f;
  float pidCurrentRefMax = 0.30f;
  float pidDerivativeTau = 0.020f;
  // Manuscript previewed polar-capture outer loop.  The existing wire keys
  // outer_kp/outer_vbar/outer_ktheta are retained for PC compatibility and
  // now denote k_rho/bar(v_c)/k_alpha, respectively.
  float outerKp = 0.80f;
  float outerVbar = 0.50f;
  float outerKtheta = 0.70f;
  float outerPreviewHorizon = 1.40f;
  float outerCaptureRadius = 0.010f;
  float outerBlendRadius = 0.030f;
  // Closed-loop motion-reference governor.  It starts from the measured
  // vehicle state, tracks the prescribed target, and stays inside a feasible
  // velocity/acceleration envelope before the vehicle outer loop sees it.
  float motionKx = 2.0f;
  float motionKy = 6.0f;
  float motionKth = 3.0f;
  float motionMaxV = 0.10f;
  float motionMaxOmega = 1.00f;
  float motionMaxAccel = 0.20f;
  float motionMaxAngularAccel = 2.50f;
  float outerMaxV = 0.30f;
  float outerMaxOmega = 3.00f;
  float motionMaxTargetError = 0.20f;
  float refA = 0.40f;
  float refB = 0.20f;
  float refNu = 0.20f;
  // 参考轨迹形状：0=直线往返/双纽线，1=圆，2=四叶草。
  float refShape = 0.0f;
  float theoryMaxPositionError = 0.20f;
  float theoryMaxHeadingError = 1.047198f;
  float theoryMaxZ2 = 2.00f;
  float theoryMaxZ3 = 2.00f;
  float theorySafetyGrace = 0.75f;
  float collectScale = 0.55f;
  uint16_t collectSamples = 800;
  bool manualSnapshots = false;
};

struct ControllerWeights {
  uint32_t magic = 0x4D494D4Fu;  // MIMO
  uint16_t version = 4;
  uint16_t reserved = 0;
  float w2[BLOCK_N][REGRESSOR_N] = {};
  float w3[BLOCK_N][REGRESSOR_N] = {};
  float kappa2 = 1.0f;
  float kappa3 = 2.0f;
  float kappaMin2 = NAN;
  float kappaMin3 = NAN;
  float sigmaMargin2 = NAN;
  float sigmaMargin3 = NAN;
  float epsilon2 = 0.8f;
  float epsilon3 = 3.0f;
  float dbar2 = 0.004f;
  float dbar3 = 0.008f;
  float certificateMax2 = NAN;
  float certificateMax3 = NAN;
  uint16_t rank2 = 0;
  uint16_t rank3 = 0;
  uint32_t checksum = 0;
};

struct SensorState {
  uint64_t timestampUs = 0;
  int64_t ticksRight = 0;
  int64_t ticksLeft = 0;
  float wheelRight = 0.0f;
  float wheelLeft = 0.0f;
  Vec2 velocityRaw{};    // pre-filter [v, omega]
  Vec2 velocity{};       // [v, omega]
  Vec2 velocityDot{};
  Vec2 currentRaw{};     // sanitized pre-filter current [iR, iL]
  Vec2 current{};        // signed armature-current proxy [iR, iL]
  Vec2 currentDot{};
  float channelCurrent[3] = {};
  float busVoltage[3] = {};
  float gyroZ = 0.0f;
  Pose2D pose{};
  bool inaOk = false;
  bool imuOk = false;
  bool imuCalibrated = false;
};

struct SnapshotFrame {
  bool valid = false;
  float windowSeconds = 0.0f;
  float zdot2[2] = {};
  float y2[8] = {};
  float x3[2] = {};
  float zdot3[2] = {};
  float y3[8] = {};
  float x4[2] = {};
  float velocityRaw[2] = {};
  float currentRaw[2] = {};
};

struct CollectionPoint {
  bool valid = false;
  float z2[2] = {};
  float y2[8] = {};
  float x3[2] = {};
  float z3[2] = {};
  float y3[8] = {};
  float x4[2] = {};
  float velocityRaw[2] = {};
  float currentRaw[2] = {};
};

struct ControlState {
  Vec2 alpha1{};
  Vec2 beta1{};
  Vec2 beta1Dot{};
  Vec2 alpha2{};
  Vec2 beta2{};
  Vec2 beta2Dot{};
  Vec2 z2{};
  Vec2 z3{};
  Vec2 uc{};
  Vec2 u{};
  Pose2D reference{};
  Pose2D motionReference{};
  Vec2 motionVelocity{};
  float poseError[3] = {};
  float motionPoseError[3] = {};
  float mhat2[2][6] = {};
  float mhat3[2][6] = {};
};

struct TelemetryFrame {
  uint64_t timestampUs = 0;
  uint32_t sequence = 0;
  RunMode mode = RunMode::IDLE;
  FaultCode fault = FaultCode::NONE;
  bool armed = false;
  bool weightsValid = false;
  uint32_t droppedFrames = 0;
  uint32_t loopTimeUs = 0;
  SensorState sensors{};
  ControlState control{};
  SnapshotFrame snapshot{};
};

inline float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}

inline float gateGyroForAlgorithm(float gyroRadS, float thresholdRadS) {
  return std::fabs(gyroRadS) <= thresholdRadS ? 0.0f : gyroRadS;
}

inline bool finite2(const Vec2& value) {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

// ===== include/vehicle_io.h =====
class VehicleIO {
 public:
  void configure(const RuntimeConfig& config) { config_ = config; }
  bool begin();
  bool sample(float dtSeconds);
  void setMotorVoltages(const Vec2& voltage);
  void stopMotors();
  void zeroPose();
  void setPose(const Pose2D& pose);
  bool calibrateGyro(uint16_t samples = 600);

  const SensorState& state() const { return state_; }
  Vec2 lastMotorVoltage() const { return lastMotorVoltage_; }
  bool inaAvailable() const { return state_.inaOk; }
  bool imuAvailable() const { return state_.imuOk; }
  bool gyroCalibrationValid() const { return state_.imuCalibrated; }

 private:
  static VehicleIO* instance_;
  bool initializeEncoders();

  bool initializeIna3221();
  bool initializeMpu6050();
  bool readIna3221();
  bool readMpu6050(float& gyroZRadS);
  bool writeI2c8(uint8_t address, uint8_t reg, uint8_t value);
  bool writeI2c16(uint8_t address, uint8_t reg, uint16_t value);
  bool readI2c(uint8_t address, uint8_t reg, uint8_t* data, size_t length);
  bool readI2c16(uint8_t address, uint8_t reg, uint16_t& value);
  void initializePwm();
  void writeBridge(int pinIn1, int pinIn2, int channelIn1, int channelIn2,
                   float voltage, int motorSign, bool& active,
                   int8_t& direction);

  // PCNT 硬件计数（×4 正交，100 ns 毛刺滤波）：int16 计数器，5 ms 采样周期
  // 内正常增量仅几十 tick；异常跳变（回绕/干扰）按 MAX_PCNT_DELTA 丢弃。
  static constexpr int64_t MAX_PCNT_DELTA = 10000;
  int64_t lastPcntRight_ = 0;
  int64_t lastPcntLeft_ = 0;
  int64_t lastTicksRight_ = 0;
  int64_t lastTicksLeft_ = 0;
  bool firstSample_ = true;
  // 编码器存活监控：电机有电压输出但双轮长时间无 tick 时打印诊断（不锁故障）。
  uint32_t encNoTickCount_ = 0;
  uint32_t encNoTickLogMs_ = 0;
  Vec2 velocityTracked_{};
  Vec2 currentTracked_{};
  float gyroBiasRadS_ = 0.0f;
  float gyroFilteredRadS_ = 0.0f;
  float gyroRawRadS_ = 0.0f;  // 去零偏原始读数，供算法直接使用（方案 A）
  bool gyroFilterInitialized_ = false;
  static constexpr uint8_t STUCK_THRESHOLD = 100;  // 500 ms at 200 Hz
  int16_t lastRawShunt_[3] = {};
  uint8_t shuntStuckCount_[3] = {};
  int16_t lastRawBus_[3] = {};
  uint8_t busStuckCount_[3] = {};
  float rightCurrentSign_ = 1.0f;
  float leftCurrentSign_ = 1.0f;
  bool rightBridgeActive_ = false;
  bool leftBridgeActive_ = false;
  int8_t rightBridgeDirection_ = 0;
  int8_t leftBridgeDirection_ = 0;
  Vec2 lastMotorVoltage_{};
  SensorState state_{};
  RuntimeConfig config_{};
};

// ===== include/theory_controller.h =====
class TheoryController {
 public:
  void configure(const RuntimeConfig& config) { config_ = config; }
  bool setWeights(const ControllerWeights& weights);
  bool weightsValid() const { return weightsValid_; }
  const ControllerWeights& weights() const { return weights_; }
  const ControlState& state() const { return state_; }

  void reset(const SensorState& sensors, float runTimeSeconds = 0.0f);
  Vec2 step(const SensorState& sensors, float runTimeSeconds, float dtSeconds);
  Vec2 collectionExcitation(float runTimeSeconds) const;
  CollectionPoint collectionPoint(const SensorState& sensors,
                                  const Vec2& appliedVoltage,
                                  float runTimeSeconds) const;

  static uint32_t checksum(const ControllerWeights& weights);
  static bool validate(const ControllerWeights& weights);

 private:
  static float wrapAngle(float angle);
  static float smoothSign(float value);
  static Vec2 matVec2x8(const float matrix[2][8], const float vector[8]);
  static Vec2 matVec2x6(const float matrix[2][6], const float vector[6]);
  Vec2 referencePosition(float t) const;
  void reference(float t, Pose2D& pose, Vec2& velocity,
                 Vec2& cartesianVelocity, Vec2& cartesianAcceleration);
  Vec2 outerCommand(const SensorState& sensors, bool terminalCapture);
  Vec2 stepCascadedPid(const SensorState& sensors, float runTimeSeconds,
                       float dtSeconds);
  void sigma2(const SensorState& sensors, const Vec2& betaDot, float out[6]) const;
  void sigma3(const SensorState& sensors, const Vec2& betaDot, float out[6]) const;
  void syntheticFilter(float t, int block, Vec2& beta, Vec2& betaDot) const;

  RuntimeConfig config_{};
  ControllerWeights weights_{};
  ControlState state_{};
  bool initialized_ = false;
  bool weightsValid_ = false;
  float lastReferenceTheta_ = 0.0f;
  float refStartTheta_ = 0.0f;  // 启动斜坡起点航向（车当前航向）
  Vec2 pidVelocityIntegral_{};
  Vec2 pidVelocityPreviousError_{};
  Vec2 pidVelocityDerivative_{};
  Vec2 pidCurrentIntegral_{};
  Vec2 pidCurrentPreviousError_{};
  Vec2 pidCurrentDerivative_{};
};

// ===== src/vehicle_io.cpp =====
namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float DEG_TO_RAD_F = PI_F / 180.0f;
constexpr int PWM_CHANNEL_RIGHT_IN1 = 0;
constexpr int PWM_CHANNEL_RIGHT_IN2 = 1;
constexpr int PWM_CHANNEL_LEFT_IN1 = 2;
constexpr int PWM_CHANNEL_LEFT_IN2 = 3;

float lowPass(float previous, float input, float tau, float dt) {
  const float alpha = dt / (tau + dt);
  return previous + alpha * (input - previous);
}

float wrapAngle(float angle) {
  while (angle > PI_F) angle -= 2.0f * PI_F;
  while (angle < -PI_F) angle += 2.0f * PI_F;
  return angle;
}

}  // namespace

VehicleIO* VehicleIO::instance_ = nullptr;

bool VehicleIO::begin() {
  instance_ = this;
  Wire.begin(UserConfig::PIN_I2C_SDA, UserConfig::PIN_I2C_SCL);
  // The 5-ms control loop needs fast-mode I2C for the MPU6050 and INA3221 reads.
  Wire.setClock(400000);
  Wire.setTimeOut(20);

  initializeEncoders();

  initializePwm();
  stopMotors();

  state_.inaOk = initializeIna3221();
  // IMU 机制已取消（用户决策）：不再初始化/读取 MPU6050，ω 全部由编码器
  // 差速计算。imu_ok/imuCalibrated 恒 false（协议字段保留兼容）。
  state_.imuOk = false;
  state_.imuCalibrated = false;
  return state_.inaOk;
}

bool VehicleIO::initializeEncoders() {
  pinMode(UserConfig::PIN_RIGHT_ENC_A, INPUT_PULLUP);
  pinMode(UserConfig::PIN_RIGHT_ENC_B, INPUT_PULLUP);
  pinMode(UserConfig::PIN_LEFT_ENC_A, INPUT_PULLUP);
  pinMode(UserConfig::PIN_LEFT_ENC_B, INPUT_PULLUP);
  // 硬件正交解码 ×4：每个 PCNT unit 双通道（通道 0：A 脉冲 + B 判向；
  // 通道 1：B 脉冲 + A 判向），共享计数器得到四倍频带符号计数。
  // 100 ns 毛刺滤波器吃掉编码器抖动与电机 EMI 假沿——前置 SISO 小车固件
  // （ADNDSC_controller）采用同一做法；GPIO 中断方案对此类毛刺无防护。
  const pcnt_unit_t units[2] = {PCNT_UNIT_0, PCNT_UNIT_1};
  const int pinsA[2] = {UserConfig::PIN_RIGHT_ENC_A, UserConfig::PIN_LEFT_ENC_A};
  const int pinsB[2] = {UserConfig::PIN_RIGHT_ENC_B, UserConfig::PIN_LEFT_ENC_B};
  for (int wheel = 0; wheel < 2; ++wheel) {
    pcnt_config_t cfg = {};
    for (int channel = 0; channel < 2; ++channel) {
      cfg.pulse_gpio_num = channel == 0 ? pinsA[wheel] : pinsB[wheel];
      cfg.ctrl_gpio_num = channel == 0 ? pinsB[wheel] : pinsA[wheel];
      cfg.channel = static_cast<pcnt_channel_t>(channel);
      // 官方 rotary-encoder ×4 配置（新 pulse_cnt API 映射到 legacy PCNT）：
      // 两个通道的边沿动作必须相反，否则每周期计数相互抵消为 0。
      if (channel == 0) {
        // A 边沿通道：上升沿 +1、下降沿 -1；控制低电平保持、高电平反转。
        cfg.pos_mode = PCNT_COUNT_INC;
        cfg.neg_mode = PCNT_COUNT_DEC;
        cfg.lctrl_mode = PCNT_MODE_KEEP;
        cfg.hctrl_mode = PCNT_MODE_REVERSE;
      } else {
        // B 边沿通道与 A 通道边沿动作相反：上升沿 -1、下降沿 +1。
        cfg.pos_mode = PCNT_COUNT_DEC;
        cfg.neg_mode = PCNT_COUNT_INC;
        cfg.lctrl_mode = PCNT_MODE_KEEP;
        cfg.hctrl_mode = PCNT_MODE_REVERSE;
      }
      cfg.counter_h_lim = 32767;
      cfg.counter_l_lim = -32768;
      cfg.unit = units[wheel];
      const esp_err_t err = pcnt_unit_config(&cfg);
      if (err != ESP_OK) {
        Serial.printf("[PCNT] unit %d channel %d config FAILED err=0x%x\n",
                      wheel, channel, static_cast<unsigned int>(err));
        return false;
      }
    }
    pcnt_set_filter_value(units[wheel], 100);  // 100 ns 毛刺滤波
    pcnt_filter_enable(units[wheel]);
    pcnt_counter_pause(units[wheel]);
    pcnt_counter_clear(units[wheel]);
    pcnt_counter_resume(units[wheel]);
  }
  Serial.printf("[PCNT] encoders ready: units 0/1 x4 quadrature, filter=100\n");
  return true;
}

bool VehicleIO::sample(float dtSeconds) {
  if (!(dtSeconds > 0.0f) || dtSeconds > 0.050f) return false;

  int16_t pcntRight = 0;
  int16_t pcntLeft = 0;
  pcnt_get_counter_value(PCNT_UNIT_0, &pcntRight);
  pcnt_get_counter_value(PCNT_UNIT_1, &pcntLeft);
  const int64_t rawDeltaRight = static_cast<int64_t>(pcntRight) - lastPcntRight_;
  const int64_t rawDeltaLeft = static_cast<int64_t>(pcntLeft) - lastPcntLeft_;
  lastPcntRight_ = pcntRight;
  lastPcntLeft_ = pcntLeft;
  // int16 计数器回绕或干扰跳变时丢弃该周期（5 ms 内正常增量仅几十 tick）。
  const int64_t deltaRight =
      (rawDeltaRight > MAX_PCNT_DELTA || rawDeltaRight < -MAX_PCNT_DELTA)
          ? 0 : rawDeltaRight;
  const int64_t deltaLeft =
      (rawDeltaLeft > MAX_PCNT_DELTA || rawDeltaLeft < -MAX_PCNT_DELTA)
          ? 0 : rawDeltaLeft;
  // 编码器存活监控：有电压输出但双轮 0.5 s 无 tick → 打印诊断（节流 2 s）。
  const bool wheelsPowered =
      std::fabs(lastMotorVoltage_.x) > 0.5f ||
      std::fabs(lastMotorVoltage_.y) > 0.5f;
  if (wheelsPowered && deltaRight == 0 && deltaLeft == 0) {
    if (++encNoTickCount_ >= 100) {
      const uint32_t nowMs = static_cast<uint32_t>(millis());
      if (nowMs - encNoTickLogMs_ >= 2000) {
        Serial.printf(
            "[ENC] WARN powered uR=%.2f uL=%.2f but no ticks 0.5s+ "
            "(pcntR=%d pcntL=%d)\n",
            lastMotorVoltage_.x, lastMotorVoltage_.y,
            static_cast<int>(pcntRight), static_cast<int>(pcntLeft));
        encNoTickLogMs_ = nowMs;
      }
    }
  } else {
    encNoTickCount_ = 0;
  }
  // 符号修正统一应用到带符号 delta：ticks 累计与速度计算必须共用同一
  // 修正，否则改 SIGN 只影响遥测 ticks，v 方向纹丝不动（早期 bug）。
  const int64_t signedDeltaRight = UserConfig::RIGHT_ENCODER_SIGN * deltaRight;
  const int64_t signedDeltaLeft = UserConfig::LEFT_ENCODER_SIGN * deltaLeft;
  int64_t ticksRight = lastTicksRight_ + signedDeltaRight;
  int64_t ticksLeft = lastTicksLeft_ + signedDeltaLeft;
  lastTicksRight_ = ticksRight;
  lastTicksLeft_ = ticksLeft;

  const float metersPerTick =
      2.0f * PI_F * UserConfig::WHEEL_RADIUS_M /
      static_cast<float>(UserConfig::ENCODER_TICKS_PER_WHEEL_REV);
  const float rawRight = metersPerTick * static_cast<float>(signedDeltaRight) / dtSeconds;
  const float rawLeft = metersPerTick * static_cast<float>(signedDeltaLeft) / dtSeconds;
  constexpr float MAX_WHEEL_SPEED_MPS = 5.0f;
  const float clampedRight = clampf(rawRight, -MAX_WHEEL_SPEED_MPS, MAX_WHEEL_SPEED_MPS);
  const float clampedLeft = clampf(rawLeft, -MAX_WHEEL_SPEED_MPS, MAX_WHEEL_SPEED_MPS);
  const float encoderOmegaRaw =
      (clampedRight - clampedLeft) / UserConfig::TRACK_WIDTH_M;
  state_.velocityRaw.x = 0.5f * (clampedRight + clampedLeft);
  state_.wheelRight = lowPass(state_.wheelRight, clampedRight, config_.velocityTau, dtSeconds);
  state_.wheelLeft = lowPass(state_.wheelLeft, clampedLeft, config_.velocityTau, dtSeconds);

  const float encoderOmega =
      (state_.wheelRight - state_.wheelLeft) / UserConfig::TRACK_WIDTH_M;
  state_.velocity.x = lowPass(
      state_.velocity.x, 0.5f * (state_.wheelRight + state_.wheelLeft),
      config_.velocityTau, dtSeconds);
  // IMU 机制已取消（用户决策）：ω 全部由编码器差速计算，不再读取
  // MPU6050。编码器 ω 无零偏，不需要死区门控；直接低通后进入
  // theta 积分 / 控制器 / 回归量。imu_ok 恒 false（协议字段保留兼容）。
  state_.imuOk = false;
  state_.velocityRaw.y = encoderOmegaRaw;
  state_.velocity.y = lowPass(state_.velocity.y, encoderOmega,
                              config_.velocityTau, dtSeconds);
  state_.gyroZ = 0.0f;
  gyroRawRadS_ = 0.0f;
  gyroFilteredRadS_ = 0.0f;

  const bool inaReadOk = readIna3221();
  state_.inaOk = inaReadOk;
  if (lastMotorVoltage_.x > 0.10f) rightCurrentSign_ = 1.0f;
  if (lastMotorVoltage_.x < -0.10f) rightCurrentSign_ = -1.0f;
  if (lastMotorVoltage_.y > 0.10f) leftCurrentSign_ = 1.0f;
  if (lastMotorVoltage_.y < -0.10f) leftCurrentSign_ = -1.0f;
  const float measuredRight = rightCurrentSign_ *
                              std::fabs(state_.channelCurrent[UserConfig::INA_CHANNEL_RIGHT]);
  const float measuredLeft = leftCurrentSign_ *
                             std::fabs(state_.channelCurrent[UserConfig::INA_CHANNEL_LEFT]);
  constexpr float MAX_CURRENT_A = 5.0f;
  constexpr float CURRENT_JUMP_A = 1.0f;
  float rawCurrentRight = measuredRight;
  float rawCurrentLeft = measuredLeft;
  if (!std::isfinite(rawCurrentRight) || std::fabs(rawCurrentRight) > MAX_CURRENT_A ||
      std::fabs(rawCurrentRight - state_.current.x) > CURRENT_JUMP_A) {
    rawCurrentRight = state_.current.x;
  }
  if (!std::isfinite(rawCurrentLeft) || std::fabs(rawCurrentLeft) > MAX_CURRENT_A ||
      std::fabs(rawCurrentLeft - state_.current.y) > CURRENT_JUMP_A) {
    rawCurrentLeft = state_.current.y;
  }
  state_.currentRaw = {rawCurrentRight, rawCurrentLeft};
  state_.current.x = lowPass(state_.current.x, rawCurrentRight, config_.currentTau, dtSeconds);
  state_.current.y = lowPass(state_.current.y, rawCurrentLeft, config_.currentTau, dtSeconds);

  if (firstSample_) {
    state_.velocityDot = {};
    state_.currentDot = {};
    velocityTracked_ = state_.velocity;
    currentTracked_ = state_.current;
    firstSample_ = false;
  } else {
    auto observerStep = [dt = dtSeconds, tau = config_.derivativeTau](
                            float value, float& tracked) -> float {
      const float dot = (value - tracked) / tau;
      tracked += (dt / tau) * (value - tracked);
      return dot;
    };
    state_.velocityDot.x = observerStep(state_.velocity.x, velocityTracked_.x);
    state_.velocityDot.y = observerStep(state_.velocity.y, velocityTracked_.y);
    state_.currentDot.x = observerStep(state_.current.x, currentTracked_.x);
    state_.currentDot.y = observerStep(state_.current.y, currentTracked_.y);
  }

  state_.pose.theta = wrapAngle(state_.pose.theta + state_.velocity.y * dtSeconds);
  state_.pose.x += state_.velocity.x * std::cos(state_.pose.theta) * dtSeconds;
  state_.pose.y += state_.velocity.x * std::sin(state_.pose.theta) * dtSeconds;
  state_.ticksRight = ticksRight;
  state_.ticksLeft = ticksLeft;
  state_.timestampUs = esp_timer_get_time();
  return inaReadOk;
}

void VehicleIO::initializePwm() {
  pinMode(UserConfig::PIN_RIGHT_IN1, OUTPUT);
  pinMode(UserConfig::PIN_RIGHT_IN2, OUTPUT);
  pinMode(UserConfig::PIN_LEFT_IN1, OUTPUT);
  pinMode(UserConfig::PIN_LEFT_IN2, OUTPUT);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttachChannel(UserConfig::PIN_RIGHT_IN1, UserConfig::PWM_FREQUENCY_HZ,
                    UserConfig::PWM_RESOLUTION_BITS, PWM_CHANNEL_RIGHT_IN1);
  ledcAttachChannel(UserConfig::PIN_RIGHT_IN2, UserConfig::PWM_FREQUENCY_HZ,
                    UserConfig::PWM_RESOLUTION_BITS, PWM_CHANNEL_RIGHT_IN2);
  ledcAttachChannel(UserConfig::PIN_LEFT_IN1, UserConfig::PWM_FREQUENCY_HZ,
                    UserConfig::PWM_RESOLUTION_BITS, PWM_CHANNEL_LEFT_IN1);
  ledcAttachChannel(UserConfig::PIN_LEFT_IN2, UserConfig::PWM_FREQUENCY_HZ,
                    UserConfig::PWM_RESOLUTION_BITS, PWM_CHANNEL_LEFT_IN2);
#else
  ledcSetup(PWM_CHANNEL_RIGHT_IN1, UserConfig::PWM_FREQUENCY_HZ,
            UserConfig::PWM_RESOLUTION_BITS);
  ledcSetup(PWM_CHANNEL_RIGHT_IN2, UserConfig::PWM_FREQUENCY_HZ,
            UserConfig::PWM_RESOLUTION_BITS);
  ledcSetup(PWM_CHANNEL_LEFT_IN1, UserConfig::PWM_FREQUENCY_HZ,
            UserConfig::PWM_RESOLUTION_BITS);
  ledcSetup(PWM_CHANNEL_LEFT_IN2, UserConfig::PWM_FREQUENCY_HZ,
            UserConfig::PWM_RESOLUTION_BITS);
  ledcAttachPin(UserConfig::PIN_RIGHT_IN1, PWM_CHANNEL_RIGHT_IN1);
  ledcAttachPin(UserConfig::PIN_RIGHT_IN2, PWM_CHANNEL_RIGHT_IN2);
  ledcAttachPin(UserConfig::PIN_LEFT_IN1, PWM_CHANNEL_LEFT_IN1);
  ledcAttachPin(UserConfig::PIN_LEFT_IN2, PWM_CHANNEL_LEFT_IN2);
#endif
}

void VehicleIO::writeBridge(int pinIn1, int pinIn2, int channelIn1,
                            int channelIn2, float voltage, int motorSign,
                            bool& active, int8_t& direction) {
  if (std::fabs(voltage) <= config_.motorDeadzone) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(pinIn1, 0);
    ledcWrite(pinIn2, 0);
#else
    ledcWrite(channelIn1, 0);
    ledcWrite(channelIn2, 0);
#endif
    active = false;
    direction = 0;
    return;
  }
  const float measuredSupply =
      fmaxf(state_.busVoltage[UserConfig::INA_CHANNEL_LEFT],
            state_.busVoltage[UserConfig::INA_CHANNEL_RIGHT]);
  const float supply =
      measuredSupply > 6.5f ? measuredSupply : UserConfig::NOMINAL_BATTERY_V;
  const float signedVoltage = motorSign * voltage;
  const float dutyRatio = clampf(std::fabs(signedVoltage) / supply, 0.0f, 1.0f);
  const uint32_t maxDuty = (1u << UserConfig::PWM_RESOLUTION_BITS) - 1u;
  const uint32_t duty = static_cast<uint32_t>(std::lround(dutyRatio * maxDuty));
  const int8_t requestedDirection = signedVoltage >= 0.0f ? 1 : -1;

  if (duty == 0) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(pinIn1, 0);
    ledcWrite(pinIn2, 0);
#else
    ledcWrite(channelIn1, 0);
    ledcWrite(channelIn2, 0);
#endif
    active = false;
    direction = 0;
    return;
  }

  if (!active || direction != requestedDirection) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(pinIn1, maxDuty);
    ledcWrite(pinIn2, maxDuty);
#else
    ledcWrite(channelIn1, maxDuty);
    ledcWrite(channelIn2, maxDuty);
#endif
    delayMicroseconds(UserConfig::DRV8871_WAKE_PULSE_US);
    active = true;
    direction = requestedDirection;
  }

  const uint32_t brakeDuty = maxDuty - duty;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  if (signedVoltage >= 0.0f) {
    ledcWrite(pinIn1, maxDuty);
    ledcWrite(pinIn2, brakeDuty);
  } else {
    ledcWrite(pinIn1, brakeDuty);
    ledcWrite(pinIn2, maxDuty);
  }
#else
  if (signedVoltage >= 0.0f) {
    ledcWrite(channelIn1, maxDuty);
    ledcWrite(channelIn2, brakeDuty);
  } else {
    ledcWrite(channelIn1, brakeDuty);
    ledcWrite(channelIn2, maxDuty);
  }
#endif
}

void VehicleIO::setMotorVoltages(const Vec2& voltage) {
  Vec2 appliedVoltage{
      clampf(voltage.x, -UserConfig::MAX_MOTOR_COMMAND_V,
             UserConfig::MAX_MOTOR_COMMAND_V),
      clampf(voltage.y, -UserConfig::MAX_MOTOR_COMMAND_V,
             UserConfig::MAX_MOTOR_COMMAND_V),
  };
  if (std::fabs(appliedVoltage.x) <= config_.motorDeadzone) {
    appliedVoltage.x = 0.0f;
  }
  if (std::fabs(appliedVoltage.y) <= config_.motorDeadzone) {
    appliedVoltage.y = 0.0f;
  }
  writeBridge(UserConfig::PIN_RIGHT_IN1, UserConfig::PIN_RIGHT_IN2,
              PWM_CHANNEL_RIGHT_IN1, PWM_CHANNEL_RIGHT_IN2, appliedVoltage.x,
              UserConfig::RIGHT_MOTOR_SIGN, rightBridgeActive_,
              rightBridgeDirection_);
  writeBridge(UserConfig::PIN_LEFT_IN1, UserConfig::PIN_LEFT_IN2,
              PWM_CHANNEL_LEFT_IN1, PWM_CHANNEL_LEFT_IN2, appliedVoltage.y,
              UserConfig::LEFT_MOTOR_SIGN, leftBridgeActive_,
              leftBridgeDirection_);
  // Keep actuator feedback and integral snapshots consistent with the voltage
  // actually applied after saturation and dead-zone suppression.
  lastMotorVoltage_ = appliedVoltage;
}

void VehicleIO::stopMotors() {
  lastMotorVoltage_ = {};
  rightBridgeActive_ = false;
  leftBridgeActive_ = false;
  rightBridgeDirection_ = 0;
  leftBridgeDirection_ = 0;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(UserConfig::PIN_RIGHT_IN1, 0);
  ledcWrite(UserConfig::PIN_RIGHT_IN2, 0);
  ledcWrite(UserConfig::PIN_LEFT_IN1, 0);
  ledcWrite(UserConfig::PIN_LEFT_IN2, 0);
#else
  ledcWrite(PWM_CHANNEL_RIGHT_IN1, 0);
  ledcWrite(PWM_CHANNEL_RIGHT_IN2, 0);
  ledcWrite(PWM_CHANNEL_LEFT_IN1, 0);
  ledcWrite(PWM_CHANNEL_LEFT_IN2, 0);
#endif
}

void VehicleIO::zeroPose() {
  setPose({});
  // PCNT 硬件计数清零；lastPcnt/累计 ticks 同步归零，避免清零后的首个
  // 采样周期产生虚假增量。
  pcnt_counter_pause(PCNT_UNIT_0);
  pcnt_counter_pause(PCNT_UNIT_1);
  pcnt_counter_clear(PCNT_UNIT_0);
  pcnt_counter_clear(PCNT_UNIT_1);
  pcnt_counter_resume(PCNT_UNIT_0);
  pcnt_counter_resume(PCNT_UNIT_1);
  lastPcntRight_ = 0;
  lastPcntLeft_ = 0;
  lastTicksRight_ = 0;
  lastTicksLeft_ = 0;
  // zero_pose is issued only after the motors have been stopped.  Clear the
  // complete encoder-velocity filter state as well as the counters so a
  // decaying nonzero float cannot enter the discontinuous sgn(.) regressor as
  // +/-1 at the beginning of the next independent trial.
  state_.wheelRight = 0.0f;
  state_.wheelLeft = 0.0f;
  state_.velocityRaw = {};
  state_.velocity = {};
  state_.velocityDot = {};
  velocityTracked_ = {};
  firstSample_ = true;
}

void VehicleIO::setPose(const Pose2D& pose) {
  state_.pose = pose;
  state_.pose.theta = wrapAngle(state_.pose.theta);
}

bool VehicleIO::calibrateGyro(uint16_t samples) {
  // IMU 机制已取消（用户决策）：校准为空操作，立即成功，保持流程兼容。
  (void)samples;
  state_.imuCalibrated = false;
  return true;
}

bool VehicleIO::initializeIna3221() {
  writeI2c16(UserConfig::INA3221_ADDRESS, 0x00, 0x8000);
  delay(2);
  if (!writeI2c16(UserConfig::INA3221_ADDRESS, 0x00, 0x7097)) return false;
  uint16_t manufacturer = 0;
  uint16_t die = 0;
  if (!readI2c16(UserConfig::INA3221_ADDRESS, 0xFE, manufacturer)) return false;
  if (!readI2c16(UserConfig::INA3221_ADDRESS, 0xFF, die)) return false;
  return manufacturer == 0x5449 && die == 0x3220;
}

bool VehicleIO::initializeMpu6050() {
  uint8_t who = 0;
  if (!readI2c(UserConfig::MPU6050_ADDRESS, 0x75, &who, 1)) return false;
  if ((who & 0x7E) != 0x68) return false;
  return writeI2c8(UserConfig::MPU6050_ADDRESS, 0x6B, 0x00) &&
         writeI2c8(UserConfig::MPU6050_ADDRESS, 0x19, 0x04) &&
         writeI2c8(UserConfig::MPU6050_ADDRESS, 0x1A, 0x04) &&
         writeI2c8(UserConfig::MPU6050_ADDRESS, 0x1B, 0x08) &&
         writeI2c8(UserConfig::MPU6050_ADDRESS, 0x1C, 0x08);
}

bool VehicleIO::readIna3221() {
  bool ok = true;
  for (int channel = 0; channel < 3; ++channel) {
    uint16_t rawShunt = 0;
    uint16_t rawBus = 0;
    ok = readI2c16(UserConfig::INA3221_ADDRESS, 0x01 + 2 * channel, rawShunt) && ok;
    ok = readI2c16(UserConfig::INA3221_ADDRESS, 0x02 + 2 * channel, rawBus) && ok;
    const int16_t signedShunt = static_cast<int16_t>(rawShunt);
    const int16_t signedBus = static_cast<int16_t>(rawBus);
    if (signedShunt == lastRawShunt_[channel]) {
      if (++shuntStuckCount_[channel] >= STUCK_THRESHOLD) ok = false;
    } else {
      shuntStuckCount_[channel] = 0;
      lastRawShunt_[channel] = signedShunt;
    }
    if (signedBus == lastRawBus_[channel]) {
      if (++busStuckCount_[channel] >= STUCK_THRESHOLD) ok = false;
    } else {
      busStuckCount_[channel] = 0;
      lastRawBus_[channel] = signedBus;
    }
    const float shuntVoltage = static_cast<float>(signedShunt) * 5.0e-6f;
    state_.channelCurrent[channel] = shuntVoltage / UserConfig::INA_SHUNT_OHM;
    state_.busVoltage[channel] = static_cast<float>(signedBus) * 1.0e-3f;
  }
  return ok;
}

bool VehicleIO::readMpu6050(float& gyroZRadS) {
  uint8_t data[14] = {};
  if (!readI2c(UserConfig::MPU6050_ADDRESS, 0x3B, data, sizeof(data))) return false;
  {
    bool allZero = true, allFF = true;
    for (size_t i = 0; i < sizeof(data); ++i) {
      if (data[i] != 0x00) allZero = false;
      if (data[i] != 0xFF) allFF = false;
    }
    if (allZero || allFF) return false;
  }
  const int16_t rawGz = static_cast<int16_t>((data[12] << 8) | data[13]);
  // Repeated quantized gyro samples are valid while the car is stationary or
  // moving slowly.  Treating equality as a sensor fault caused false IMU
  // failures and silently switched the pose estimator to bad encoder yaw.
  const float measured = UserConfig::GYRO_Z_SIGN *
                         (static_cast<float>(rawGz) / 65.5f) * DEG_TO_RAD_F;
  gyroZRadS = measured - gyroBiasRadS_;
  return std::isfinite(gyroZRadS);
}

bool VehicleIO::writeI2c8(uint8_t address, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool VehicleIO::writeI2c16(uint8_t address, uint8_t reg, uint16_t value) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  Wire.write(static_cast<uint8_t>(value >> 8));
  Wire.write(static_cast<uint8_t>(value & 0xFF));
  return Wire.endTransmission() == 0;
}

bool VehicleIO::readI2c(uint8_t address, uint8_t reg, uint8_t* data,
                        size_t length) {
  Wire.beginTransmission(address);
  Wire.write(reg);
  if (Wire.endTransmission(true) != 0) return false;
  const size_t received = Wire.requestFrom(address, static_cast<uint8_t>(length),
                                           static_cast<uint8_t>(true));
  if (received != length) return false;
  for (size_t i = 0; i < length; ++i) data[i] = Wire.read();
  return true;
}

bool VehicleIO::readI2c16(uint8_t address, uint8_t reg, uint16_t& value) {
  uint8_t data[2] = {};
  if (!readI2c(address, reg, data, sizeof(data))) return false;
  value = (static_cast<uint16_t>(data[0]) << 8) | data[1];
  return true;
}

// ===== src/theory_controller.cpp =====
namespace {
constexpr float PI_F_TC = 3.14159265358979323846f;
constexpr float REFERENCE_SPEED_EPS = 1.0e-4f;
// Keep this sign basis identical to the offline reconstruction and the
// simulation: the paper's Coulomb-friction basis is sgn(s) with sgn(0)=0.
}

uint32_t TheoryController::checksum(const ControllerWeights& weights) {
  ControllerWeights copy = weights;
  copy.checksum = 0;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&copy);
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < sizeof(copy); ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

bool TheoryController::validate(const ControllerWeights& weights) {
  if (weights.magic != 0x4D494D4Fu || weights.version != 4) return false;
  if (weights.rank2 != 10 || weights.rank3 != 10) return false;
  if (!std::isfinite(weights.kappa2) || !std::isfinite(weights.kappa3) ||
      !std::isfinite(weights.kappaMin2) || !std::isfinite(weights.kappaMin3) ||
      weights.kappaMin2 < 0.0f || weights.kappaMin3 < 0.0f ||
      weights.kappa2 <= weights.kappaMin2 || weights.kappa3 <= weights.kappaMin3) {
    return false;
  }
  if (!std::isfinite(weights.sigmaMargin2) ||
      !std::isfinite(weights.sigmaMargin3) ||
      weights.sigmaMargin2 <= 0.0f || weights.sigmaMargin3 <= 0.0f) return false;
  if (!std::isfinite(weights.epsilon2) || !std::isfinite(weights.epsilon3) ||
      weights.epsilon2 <= 0.0f || weights.epsilon3 <= 0.0f) return false;
  if (!std::isfinite(weights.dbar2) || !std::isfinite(weights.dbar3) ||
      weights.dbar2 <= 0.0f || weights.dbar3 <= 0.0f) return false;
  if (!std::isfinite(weights.certificateMax2) ||
      !std::isfinite(weights.certificateMax3)) return false;
  if (weights.certificateMax2 > 0.0f || weights.certificateMax3 > 0.0f) {
    return false;
  }
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 8; ++col) {
      if (!std::isfinite(weights.w2[row][col]) ||
          !std::isfinite(weights.w3[row][col])) return false;
    }
  }
  return weights.checksum == 0 || weights.checksum == checksum(weights);
}

bool TheoryController::setWeights(const ControllerWeights& weights) {
  ControllerWeights candidate = weights;
  candidate.checksum = checksum(candidate);
  if (!validate(candidate)) {
    weightsValid_ = false;
    return false;
  }
  weights_ = candidate;
  weightsValid_ = true;
  initialized_ = false;
  return true;
}

float TheoryController::wrapAngle(float angle) {
  while (angle > PI_F_TC) angle -= 2.0f * PI_F_TC;
  while (angle < -PI_F_TC) angle += 2.0f * PI_F_TC;
  return angle;
}

float TheoryController::smoothSign(float value) {
  // Paper Coulomb-friction basis sgn(s), with sgn(0) = 0.
  return (value > 0.0f) - (value < 0.0f);
}

Vec2 TheoryController::matVec2x8(const float matrix[2][8], const float vector[8]) {
  Vec2 result{};
  for (int col = 0; col < 8; ++col) {
    result.x += matrix[0][col] * vector[col];
    result.y += matrix[1][col] * vector[col];
  }
  return result;
}

Vec2 TheoryController::matVec2x6(const float matrix[2][6], const float vector[6]) {
  Vec2 result{};
  for (int col = 0; col < 6; ++col) {
    result.x += matrix[0][col] * vector[col];
    result.y += matrix[1][col] * vector[col];
  }
  return result;
}

struct CircleProgress {
  float time;
  float rate;
  float acceleration;
};

static CircleProgress circleProgress(float t, float runDuration);

// Constant-speed phase schedule for ref_shape=3. The circle runs at unit
// phase rate from t=0 until the configured run duration, then the reference
// velocity is set to zero and the autonomous stop is issued at that pose.
static CircleProgress circleProgress(float t, float runDuration) {
  if (t >= runDuration) return {runDuration, 0.0f, 0.0f};
  return {fmaxf(0.0f, t), 1.0f, 0.0f};
}

Vec2 TheoryController::referencePosition(float t) const {
  const float a = config_.refA;
  const float b = config_.refB;
  const float nu = config_.refNu;
  constexpr float REFERENCE_RAMP_S = 1.5f;
  const float boundedTime = clampf(t, 0.0f, config_.runDuration);
  if (boundedTime <= REFERENCE_RAMP_S && config_.refShape <= 2.5f)
    return Vec2{};
  if (config_.refShape > 2.5f) {
    // 车辆从圆心出发，参考从圆周点 (R,0) 出发，并自 t=0 按恒定相位率运行。
    const CircleProgress progress = circleProgress(boundedTime, config_.runDuration);
    return {a * std::cos(nu * progress.time), a * std::sin(nu * progress.time)};
  }
  const float tau = boundedTime - REFERENCE_RAMP_S;
  if (config_.refShape > 1.5f) {
    const float phi = nu * tau;
    return {a * std::sin(2.0f * phi) * std::cos(phi),
            a * std::sin(2.0f * phi) * std::sin(phi)};
  }
  if (config_.refShape > 0.5f) {
    return {a * std::sin(nu * tau),
            a * (1.0f - std::cos(nu * tau))};
  }
  if (b <= 0.0f) {
    return {a / nu * (1.0f - std::cos(nu * tau)), 0.0f};
  }
  return {a * std::sin(nu * tau),
          b * std::sin(2.0f * nu * tau)};
}

void TheoryController::reference(float t, Pose2D& pose, Vec2& velocity,
                                 Vec2& cartesianVelocity,
                                 Vec2& cartesianAcceleration) {
  const float a = config_.refA;
  const float b = config_.refB;
  const float nu = config_.refNu;
  // 启动斜坡：位置暂停在起点，航向从车当前航向平滑转到参考切线，
  // 速度为零；斜坡结束后轨迹时钟从起点重新开始。
  constexpr float REFERENCE_RAMP_S = 1.5f;
  if (t < REFERENCE_RAMP_S && config_.refShape <= 2.5f) {
    const float u = t / REFERENCE_RAMP_S;
    const float w = u * u * u * (10.0f - 15.0f * u + 6.0f * u * u);
    const float wDot =
        30.0f * u * u * (1.0f - u) * (1.0f - u) / REFERENCE_RAMP_S;
    const float theta0 =
        config_.refShape > 2.5f ? 1.5707963f
        : b > 0.0f ? std::atan2(2.0f * b * nu, a * nu) : 0.0f;
    pose.x = config_.refShape > 2.5f ? a : 0.0f;
    pose.y = 0.0f;
    pose.theta = refStartTheta_ + wrapAngle(theta0 - refStartTheta_) * w;
    velocity.x = 0.0f;
    velocity.y = wrapAngle(theta0 - refStartTheta_) * wDot;
    cartesianVelocity = {};
    cartesianAcceleration = {};
    lastReferenceTheta_ = theta0;
    return;
  }
  const float tau = t - REFERENCE_RAMP_S;
  if (config_.refShape > 2.5f) {
    // 圆心起步匀速圆：x=R cos νt, y=R sin νt，圆心为零点位姿；
    // q_r(0)=(R,0) 切向 +y（θ_r(0)=π/2），从 t=0 即匀速运动，
    // 速度 v=Rν、角速度 ω=+ν 均恒定，θ_r=π/2+ντ 闭式延拓。
    // 参考从 t=0 即沿圆周匀速运动；PC 预设将 run_duration 设为一整圈
    // 加 3 s，因此额外 3 s 仍沿圆周运行，随后在当前圆周位置直接停车。
    const CircleProgress progress = circleProgress(t, config_.runDuration);
    const bool holding = t >= config_.runDuration;
    const float tauC = progress.time;
    const float phaseRate = nu * progress.rate;
    const float phaseAcceleration = nu * progress.acceleration;
    pose.x = a * std::cos(nu * tauC);
    pose.y = a * std::sin(nu * tauC);
    velocity.x = a * phaseRate;
    velocity.y = nu * phaseRate;
    if (holding) {
      cartesianVelocity = {};
      cartesianAcceleration = {};
    } else {
      cartesianVelocity = {
          -a * phaseRate * std::sin(nu * tauC),
          a * phaseRate * std::cos(nu * tauC),
      };
      cartesianAcceleration = {
          -a * phaseRate * phaseRate * std::cos(nu * tauC)
              - a * phaseAcceleration * std::sin(nu * tauC),
          -a * phaseRate * phaseRate * std::sin(nu * tauC)
              + a * phaseAcceleration * std::cos(nu * tauC),
      };
    }
    lastReferenceTheta_ = 1.5707963f + nu * tauC;
    pose.theta = lastReferenceTheta_;
    return;
  }
  if (config_.refShape > 1.5f) {
    // 四叶草：极坐标 r=R sin(2φ), φ=ντ。参数速度始终非零，
    // 因而参考航向和角速度可连续生成。
    const float phi = nu * tau;
    const float s = std::sin(phi);
    const float c = std::cos(phi);
    const float s2 = std::sin(2.0f * phi);
    const float c2 = std::cos(2.0f * phi);
    pose.x = a * s2 * c;
    pose.y = a * s2 * s;
    const float dx = a * nu * (2.0f * c2 * c - s2 * s);
    const float dy = a * nu * (2.0f * c2 * s + s2 * c);
    const float ddx = a * nu * nu * (-5.0f * s2 * c - 4.0f * c2 * s);
    const float ddy = a * nu * nu * (-5.0f * s2 * s + 4.0f * c2 * c);
    cartesianVelocity = {dx, dy};
    cartesianAcceleration = {ddx, ddy};
    const float speedSquared = dx * dx + dy * dy;
    velocity.x = std::sqrt(speedSquared);
    const float omegaRef = (dx * ddy - dy * ddx) / speedSquared;
    lastReferenceTheta_ += omegaRef * config_.dtControl;
    velocity.y = omegaRef;
    pose.theta = lastReferenceTheta_;
    return;
  }
  if (config_.refShape > 0.5f) {
    // 大圆：x=R sin ντ, y=R(1−cos ντ)，从原点出发逆时针画圆；
    // 速度 v=Rν、角速度 ω=ν 均恒定，θ=ντ 连续延拓
    pose.x = a * std::sin(nu * tau);
    pose.y = a * (1.0f - std::cos(nu * tau));
    velocity.x = a * nu;
    velocity.y = nu;
    cartesianVelocity = {
        a * nu * std::cos(nu * tau),
        a * nu * std::sin(nu * tau),
    };
    cartesianAcceleration = {
        -a * nu * nu * std::sin(nu * tau),
        a * nu * nu * std::cos(nu * tau),
    };
    lastReferenceTheta_ = nu * tau;
    pose.theta = lastReferenceTheta_;
    return;
  }
  if (b <= 0.0f) {
    // 直线往返：x=(a/ν)(1-cos ντ)，v=a sin ντ，θ=0
    pose.x = a / nu * (1.0f - std::cos(nu * tau));
    pose.y = 0.0f;
    pose.theta = 0.0f;
    velocity.x = a * std::sin(nu * tau);
    velocity.y = 0.0f;
    cartesianVelocity = {velocity.x, 0.0f};
    cartesianAcceleration = {a * nu * std::cos(nu * tau), 0.0f};
    lastReferenceTheta_ = 0.0f;
    return;
  }
  // 双纽线：θ 积分延拓（eq:flatomega），全程连续无跳变；v 恒正
  pose.x = a * std::sin(nu * tau);
  pose.y = b * std::sin(2.0f * nu * tau);
  const float dx = a * nu * std::cos(nu * tau);
  const float dy = 2.0f * b * nu * std::cos(2.0f * nu * tau);
  const float ddx = -a * nu * nu * std::sin(nu * tau);
  const float ddy = -4.0f * b * nu * nu * std::sin(2.0f * nu * tau);
  cartesianVelocity = {dx, dy};
  cartesianAcceleration = {ddx, ddy};
  const float speedSquared = dx * dx + dy * dy;
  velocity.x = std::sqrt(speedSquared);
  if (speedSquared > REFERENCE_SPEED_EPS * REFERENCE_SPEED_EPS) {
    const float omegaRef = (dx * ddy - dy * ddx) / speedSquared;
    lastReferenceTheta_ += omegaRef * config_.dtControl;
    velocity.y = omegaRef;
  } else {
    velocity.y = 0.0f;
  }
  pose.theta = lastReferenceTheta_;
}

Vec2 TheoryController::outerCommand(const SensorState& sensors,
                                    bool terminalCapture) {
  // Manuscript law: capture q_p=q_r(min(t+T_p,T_f)) in polar coordinates.
  const float targetErrorX = state_.reference.x - sensors.pose.x;
  const float targetErrorY = state_.reference.y - sensors.pose.y;
  const float previewErrorX = state_.motionReference.x - sensors.pose.x;
  const float previewErrorY = state_.motionReference.y - sensors.pose.y;
  const float rho = std::sqrt(previewErrorX * previewErrorX +
                              previewErrorY * previewErrorY);

  float alpha = 0.0f;
  float radialSpeed = 0.0f;
  float curvatureTerm = 0.0f;
  if (rho > REFERENCE_SPEED_EPS) {
    alpha = wrapAngle(std::atan2(previewErrorY, previewErrorX) -
                      sensors.pose.theta);
    radialSpeed = config_.outerVbar *
        std::tanh(config_.outerKp * rho / config_.outerVbar);
    curvatureTerm = radialSpeed * std::sin(alpha) * std::cos(alpha) / rho;
  }

  float scale = 1.0f;
  if (terminalCapture) {
    if (rho <= config_.outerCaptureRadius) {
      scale = 0.0f;
    } else if (rho < config_.outerBlendRadius) {
      scale = (rho - config_.outerCaptureRadius) /
              (config_.outerBlendRadius - config_.outerCaptureRadius);
    }
  }

  Vec2 command{
      scale * radialSpeed * std::cos(alpha),
      scale * (config_.outerKtheta * alpha + curvatureTerm),
  };

  // pose_error reports the time-synchronized target error; motion_pose_error
  // reports the preview-point capture error used by the implemented law.
  state_.poseError[0] = targetErrorX;
  state_.poseError[1] = targetErrorY;
  state_.poseError[2] = alpha;
  state_.motionPoseError[0] = previewErrorX;
  state_.motionPoseError[1] = previewErrorY;
  state_.motionPoseError[2] = alpha;

  // These are emergency real-car guards.  With the configured admissible
  // references they remain inactive, so the commanded signal is the one in
  // the manuscript.
  command.x = clampf(command.x, -config_.outerMaxV, config_.outerMaxV);
  command.y = clampf(
      command.y, -config_.outerMaxOmega, config_.outerMaxOmega);
  return command;
}

void TheoryController::sigma2(const SensorState& sensors, const Vec2& betaDot,
                              float out[6]) const {
  out[0] = sensors.velocity.x;
  out[1] = sensors.velocity.y;
  out[2] = smoothSign(sensors.velocity.x);
  out[3] = smoothSign(sensors.velocity.y);
  out[4] = betaDot.x;
  out[5] = betaDot.y;
}

void TheoryController::sigma3(const SensorState& sensors, const Vec2& betaDot,
                              float out[6]) const {
  out[0] = sensors.current.x;
  out[1] = sensors.current.y;
  out[2] = sensors.velocity.x +
           0.5f * UserConfig::TRACK_WIDTH_M * sensors.velocity.y;
  out[3] = sensors.velocity.x -
           0.5f * UserConfig::TRACK_WIDTH_M * sensors.velocity.y;
  out[4] = betaDot.x;
  out[5] = betaDot.y;
}

void TheoryController::reset(const SensorState& sensors, float runTimeSeconds) {
  state_ = {};
  pidVelocityIntegral_ = {};
  pidVelocityPreviousError_ = {};
  pidVelocityDerivative_ = {};
  pidCurrentIntegral_ = {};
  pidCurrentPreviousError_ = {};
  pidCurrentDerivative_ = {};
  refStartTheta_ = sensors.pose.theta;  // 启动斜坡：航向从车当前航向开始
  Vec2 targetVelocity{};
  Vec2 cartesianVelocity{};
  Vec2 cartesianAcceleration{};
  reference(runTimeSeconds, state_.reference, targetVelocity,
            cartesianVelocity, cartesianAcceleration);
  const float previewTime = clampf(
      runTimeSeconds + config_.outerPreviewHorizon,
      0.0f, config_.runDuration);
  const Vec2 previewPosition = referencePosition(previewTime);
  state_.motionReference = {
      previewPosition.x, previewPosition.y, state_.reference.theta};
  const bool terminalCapture =
      previewTime >= config_.runDuration - 1.0e-4f;
  state_.motionVelocity = terminalCapture ? Vec2{} : targetVelocity;
  state_.alpha1 = outerCommand(sensors, terminalCapture);
  state_.beta1 = sensors.velocity;
  state_.z2 = {sensors.velocity.x - state_.beta1.x,
               sensors.velocity.y - state_.beta1.y};
  float sigma[6] = {};
  sigma2(sensors, {}, sigma);
  float regressor[8] = {state_.z2.x, state_.z2.y,
                        sigma[0], sigma[1], sigma[2], sigma[3], sigma[4], sigma[5]};
  if (config_.comparisonCase >= 2.5f) {
    // E3 starts its current-reference filter from the measured current.
    state_.alpha2 = sensors.current;
  } else {
    state_.alpha2 = weightsValid_ ? matVec2x8(weights_.w2, regressor) : Vec2{};
  }
  state_.beta2 = state_.alpha2;
  initialized_ = true;
}

Vec2 TheoryController::step(const SensorState& sensors, float runTimeSeconds,
                            float dtSeconds) {
  if (!initialized_) reset(sensors, runTimeSeconds);
  if (config_.comparisonCase >= 2.5f) {
    return stepCascadedPid(sensors, runTimeSeconds, dtSeconds);
  }
  Vec2 targetVelocity{};
  Vec2 cartesianVelocity{};
  Vec2 cartesianAcceleration{};
  reference(runTimeSeconds, state_.reference, targetVelocity,
            cartesianVelocity, cartesianAcceleration);
  const float previewTime = clampf(
      runTimeSeconds + config_.outerPreviewHorizon,
      0.0f, config_.runDuration);
  const Vec2 previewPosition = referencePosition(previewTime);
  state_.motionReference = {
      previewPosition.x, previewPosition.y, state_.reference.theta};
  const bool terminalCapture =
      previewTime >= config_.runDuration - 1.0e-4f;
  state_.motionVelocity = terminalCapture ? Vec2{} : targetVelocity;
  state_.alpha1 = outerCommand(sensors, terminalCapture);
  state_.beta1Dot = {
      (state_.alpha1.x - state_.beta1.x) / config_.tau1,
      (state_.alpha1.y - state_.beta1.y) / config_.tau1,
  };
  state_.z2 = {sensors.velocity.x - state_.beta1.x,
               sensors.velocity.y - state_.beta1.y};

  float sig2[6] = {};
  sigma2(sensors, state_.beta1Dot, sig2);
  float reg2[8] = {state_.z2.x, state_.z2.y,
                   sig2[0], sig2[1], sig2[2], sig2[3], sig2[4], sig2[5]};
  state_.alpha2 = matVec2x8(weights_.w2, reg2);
  const Vec2 comp2 = matVec2x6(state_.mhat2, sig2);
  state_.alpha2.x -= comp2.x;
  state_.alpha2.y -= comp2.y;
  state_.beta2Dot = {
      (state_.alpha2.x - state_.beta2.x) / config_.tau2,
      (state_.alpha2.y - state_.beta2.y) / config_.tau2,
  };
  state_.z3 = {sensors.current.x - state_.beta2.x,
               sensors.current.y - state_.beta2.y};

  float sig3[6] = {};
  sigma3(sensors, state_.beta2Dot, sig3);
  float reg3[8] = {state_.z3.x, state_.z3.y,
                   sig3[0], sig3[1], sig3[2], sig3[3], sig3[4], sig3[5]};
  state_.uc = matVec2x8(weights_.w3, reg3);
  const Vec2 comp3 = matVec2x6(state_.mhat3, sig3);
  state_.uc.x -= comp3.x;
  state_.uc.y -= comp3.y;
  state_.u = {
      clampf(state_.uc.x, -config_.uMax, config_.uMax),
      clampf(state_.uc.y, -config_.uMax, config_.uMax),
  };

  const Vec2 t2z = {0.5f * (state_.z2.x + state_.z2.y),
                    0.5f * (state_.z2.x - state_.z2.y)};
  // E2 is enforced in firmware: uploaded/nonzero EXE rates cannot
  // accidentally re-enable adaptation during the ablation experiment.
  const float adaptiveR2 = config_.comparisonCase >= 1.5f ? 0.0f : config_.r2;
  const float adaptiveR3 = config_.comparisonCase >= 1.5f ? 0.0f : config_.r3;
  for (int row = 0; row < 2; ++row) {
    const float driven2 = row == 0 ? t2z.x : t2z.y;
    const float driven3 = row == 0 ? state_.z3.x : state_.z3.y;
    for (int col = 0; col < 6; ++col) {
      state_.mhat2[row][col] += dtSeconds * adaptiveR2 *
          (driven2 * sig2[col] - config_.delta2 * state_.mhat2[row][col]);
      state_.mhat3[row][col] += dtSeconds * adaptiveR3 *
          (driven3 * sig3[col] - config_.delta3 * state_.mhat3[row][col]);
    }
  }

  state_.beta1.x += dtSeconds * state_.beta1Dot.x;
  state_.beta1.y += dtSeconds * state_.beta1Dot.y;
  state_.beta2.x += dtSeconds * state_.beta2Dot.x;
  state_.beta2.y += dtSeconds * state_.beta2Dot.y;
  return state_.u;
}

Vec2 TheoryController::stepCascadedPid(const SensorState& sensors,
                                       float runTimeSeconds,
                                       float dtSeconds) {
  Vec2 targetVelocity{};
  Vec2 cartesianVelocity{};
  Vec2 cartesianAcceleration{};
  reference(runTimeSeconds, state_.reference, targetVelocity,
            cartesianVelocity, cartesianAcceleration);
  const float previewTime = clampf(
      runTimeSeconds + config_.outerPreviewHorizon,
      0.0f, config_.runDuration);
  const Vec2 previewPosition = referencePosition(previewTime);
  state_.motionReference = {
      previewPosition.x, previewPosition.y, state_.reference.theta};
  const bool terminalCapture =
      previewTime >= config_.runDuration - 1.0e-4f;
  state_.motionVelocity = terminalCapture ? Vec2{} : targetVelocity;
  state_.alpha1 = outerCommand(sensors, terminalCapture);
  state_.beta1Dot = {
      (state_.alpha1.x - state_.beta1.x) / config_.tau1,
      (state_.alpha1.y - state_.beta1.y) / config_.tau1,
  };
  state_.z2 = {sensors.velocity.x - state_.beta1.x,
               sensors.velocity.y - state_.beta1.y};

  const float halfTrack = 0.5f * UserConfig::TRACK_WIDTH_M;
  const Vec2 targetWheel = {
      state_.beta1.x + halfTrack * state_.beta1.y,
      state_.beta1.x - halfTrack * state_.beta1.y,
  };
  const Vec2 measuredWheel = {
      sensors.velocity.x + halfTrack * sensors.velocity.y,
      sensors.velocity.x - halfTrack * sensors.velocity.y,
  };
  const Vec2 velocityError = {
      targetWheel.x - measuredWheel.x,
      targetWheel.y - measuredWheel.y,
  };
  const float derivativeAlpha =
      dtSeconds / (config_.pidDerivativeTau + dtSeconds);
  const float safeDt = dtSeconds > 1.0e-4f ? dtSeconds : 1.0e-4f;
  const Vec2 velocityDerivativeRaw = {
      (velocityError.x - pidVelocityPreviousError_.x) / safeDt,
      (velocityError.y - pidVelocityPreviousError_.y) / safeDt,
  };
  pidVelocityDerivative_.x += derivativeAlpha *
      (velocityDerivativeRaw.x - pidVelocityDerivative_.x);
  pidVelocityDerivative_.y += derivativeAlpha *
      (velocityDerivativeRaw.y - pidVelocityDerivative_.y);
  pidVelocityPreviousError_ = velocityError;

  const Vec2 velocityIntegralCandidate = {
      pidVelocityIntegral_.x + dtSeconds * velocityError.x,
      pidVelocityIntegral_.y + dtSeconds * velocityError.y,
  };
  auto velocityPidOutput = [&](float target, float error, float integral,
                               float derivative) {
    return config_.pidVelocityFf * target +
           config_.pidVelocityKp * error +
           config_.pidVelocityKi * integral +
           config_.pidVelocityKd * derivative;
  };
  Vec2 currentReferenceRaw = {
      velocityPidOutput(targetWheel.x, velocityError.x,
                        velocityIntegralCandidate.x, pidVelocityDerivative_.x),
      velocityPidOutput(targetWheel.y, velocityError.y,
                        velocityIntegralCandidate.y, pidVelocityDerivative_.y),
  };
  auto integralMayAdvance = [](float raw, float limit, float error) {
    return std::fabs(raw) <= limit || (raw > limit && error < 0.0f) ||
           (raw < -limit && error > 0.0f);
  };
  if (integralMayAdvance(currentReferenceRaw.x, config_.pidCurrentRefMax,
                         velocityError.x)) {
    pidVelocityIntegral_.x = velocityIntegralCandidate.x;
  }
  if (integralMayAdvance(currentReferenceRaw.y, config_.pidCurrentRefMax,
                         velocityError.y)) {
    pidVelocityIntegral_.y = velocityIntegralCandidate.y;
  }
  currentReferenceRaw = {
      velocityPidOutput(targetWheel.x, velocityError.x,
                        pidVelocityIntegral_.x, pidVelocityDerivative_.x),
      velocityPidOutput(targetWheel.y, velocityError.y,
                        pidVelocityIntegral_.y, pidVelocityDerivative_.y),
  };
  state_.alpha2 = {
      clampf(currentReferenceRaw.x, -config_.pidCurrentRefMax,
             config_.pidCurrentRefMax),
      clampf(currentReferenceRaw.y, -config_.pidCurrentRefMax,
             config_.pidCurrentRefMax),
  };
  state_.beta2Dot = {
      (state_.alpha2.x - state_.beta2.x) / config_.tau2,
      (state_.alpha2.y - state_.beta2.y) / config_.tau2,
  };
  state_.z3 = {sensors.current.x - state_.beta2.x,
               sensors.current.y - state_.beta2.y};

  const Vec2 currentError = {-state_.z3.x, -state_.z3.y};
  const Vec2 currentDerivativeRaw = {
      (currentError.x - pidCurrentPreviousError_.x) / safeDt,
      (currentError.y - pidCurrentPreviousError_.y) / safeDt,
  };
  pidCurrentDerivative_.x += derivativeAlpha *
      (currentDerivativeRaw.x - pidCurrentDerivative_.x);
  pidCurrentDerivative_.y += derivativeAlpha *
      (currentDerivativeRaw.y - pidCurrentDerivative_.y);
  pidCurrentPreviousError_ = currentError;
  const Vec2 currentIntegralCandidate = {
      pidCurrentIntegral_.x + dtSeconds * currentError.x,
      pidCurrentIntegral_.y + dtSeconds * currentError.y,
  };
  auto currentPidOutput = [&](float wheelTarget, float error, float integral,
                              float derivative) {
    return config_.pidVoltageFf * wheelTarget +
           config_.pidCurrentKp * error +
           config_.pidCurrentKi * integral +
           config_.pidCurrentKd * derivative;
  };
  Vec2 voltageRaw = {
      currentPidOutput(targetWheel.x, currentError.x,
                       currentIntegralCandidate.x, pidCurrentDerivative_.x),
      currentPidOutput(targetWheel.y, currentError.y,
                       currentIntegralCandidate.y, pidCurrentDerivative_.y),
  };
  if (integralMayAdvance(voltageRaw.x, config_.uMax, currentError.x)) {
    pidCurrentIntegral_.x = currentIntegralCandidate.x;
  }
  if (integralMayAdvance(voltageRaw.y, config_.uMax, currentError.y)) {
    pidCurrentIntegral_.y = currentIntegralCandidate.y;
  }
  state_.uc = {
      currentPidOutput(targetWheel.x, currentError.x, pidCurrentIntegral_.x,
                       pidCurrentDerivative_.x),
      currentPidOutput(targetWheel.y, currentError.y, pidCurrentIntegral_.y,
                       pidCurrentDerivative_.y),
  };
  // Paper actuator map: componentwise metric projection onto [-uMax, uMax].
  state_.u = {
      clampf(state_.uc.x, -config_.uMax, config_.uMax),
      clampf(state_.uc.y, -config_.uMax, config_.uMax),
  };

  state_.beta1.x += dtSeconds * state_.beta1Dot.x;
  state_.beta1.y += dtSeconds * state_.beta1Dot.y;
  state_.beta2.x += dtSeconds * state_.beta2Dot.x;
  state_.beta2.y += dtSeconds * state_.beta2Dot.y;
  return state_.u;
}

Vec2 TheoryController::collectionExcitation(float t) const {
  const float scale = config_.collectScale;
  const float speedCmd =
      scale * (0.30f * std::sin(0.7f * t) +
               0.15f * std::sin(1.4f * t + 0.3f));
  const float diffCmd =
      scale * (0.90f * std::sin(0.6f * t + 0.5f) +
               0.35f * std::sin(1.8f * t + 1.2f));
  const float wheelDiff = 0.5f * UserConfig::TRACK_WIDTH_M * diffCmd;
  const float uRight = config_.uMax * std::tanh(
      (speedCmd + wheelDiff) * UserConfig::MOTOR_V_PER_MS / config_.uMax);
  const float uLeft = config_.uMax * std::tanh(
      (speedCmd - wheelDiff) * UserConfig::MOTOR_V_PER_MS / config_.uMax);
  return {uRight, uLeft};
}

void TheoryController::syntheticFilter(float t, int block, Vec2& beta,
                                       Vec2& betaDot) const {
  if (block == 2) {
    beta = {0.18f * std::sin(0.47f * t) + 0.06f * std::sin(1.31f * t),
            0.42f * std::sin(0.59f * t + 0.4f) +
                0.10f * std::sin(1.73f * t)};
    betaDot = {0.18f * 0.47f * std::cos(0.47f * t) +
                   0.06f * 1.31f * std::cos(1.31f * t),
               0.42f * 0.59f * std::cos(0.59f * t + 0.4f) +
                   0.10f * 1.73f * std::cos(1.73f * t)};
  } else {
    beta = {0.25f * std::sin(0.83f * t + 0.2f) +
                0.08f * std::sin(2.11f * t),
            0.23f * std::sin(0.71f * t - 0.5f) +
                0.09f * std::sin(1.89f * t + 0.7f)};
    betaDot = {0.25f * 0.83f * std::cos(0.83f * t + 0.2f) +
                   0.08f * 2.11f * std::cos(2.11f * t),
               0.23f * 0.71f * std::cos(0.71f * t - 0.5f) +
                   0.09f * 1.89f * std::cos(1.89f * t + 0.7f)};
  }
}

CollectionPoint TheoryController::collectionPoint(
    const SensorState& sensors, const Vec2& appliedVoltage, float t) const {
  CollectionPoint point{};
  point.valid = true;
  Vec2 beta1{};
  Vec2 dbeta1{};
  Vec2 beta2{};
  Vec2 dbeta2{};
  syntheticFilter(t, 2, beta1, dbeta1);
  syntheticFilter(t, 3, beta2, dbeta2);
  const Vec2 z2 = {sensors.velocity.x - beta1.x,
                   sensors.velocity.y - beta1.y};
  const Vec2 z3 = {sensors.current.x - beta2.x,
                   sensors.current.y - beta2.y};
  float sig2[6] = {};
  float sig3[6] = {};
  sigma2(sensors, dbeta1, sig2);
  sigma3(sensors, dbeta2, sig3);
  point.z2[0] = z2.x;
  point.z2[1] = z2.y;
  point.z3[0] = z3.x;
  point.z3[1] = z3.y;
  point.y2[0] = z2.x;
  point.y2[1] = z2.y;
  point.y3[0] = z3.x;
  point.y3[1] = z3.y;
  for (int i = 0; i < 6; ++i) {
    point.y2[2 + i] = sig2[i];
    point.y3[2 + i] = sig3[i];
  }
  point.x3[0] = sensors.current.x;
  point.x3[1] = sensors.current.y;
  point.x4[0] = appliedVoltage.x;
  point.x4[1] = appliedVoltage.y;
  point.velocityRaw[0] = sensors.velocityRaw.x;
  point.velocityRaw[1] = sensors.velocityRaw.y;
  point.currentRaw[0] = sensors.currentRaw.x;
  point.currentRaw[1] = sensors.currentRaw.y;
  return point;
}

class IntegralSnapshotAccumulator {
 public:
  void reset() {
    initialized_ = false;
    elapsedSeconds_ = 0.0f;
    memset(integralY2_, 0, sizeof(integralY2_));
    memset(integralX3_, 0, sizeof(integralX3_));
    memset(integralY3_, 0, sizeof(integralY3_));
    memset(integralX4_, 0, sizeof(integralX4_));
    memset(integralVelocityRaw_, 0, sizeof(integralVelocityRaw_));
    memset(integralCurrentRaw_, 0, sizeof(integralCurrentRaw_));
  }

  SnapshotFrame update(const CollectionPoint& point, uint64_t timestampUs,
                       float requestedWindowSeconds) {
    SnapshotFrame frame{};
    if (!point.valid || timestampUs == 0 ||
        !std::isfinite(requestedWindowSeconds) || requestedWindowSeconds <= 0.0f) {
      reset();
      return frame;
    }
    if (!initialized_) {
      start(point, timestampUs);
      return frame;
    }
    if (timestampUs <= lastTimestampUs_) return frame;
    const float dt = static_cast<float>(timestampUs - lastTimestampUs_) * 1.0e-6f;
    // A long control-loop gap invalidates the time integral.  Start a clean
    // non-overlapping window instead of silently bridging missing dynamics.
    if (!std::isfinite(dt) || dt <= 0.0f || dt > 0.05f) {
      reset();
      start(point, timestampUs);
      return frame;
    }
    trapezoid(prev_.y2, point.y2, integralY2_, 8, dt);
    trapezoid(prev_.x3, point.x3, integralX3_, 2, dt);
    trapezoid(prev_.y3, point.y3, integralY3_, 8, dt);
    trapezoid(prev_.x4, point.x4, integralX4_, 2, dt);
    trapezoid(prev_.velocityRaw, point.velocityRaw, integralVelocityRaw_, 2, dt);
    trapezoid(prev_.currentRaw, point.currentRaw, integralCurrentRaw_, 2, dt);
    elapsedSeconds_ += dt;
    prev_ = point;
    lastTimestampUs_ = timestampUs;
    if (elapsedSeconds_ + 1.0e-6f < requestedWindowSeconds) return frame;

    frame.valid = true;
    frame.windowSeconds = elapsedSeconds_;
    for (int i = 0; i < 2; ++i) {
      frame.zdot2[i] = (point.z2[i] - startZ2_[i]) / elapsedSeconds_;
      frame.zdot3[i] = (point.z3[i] - startZ3_[i]) / elapsedSeconds_;
      frame.x3[i] = static_cast<float>(integralX3_[i] / elapsedSeconds_);
      frame.x4[i] = static_cast<float>(integralX4_[i] / elapsedSeconds_);
      frame.velocityRaw[i] =
          static_cast<float>(integralVelocityRaw_[i] / elapsedSeconds_);
      frame.currentRaw[i] =
          static_cast<float>(integralCurrentRaw_[i] / elapsedSeconds_);
    }
    for (int i = 0; i < 8; ++i) {
      frame.y2[i] = static_cast<float>(integralY2_[i] / elapsedSeconds_);
      frame.y3[i] = static_cast<float>(integralY3_[i] / elapsedSeconds_);
    }
    // The endpoint of this window is the start of the next one.  No control
    // interval contributes to two snapshots.
    reset();
    start(point, timestampUs);
    return frame;
  }

 private:
  static void trapezoid(const float* before, const float* after,
                        double* integral, int count, float dt) {
    for (int i = 0; i < count; ++i) {
      integral[i] += 0.5 * static_cast<double>(before[i] + after[i]) * dt;
    }
  }

  void start(const CollectionPoint& point, uint64_t timestampUs) {
    initialized_ = true;
    elapsedSeconds_ = 0.0f;
    lastTimestampUs_ = timestampUs;
    prev_ = point;
    memcpy(startZ2_, point.z2, sizeof(startZ2_));
    memcpy(startZ3_, point.z3, sizeof(startZ3_));
  }

  bool initialized_ = false;
  uint64_t lastTimestampUs_ = 0;
  float elapsedSeconds_ = 0.0f;
  float startZ2_[2] = {};
  float startZ3_[2] = {};
  CollectionPoint prev_{};
  double integralY2_[8] = {};
  double integralX3_[2] = {};
  double integralY3_[8] = {};
  double integralX4_[2] = {};
  double integralVelocityRaw_[2] = {};
  double integralCurrentRaw_[2] = {};
};

// ===== src/main.cpp =====
namespace {

constexpr uint32_t CONFIG_MAGIC = 0x43464731u;
constexpr uint16_t PROTOCOL_VERSION = 1;
constexpr uint32_t NETWORK_TIMEOUT_MS = 5000;
constexpr uint32_t COLLECTION_WARMUP_MS = 1000;
constexpr uint32_t MANUAL_SNAPSHOT_WARMUP_MS = 150;
constexpr size_t TELEMETRY_QUEUE_LENGTH = 8;
constexpr size_t USB_FRAME_QUEUE_LENGTH = 64;
constexpr size_t JSON_CAPACITY = 8192;
constexpr uint32_t TCP_SEND_TIMEOUT_US = 20000;
constexpr uint32_t TCP_PARTIAL_FRAME_RETRY_MS = 250;
constexpr uint32_t SAFETY_TASK_PERIOD_MS = 2;
constexpr uint32_t CONTROL_OVERRUN_LIMIT_US = 8000;
constexpr uint8_t CONTROL_OVERRUN_TRIP_COUNT = 8;
constexpr uint8_t SENSOR_READY_REQUIRED_COUNT = 5;
constexpr uint8_t INA_FAILURE_TRIP_COUNT = 5;
constexpr float ENCODER_MONITOR_COMMAND_V = 2.5f;
constexpr float ENCODER_MONITOR_MIN_SPEED_MPS = 0.003f;
constexpr uint16_t ENCODER_MONITOR_TRIP_COUNT = 200;  // 1.0 s at 200 Hz.
constexpr bool I2C_MINIMAL_SCANNER_MODE = false;
constexpr bool I2C_LINE_TEST_MODE = false;

enum class UsbFrameType : uint8_t {
  SNAPSHOT = 1,
  STATE = 2,
};

struct UsbFrame {
  UsbFrameType type = UsbFrameType::SNAPSHOT;
  uint64_t tUs = 0;
  float windowSeconds = 0.0f;
  float data[40] = {};
  RunMode mode = RunMode::IDLE;
  FaultCode fault = FaultCode::NONE;
  bool armed = false;
  bool imuOk = false;
  bool imuCalibrated = false;
  bool inaOk = false;
};

struct StoredConfig {
  uint32_t magic = CONFIG_MAGIC;
  RuntimeConfig value{};
};

struct SharedCommands {
  RuntimeConfig config{};
  ControllerWeights weights{};
  RunMode requestedMode = RunMode::IDLE;
  Vec2 manualVoltage{};
  Pose2D requestedPose{};
  bool modePending = false;
  bool manualPending = false;
  bool zeroPosePending = false;
  bool setPosePending = false;
  bool calibratePending = false;
  bool clearFaultPending = false;
  bool weightsPending = false;
  bool configPending = false;
  bool clientConnected = false;
  bool usbClientConnected = false;
  uint32_t lastClientRxMs = 0;
  uint32_t lastUsbRxMs = 0;
};

VehicleIO vehicle;
TheoryController controller;
Preferences preferences;
WiFiUDP udpDiag;
int tcpListenFd = -1;
int tcpClientFd = -1;
String receiveLine;
String usbReceiveLine;
SemaphoreHandle_t sharedMutex = nullptr;
QueueHandle_t telemetryQueue = nullptr;
QueueHandle_t usbFrameQueue = nullptr;
SharedCommands shared;
portMUX_TYPE stopMux = portMUX_INITIALIZER_UNLOCKED;
volatile bool hardStopRequested = false;
volatile uint32_t autonomousStopDeadlineMs = 0;
uint8_t tcpTxFailCount = 0;

RunMode activeMode = RunMode::IDLE;
FaultCode activeFault = FaultCode::NONE;
bool armed = false;
uint32_t modeStartMs = 0;
uint32_t sequenceNumber = 0;
uint32_t droppedFrames = 0;
uint16_t collectedSamples = 0;
esp_reset_reason_t bootResetReason = ESP_RST_UNKNOWN;
uint32_t bootId = 0;

const char* resetReasonName(esp_reset_reason_t reason) {
  switch (reason) {
    case ESP_RST_POWERON: return "power_on";
    case ESP_RST_EXT: return "external_reset";
    case ESP_RST_SW: return "software_reset";
    case ESP_RST_PANIC: return "panic";
    case ESP_RST_INT_WDT: return "interrupt_watchdog";
    case ESP_RST_TASK_WDT: return "task_watchdog";
    case ESP_RST_WDT: return "watchdog";
    case ESP_RST_DEEPSLEEP: return "deep_sleep";
    case ESP_RST_BROWNOUT: return "brownout";
    case ESP_RST_SDIO: return "sdio";
    default: return "unknown";
  }
}

const char* modeName(RunMode mode) {
  switch (mode) {
    case RunMode::IDLE: return "idle";
    case RunMode::MONITOR: return "monitor";
    case RunMode::MANUAL: return "manual";
    case RunMode::COLLECT: return "collect";
    case RunMode::THEORY: return "theory";
  }
  return "unknown";
}

const char* faultName(FaultCode fault) {
  switch (fault) {
    case FaultCode::NONE: return "none";
    case FaultCode::NETWORK_LOST: return "network_lost";
    case FaultCode::OVER_CURRENT: return "over_current";
    case FaultCode::UNDER_VOLTAGE: return "under_voltage";
    case FaultCode::SENSOR_FAILURE: return "sensor_failure";
    case FaultCode::NONFINITE_CONTROL: return "nonfinite_control";
    case FaultCode::CONTROL_OVERRUN: return "control_overrun";
    case FaultCode::INVALID_WEIGHTS: return "invalid_weights";
    case FaultCode::TRACKING_DIVERGENCE: return "tracking_divergence";
    case FaultCode::IMU_FAILURE: return "imu_failure";
    case FaultCode::RIGHT_ENCODER_FAILURE: return "right_encoder_failure";
    case FaultCode::LEFT_ENCODER_FAILURE: return "left_encoder_failure";
  }
  return "unknown";
}

bool parseMode(const char* value, RunMode& mode) {
  if (strcmp(value, "idle") == 0) mode = RunMode::IDLE;
  else if (strcmp(value, "monitor") == 0) mode = RunMode::MONITOR;
  else if (strcmp(value, "manual") == 0) mode = RunMode::MANUAL;
  else if (strcmp(value, "collect") == 0) mode = RunMode::COLLECT;
  else if (strcmp(value, "theory") == 0) mode = RunMode::THEORY;
  else return false;
  return true;
}

bool configValid(const RuntimeConfig& c) {
  return c.telemetryPeriod >= 0.005f && c.telemetryPeriod <= 1.0f &&
         c.collectionPeriod >= 0.005f && c.collectionPeriod <= 0.1f &&
         c.integralWindow >= 0.05f && c.integralWindow <= 1.0f &&
         c.runDuration >= 0.1f && c.runDuration <= 600.0f &&
         c.holdDuration >= 0.0f && c.holdDuration <= 60.0f &&
         c.uMax > 0.2f && c.uMax <= UserConfig::MAX_MOTOR_COMMAND_V &&
         c.motorDeadzone >= 0.0f && c.motorDeadzone < c.uMax &&
         c.currentLimit > 0.1f && c.currentLimit <= 1.60f &&
         c.batteryMin >= 6.0f && c.batteryMin <= 11.0f &&
         c.velocityTau > 0.0f && c.currentTau > 0.0f &&
         c.derivativeTau > 0.0f && c.gyroTau > 0.0f && c.gyroTau <= 1.0f &&
         c.gyroDeadband >= 0.0f && c.gyroDeadband <= 1.0f &&
          c.gyroBlend >= 0.0f && c.gyroBlend <= 1.0f &&
          c.tau1 > 0.0f && c.tau2 > 0.0f && c.r2 >= 0.0f && c.r3 >= 0.0f &&
          c.delta2 >= 0.0f && c.delta3 >= 0.0f &&
          (c.comparisonCase == 1.0f || c.comparisonCase == 2.0f ||
           c.comparisonCase == 3.0f) &&
          std::isfinite(c.pidVelocityFf) && c.pidVelocityFf >= 0.0f &&
          std::isfinite(c.pidVelocityKp) && c.pidVelocityKp >= 0.0f &&
          std::isfinite(c.pidVelocityKi) && c.pidVelocityKi >= 0.0f &&
          std::isfinite(c.pidVelocityKd) && c.pidVelocityKd >= 0.0f &&
          std::isfinite(c.pidCurrentKp) && c.pidCurrentKp >= 0.0f &&
          std::isfinite(c.pidCurrentKi) && c.pidCurrentKi >= 0.0f &&
          std::isfinite(c.pidCurrentKd) && c.pidCurrentKd >= 0.0f &&
          std::isfinite(c.pidVoltageFf) && c.pidVoltageFf >= 0.0f &&
          c.pidCurrentRefMax >= 0.05f && c.pidCurrentRefMax <= 1.0f &&
          c.pidDerivativeTau >= 0.001f && c.pidDerivativeTau <= 1.0f &&
          std::isfinite(c.outerKp) && c.outerKp > 0.0f &&
          c.outerKp <= 20.0f && std::isfinite(c.outerVbar) &&
           c.outerVbar > 0.0f && c.outerVbar <= 0.50f &&
           std::isfinite(c.outerKtheta) && c.outerKtheta > 0.0f &&
           c.outerKtheta <= 20.0f &&
           std::isfinite(c.outerPreviewHorizon) &&
           c.outerPreviewHorizon >= 0.0f &&
           c.outerPreviewHorizon <= 10.0f &&
           c.outerPreviewHorizon < c.runDuration &&
           std::isfinite(c.outerCaptureRadius) &&
           c.outerCaptureRadius > 0.0f && c.outerCaptureRadius <= 1.0f &&
           std::isfinite(c.outerBlendRadius) &&
           c.outerBlendRadius > c.outerCaptureRadius &&
           c.outerBlendRadius <= 1.0f &&
          c.motionKx > 0.0f && c.motionKy > 0.0f && c.motionKth > 0.0f &&
          c.motionMaxV > 0.0f && c.motionMaxV <= 0.50f &&
          c.motionMaxOmega > 0.0f && c.motionMaxOmega <= 5.0f &&
          c.motionMaxAccel > 0.0f && c.motionMaxAccel <= 5.0f &&
          c.motionMaxAngularAccel > 0.0f &&
          c.motionMaxAngularAccel <= 20.0f &&
          c.outerMaxV >= c.motionMaxV && c.outerMaxV <= 0.80f &&
          c.outerMaxOmega >= c.motionMaxOmega && c.outerMaxOmega <= 8.0f &&
          c.motionMaxTargetError >= 0.01f &&
          c.motionMaxTargetError <= 1.0f &&
          c.refA > 0.0f && c.refB >= 0.0f && c.refNu > 0.0f &&
          (c.refShape == 0.0f || c.refShape == 1.0f ||
           c.refShape == 2.0f || c.refShape == 3.0f) &&
          c.theoryMaxPositionError >= 0.01f && c.theoryMaxPositionError <= 1.0f &&
          c.theoryMaxHeadingError >= 0.05f && c.theoryMaxHeadingError <= 3.14f &&
          c.theoryMaxZ2 > 0.0f && c.theoryMaxZ3 > 0.0f &&
          c.theorySafetyGrace >= 0.0f && c.theorySafetyGrace <= 10.0f &&
          c.collectScale > 0.0f && c.collectScale <= 1.0f &&
         c.collectSamples >= 20 && c.collectSamples <= 5000;
}

template <typename T>
void updateNumber(JsonObjectConst object, const char* key, T& value) {
  if (object.containsKey(key)) value = object[key].as<T>();
}

void applyConfigJson(JsonObjectConst object, RuntimeConfig& config) {
  updateNumber(object, "telemetry_period", config.telemetryPeriod);
  updateNumber(object, "collection_period", config.collectionPeriod);
  updateNumber(object, "integral_window", config.integralWindow);
  updateNumber(object, "run_duration", config.runDuration);
  updateNumber(object, "hold_duration", config.holdDuration);
  updateNumber(object, "u_max", config.uMax);
  updateNumber(object, "motor_deadzone_v", config.motorDeadzone);
  updateNumber(object, "current_limit", config.currentLimit);
  updateNumber(object, "battery_min", config.batteryMin);
  updateNumber(object, "velocity_tau", config.velocityTau);
  updateNumber(object, "current_tau", config.currentTau);
  updateNumber(object, "derivative_tau", config.derivativeTau);
  updateNumber(object, "gyro_tau", config.gyroTau);
  updateNumber(object, "gyro_deadband", config.gyroDeadband);
  updateNumber(object, "gyro_blend", config.gyroBlend);
  updateNumber(object, "tau1", config.tau1);
  updateNumber(object, "tau2", config.tau2);
  updateNumber(object, "r2", config.r2);
  updateNumber(object, "r3", config.r3);
  updateNumber(object, "delta2", config.delta2);
  updateNumber(object, "delta3", config.delta3);
  if (object.containsKey("comparison_case")) {
    config.comparisonCase = object["comparison_case"].as<float>();
  } else if (object.containsKey("ref_a") || object.containsKey("ref_nu")) {
    // 旧版 PC 下发 THEORY 轨迹但没有实验编号时，安全回退为 E1，
    // 避免上次持久化的 E3 模式被意外沿用。
    config.comparisonCase = 1.0f;
  }
  updateNumber(object, "pid_velocity_ff", config.pidVelocityFf);
  updateNumber(object, "pid_velocity_kp", config.pidVelocityKp);
  updateNumber(object, "pid_velocity_ki", config.pidVelocityKi);
  updateNumber(object, "pid_velocity_kd", config.pidVelocityKd);
  updateNumber(object, "pid_current_kp", config.pidCurrentKp);
  updateNumber(object, "pid_current_ki", config.pidCurrentKi);
  updateNumber(object, "pid_current_kd", config.pidCurrentKd);
  updateNumber(object, "pid_voltage_ff", config.pidVoltageFf);
  updateNumber(object, "pid_current_ref_max", config.pidCurrentRefMax);
  updateNumber(object, "pid_derivative_tau", config.pidDerivativeTau);
  updateNumber(object, "outer_kp", config.outerKp);
  updateNumber(object, "outer_vbar", config.outerVbar);
  updateNumber(object, "outer_ktheta", config.outerKtheta);
  updateNumber(object, "outer_preview_horizon", config.outerPreviewHorizon);
  updateNumber(object, "outer_capture_radius", config.outerCaptureRadius);
  updateNumber(object, "outer_blend_radius", config.outerBlendRadius);
  updateNumber(object, "motion_kx", config.motionKx);
  updateNumber(object, "motion_ky", config.motionKy);
  updateNumber(object, "motion_kth", config.motionKth);
  updateNumber(object, "motion_max_v", config.motionMaxV);
  updateNumber(object, "motion_max_omega", config.motionMaxOmega);
  updateNumber(object, "motion_max_accel", config.motionMaxAccel);
  updateNumber(object, "motion_max_angular_accel", config.motionMaxAngularAccel);
  updateNumber(object, "outer_max_v", config.outerMaxV);
  updateNumber(object, "outer_max_omega", config.outerMaxOmega);
  updateNumber(object, "motion_max_target_error", config.motionMaxTargetError);
  updateNumber(object, "ref_a", config.refA);
  updateNumber(object, "ref_b", config.refB);
  updateNumber(object, "ref_nu", config.refNu);
  if (object.containsKey("ref_shape")) {
    config.refShape = object["ref_shape"].as<float>();
  } else if (object.containsKey("ref_a") || object.containsKey("ref_nu")) {
    // 轨迹参数组部分下发且未携带形状时回退默认形状，
    // 防止残留上一次运行（如大圆 ref_shape=1）导致双纽线/直线跑成大圆。
    config.refShape = 0.0f;
  }
  updateNumber(object, "theory_max_position_error", config.theoryMaxPositionError);
  updateNumber(object, "theory_max_heading_error", config.theoryMaxHeadingError);
  updateNumber(object, "theory_max_z2", config.theoryMaxZ2);
  updateNumber(object, "theory_max_z3", config.theoryMaxZ3);
  updateNumber(object, "theory_safety_grace", config.theorySafetyGrace);
  updateNumber(object, "collect_scale", config.collectScale);
  updateNumber(object, "collect_samples", config.collectSamples);
  if (object.containsKey("manual_snapshots")) {
    config.manualSnapshots = object["manual_snapshots"].as<bool>();
  }
  config.dtControl = UserConfig::CONTROL_PERIOD_US * 1.0e-6f;
}

void loadPersistent() {
  RuntimeConfig loadedConfig{};
  ControllerWeights loadedWeights{};
  preferences.begin("mimo-car", true);
  StoredConfig stored{};
  if (preferences.getBytesLength("config") == sizeof(stored)) {
    preferences.getBytes("config", &stored, sizeof(stored));
    if (stored.magic == CONFIG_MAGIC && configValid(stored.value)) loadedConfig = stored.value;
  }
  if (preferences.getBytesLength("weights") == sizeof(loadedWeights)) {
    preferences.getBytes("weights", &loadedWeights, sizeof(loadedWeights));
  }
  preferences.end();
  loadedConfig.dtControl = UserConfig::CONTROL_PERIOD_US * 1.0e-6f;
  xSemaphoreTake(sharedMutex, portMAX_DELAY);
  shared.config = loadedConfig;
  shared.weights = loadedWeights;
  xSemaphoreGive(sharedMutex);
}

bool savePersistent() {
  StoredConfig stored{};
  xSemaphoreTake(sharedMutex, portMAX_DELAY);
  stored.value = shared.config;
  const ControllerWeights weights = shared.weights;
  xSemaphoreGive(sharedMutex);
  preferences.begin("mimo-car", false);
  const bool okConfig = preferences.putBytes("config", &stored, sizeof(stored)) == sizeof(stored);
  const bool okWeights = preferences.putBytes("weights", &weights, sizeof(weights)) == sizeof(weights);
  preferences.end();
  return okConfig && okWeights;
}

void forceMotorOutputsOff() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(UserConfig::PIN_RIGHT_IN1, 0);
  ledcWrite(UserConfig::PIN_RIGHT_IN2, 0);
  ledcWrite(UserConfig::PIN_LEFT_IN1, 0);
  ledcWrite(UserConfig::PIN_LEFT_IN2, 0);
#else
  ledcWrite(PWM_CHANNEL_RIGHT_IN1, 0);
  ledcWrite(PWM_CHANNEL_RIGHT_IN2, 0);
  ledcWrite(PWM_CHANNEL_LEFT_IN1, 0);
  ledcWrite(PWM_CHANNEL_LEFT_IN2, 0);
#endif
}

void clearAutonomousStopDeadline() {
  portENTER_CRITICAL(&stopMux);
  autonomousStopDeadlineMs = 0;
  portEXIT_CRITICAL(&stopMux);
}

void armAutonomousStopDeadline(float durationSeconds) {
  const uint32_t durationMs = static_cast<uint32_t>(
      fmaxf(durationSeconds, 0.1f) * 1000.0f);
  portENTER_CRITICAL(&stopMux);
  autonomousStopDeadlineMs = millis() + durationMs;
  portEXIT_CRITICAL(&stopMux);
}

bool hardStopPending() {
  portENTER_CRITICAL(&stopMux);
  const bool requested = hardStopRequested;
  portEXIT_CRITICAL(&stopMux);
  return requested;
}

void requestHardStop() {
  portENTER_CRITICAL(&stopMux);
  hardStopRequested = true;
  autonomousStopDeadlineMs = 0;
  portEXIT_CRITICAL(&stopMux);
  // This path is intentionally independent of controlTask. Even if I2C or
  // controller code is blocked, a received stop command removes bridge drive.
  forceMotorOutputsOff();
}

bool consumeHardStop() {
  portENTER_CRITICAL(&stopMux);
  const bool requested = hardStopRequested;
  hardStopRequested = false;
  portEXIT_CRITICAL(&stopMux);
  return requested;
}

void safetyTask(void*) {
  for (;;) {
    uint32_t deadline;
    portENTER_CRITICAL(&stopMux);
    deadline = autonomousStopDeadlineMs;
    portEXIT_CRITICAL(&stopMux);
    if (deadline != 0 &&
        static_cast<int32_t>(millis() - deadline) >= 0) {
      requestHardStop();
    }
    vTaskDelay(pdMS_TO_TICKS(SAFETY_TASK_PERIOD_MS));
  }
}

enum class TcpWriteResult : uint8_t {
  COMPLETE = 0,
  BACKPRESSURE = 1,
  FAILED = 2,
};

TcpWriteResult sendAllTcp(int fd, const char* buffer, size_t n) {
  size_t offset = 0;
  const uint32_t startedMs = millis();
  while (offset < n) {
    const ssize_t written = send(fd, buffer + offset, n - offset, 0);
    if (written > 0) {
      offset += static_cast<size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) continue;
    if (written < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
      if (offset == 0) return TcpWriteResult::BACKPRESSURE;
      if (millis() - startedMs < TCP_PARTIAL_FRAME_RETRY_MS) {
        delay(1);
        continue;
      }
    }
    return TcpWriteResult::FAILED;
  }
  return TcpWriteResult::COMPLETE;
}

void sendBuffer(const char* buffer, size_t n) {
  if (tcpClientFd >= 0) {
    const int client = tcpClientFd;
    const TcpWriteResult frameResult = sendAllTcp(client, buffer, n);
    if (frameResult == TcpWriteResult::BACKPRESSURE) {
      // No byte entered the TCP stream, so dropping this telemetry/ACK frame
      // is framing-safe.  A transient 20 ms WiFi stall must not tear down the
      // control socket; commands with ACKs are retried by the PC.
      if (tcpTxFailCount < 255) ++tcpTxFailCount;
      if (tcpTxFailCount == 1) {
        Serial.println("[TCP] transient TX backpressure, frame dropped");
      }
      return;
    }
    if (frameResult == TcpWriteResult::COMPLETE) {
      tcpTxFailCount = 0;
    } else {
      // Once any portion of a JSON frame was sent, failure permanently breaks
      // newline framing.  Only that case requires a clean reconnect.
      Serial.println("[TCP] incomplete TX frame, forcing clean reconnect");
      close(client);
      if (tcpClientFd == client) tcpClientFd = -1;
      xSemaphoreTake(sharedMutex, portMAX_DELAY);
      shared.clientConnected = false;
      xSemaphoreGive(sharedMutex);
      tcpTxFailCount = 0;
    }
  } else {
    Serial.write(buffer, n);
  }
}

void sendJson(const JsonDocument& document) {
  if (tcpClientFd < 0) return;
  static char buffer[4096];
  const size_t expected = measureJson(document);
  if (expected == 0 || expected + 1 > sizeof(buffer)) return;
  const size_t n = serializeJson(document, buffer, sizeof(buffer));
  if (n != expected) return;
  buffer[n] = '\n';
  sendBuffer(buffer, n + 1);
}

void sendSerialJson(const JsonDocument& document) {
  serializeJson(document, Serial);
  Serial.write('\n');
}

void serialPrintf(const char* format, ...) {
  char buffer[192];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  Serial.print(buffer);
}

void sendReply(uint32_t seq, bool ok, const char* message, bool toUsb = false) {
  StaticJsonDocument<384> document;
  document["v"] = PROTOCOL_VERSION;
  document["type"] = ok ? "ack" : "error";
  document["seq"] = seq;
  document["message"] = message;
  if (toUsb) sendSerialJson(document);
  else sendJson(document);
  serialPrintf("[USB_CMD] seq=%lu %s %s\n", static_cast<unsigned long>(seq),
               ok ? "OK" : "ERROR", message);
}

void sendDiag(const char* text) {
  StaticJsonDocument<256> document;
  document["v"] = PROTOCOL_VERSION;
  document["type"] = "diag";
  document["text"] = text;
  sendJson(document);
}

void addConfig(JsonObject object, const RuntimeConfig& c) {
  object["telemetry_period"] = c.telemetryPeriod;
  object["collection_period"] = c.collectionPeriod;
  object["integral_window"] = c.integralWindow;
  object["run_duration"] = c.runDuration;
  object["hold_duration"] = c.holdDuration;
  object["u_max"] = c.uMax;
  object["motor_deadzone_v"] = c.motorDeadzone;
  object["gyro_deadband"] = c.gyroDeadband;
  object["current_limit"] = c.currentLimit;
  object["battery_min"] = c.batteryMin;
  object["tau1"] = c.tau1;
  object["tau2"] = c.tau2;
  object["r2"] = c.r2;
  object["r3"] = c.r3;
  object["delta2"] = c.delta2;
  object["delta3"] = c.delta3;
  object["comparison_case"] = c.comparisonCase;
  object["pid_velocity_ff"] = c.pidVelocityFf;
  object["pid_velocity_kp"] = c.pidVelocityKp;
  object["pid_velocity_ki"] = c.pidVelocityKi;
  object["pid_velocity_kd"] = c.pidVelocityKd;
  object["pid_current_kp"] = c.pidCurrentKp;
  object["pid_current_ki"] = c.pidCurrentKi;
  object["pid_current_kd"] = c.pidCurrentKd;
  object["pid_voltage_ff"] = c.pidVoltageFf;
  object["pid_current_ref_max"] = c.pidCurrentRefMax;
  object["pid_derivative_tau"] = c.pidDerivativeTau;
  object["outer_kp"] = c.outerKp;
  object["outer_vbar"] = c.outerVbar;
  object["outer_ktheta"] = c.outerKtheta;
  object["outer_preview_horizon"] = c.outerPreviewHorizon;
  object["outer_capture_radius"] = c.outerCaptureRadius;
  object["outer_blend_radius"] = c.outerBlendRadius;
  object["motion_kx"] = c.motionKx;
  object["motion_ky"] = c.motionKy;
  object["motion_kth"] = c.motionKth;
  object["motion_max_v"] = c.motionMaxV;
  object["motion_max_omega"] = c.motionMaxOmega;
  object["motion_max_accel"] = c.motionMaxAccel;
  object["motion_max_angular_accel"] = c.motionMaxAngularAccel;
  object["outer_max_v"] = c.outerMaxV;
  object["outer_max_omega"] = c.outerMaxOmega;
  object["motion_max_target_error"] = c.motionMaxTargetError;
  object["ref_a"] = c.refA;
  object["ref_b"] = c.refB;
  object["ref_nu"] = c.refNu;
  object["ref_shape"] = c.refShape;
  object["theory_max_position_error"] = c.theoryMaxPositionError;
  object["theory_max_heading_error"] = c.theoryMaxHeadingError;
  object["theory_max_z2"] = c.theoryMaxZ2;
  object["theory_max_z3"] = c.theoryMaxZ3;
  object["theory_safety_grace"] = c.theorySafetyGrace;
  object["collect_scale"] = c.collectScale;
  object["collect_samples"] = c.collectSamples;
  object["manual_snapshots"] = c.manualSnapshots;
}

void sendHello(uint32_t seq, bool toUsb = false) {
  RuntimeConfig config;
  xSemaphoreTake(sharedMutex, portMAX_DELAY);
  config = shared.config;
  xSemaphoreGive(sharedMutex);
  StaticJsonDocument<4096> document;
  document["v"] = PROTOCOL_VERSION;
  document["type"] = "hello";
  document["seq"] = seq;
  document["device"] = "MIMO differential-drive car";
  document["firmware"] = "1.7.4";
  document["regressor_basis"] = "sgn_gyro_gate";
  document["boot_id"] = bootId;
  document["reset_reason"] = resetReasonName(bootResetReason);
  document["uptime_ms"] = millis();
  document["ip"] = WiFi.localIP().toString();
  document["ap_ip"] = WiFi.softAPIP().toString();
  document["mode"] = modeName(activeMode);
  document["fault"] = faultName(activeFault);
  document["weights_valid"] = controller.weightsValid();
  JsonObject geometry = document.createNestedObject("geometry");
  geometry["wheel_radius_m"] = UserConfig::WHEEL_RADIUS_M;
  geometry["track_width_m"] = UserConfig::TRACK_WIDTH_M;
  geometry["ticks_per_rev"] = UserConfig::ENCODER_TICKS_PER_WHEEL_REV;
  addConfig(document.createNestedObject("config"), config);
  if (toUsb) sendSerialJson(document);
  else sendJson(document);
}

bool parseWeights(JsonObjectConst object, ControllerWeights& weights) {
  const char* regressorBasis = object["regressor_basis"] | "";
  if (strcmp(regressorBasis, "sgn_gyro_gate") != 0) return false;
  JsonArrayConst w2 = object["w2"].as<JsonArrayConst>();
  JsonArrayConst w3 = object["w3"].as<JsonArrayConst>();
  if (w2.size() != 16 || w3.size() != 16) return false;
  for (int i = 0; i < 16; ++i) {
    weights.w2[i / 8][i % 8] = w2[i].as<float>();
    weights.w3[i / 8][i % 8] = w3[i].as<float>();
  }
  updateNumber(object, "kappa2", weights.kappa2);
  updateNumber(object, "kappa3", weights.kappa3);
  updateNumber(object, "kappa_min2", weights.kappaMin2);
  updateNumber(object, "kappa_min3", weights.kappaMin3);
  updateNumber(object, "sigma_margin2", weights.sigmaMargin2);
  updateNumber(object, "sigma_margin3", weights.sigmaMargin3);
  updateNumber(object, "epsilon2", weights.epsilon2);
  updateNumber(object, "epsilon3", weights.epsilon3);
  updateNumber(object, "dbar2", weights.dbar2);
  updateNumber(object, "dbar3", weights.dbar3);
  updateNumber(object, "certificate_max2", weights.certificateMax2);
  updateNumber(object, "certificate_max3", weights.certificateMax3);
  updateNumber(object, "rank2", weights.rank2);
  updateNumber(object, "rank3", weights.rank3);
  weights.checksum = 0;
  return TheoryController::validate(weights);
}

void handleCommand(const String& line, bool fromUsb = false) {
  DynamicJsonDocument document(JSON_CAPACITY);
  if (deserializeJson(document, line)) {
    sendReply(0, false, "invalid_json", fromUsb);
    return;
  }
  const uint32_t seq = document["seq"] | 0;
  const uint16_t version = document["v"] | 0;
  const char* command = document["cmd"] | "";
  if (version != PROTOCOL_VERSION) {
    sendReply(seq, false, "unsupported_protocol", fromUsb);
    return;
  }
  xSemaphoreTake(sharedMutex, portMAX_DELAY);
  if (fromUsb) {
    shared.usbClientConnected = true;
    shared.lastUsbRxMs = millis();
  } else {
    shared.lastClientRxMs = millis();
  }
  xSemaphoreGive(sharedMutex);

  if (strcmp(command, "hello") == 0) {
    sendHello(seq, fromUsb);
  } else if (strcmp(command, "stop") == 0) {
    requestHardStop();
    if (telemetryQueue != nullptr) xQueueReset(telemetryQueue);
    if (usbFrameQueue != nullptr) xQueueReset(usbFrameQueue);
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.requestedMode = RunMode::IDLE;
    shared.modePending = true;
    shared.clearFaultPending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "stop_requested", fromUsb);
  } else if (strcmp(command, "start") == 0) {
    RunMode requested;
    if (!parseMode(document["mode"] | "", requested) || requested == RunMode::IDLE) {
      sendReply(seq, false, "invalid_mode", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.requestedMode = requested;
    shared.modePending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "start_requested", fromUsb);
  } else if (strcmp(command, "manual") == 0) {
    const float ur = document["u_r"] | NAN;
    const float ul = document["u_l"] | NAN;
    if (!std::isfinite(ur) || !std::isfinite(ul)) {
      sendReply(seq, false, "invalid_manual_voltage", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.manualVoltage = {ur, ul};
    shared.manualPending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "manual_updated", fromUsb);
  } else if (strcmp(command, "zero_pose") == 0) {
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.zeroPosePending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "zero_pose_requested", fromUsb);
  } else if (strcmp(command, "set_pose") == 0) {
    Pose2D pose{document["x"] | NAN, document["y"] | NAN, document["theta"] | NAN};
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.theta)) {
      sendReply(seq, false, "invalid_pose", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.requestedPose = pose;
    shared.setPosePending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "set_pose_requested", fromUsb);
  } else if (strcmp(command, "calibrate_imu") == 0) {
    if (armed) {
      sendReply(seq, false, "stop_before_calibration", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.calibratePending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "calibration_requested", fromUsb);
  } else if (strcmp(command, "configure") == 0) {
    if (armed) {
      sendReply(seq, false, "stop_before_configure", fromUsb);
      return;
    }
    RuntimeConfig candidate;
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    candidate = shared.config;
    xSemaphoreGive(sharedMutex);
    applyConfigJson(document.as<JsonObjectConst>(), candidate);
    if (!configValid(candidate)) {
      sendReply(seq, false, "invalid_configuration", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.config = candidate;
    shared.configPending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "configuration_updated", fromUsb);
  } else if (strcmp(command, "set_weights") == 0) {
    if (armed) {
      sendReply(seq, false, "stop_before_weights", fromUsb);
      return;
    }
    ControllerWeights candidate{};
    if (!parseWeights(document.as<JsonObjectConst>(), candidate)) {
      sendReply(seq, false, "weights_failed_route_ii_validation", fromUsb);
      return;
    }
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.weights = candidate;
    shared.weightsPending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "weights_accepted", fromUsb);
  } else if (strcmp(command, "save") == 0) {
    const bool ok = savePersistent();
    sendReply(seq, ok, ok ? "persistent_save_ok" : "persistent_save_failed", fromUsb);
  } else if (strcmp(command, "load") == 0) {
    if (armed) {
      sendReply(seq, false, "stop_before_load", fromUsb);
      return;
    }
    loadPersistent();
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    shared.configPending = true;
    shared.weightsPending = true;
    xSemaphoreGive(sharedMutex);
    sendReply(seq, true, "persistent_load_requested", fromUsb);
  } else {
    sendReply(seq, false, "unknown_command", fromUsb);
  }
}

void addVec(JsonObject object, const char* key, const Vec2& value) {
  JsonArray array = object.createNestedArray(key);
  array.add(value.x);
  array.add(value.y);
}

void addPose(JsonObject object, const char* key, const Pose2D& pose) {
  JsonArray array = object.createNestedArray(key);
  array.add(pose.x);
  array.add(pose.y);
  array.add(pose.theta);
}

template <size_t N>
void addArray(JsonObject object, const char* key, const float (&values)[N]) {
  JsonArray array = object.createNestedArray(key);
  for (size_t i = 0; i < N; ++i) array.add(values[i]);
}

void sendTelemetry(const TelemetryFrame& frame) {
  DynamicJsonDocument document(JSON_CAPACITY);
  document["v"] = PROTOCOL_VERSION;
  document["type"] = "telemetry";
  document["seq"] = frame.sequence;
  document["t_us"] = frame.timestampUs;
  document["mode"] = modeName(frame.mode);
  document["fault"] = faultName(frame.fault);
  document["armed"] = frame.armed;
  document["weights_valid"] = frame.weightsValid;
  document["dropped"] = frame.droppedFrames;
  document["loop_us"] = frame.loopTimeUs;
  JsonObject sensors = document.createNestedObject("s");
  sensors["ticks_r"] = frame.sensors.ticksRight;
  sensors["ticks_l"] = frame.sensors.ticksLeft;
  sensors["wheel_r"] = frame.sensors.wheelRight;
  sensors["wheel_l"] = frame.sensors.wheelLeft;
  addVec(sensors, "velocity_raw", frame.sensors.velocityRaw);
  addVec(sensors, "velocity", frame.sensors.velocity);
  addVec(sensors, "velocity_dot", frame.sensors.velocityDot);
  addVec(sensors, "current_raw", frame.sensors.currentRaw);
  addVec(sensors, "current", frame.sensors.current);
  addVec(sensors, "current_dot", frame.sensors.currentDot);
  addArray(sensors, "channel_current", frame.sensors.channelCurrent);
  addArray(sensors, "bus_voltage", frame.sensors.busVoltage);
  sensors["gyro_z"] = frame.sensors.gyroZ;
  addPose(sensors, "pose", frame.sensors.pose);
  sensors["ina_ok"] = frame.sensors.inaOk;
  sensors["imu_ok"] = frame.sensors.imuOk;
  sensors["imu_calibrated"] = frame.sensors.imuCalibrated;
  JsonObject control = document.createNestedObject("c");
  addVec(control, "alpha1", frame.control.alpha1);
  addVec(control, "beta1", frame.control.beta1);
  addVec(control, "beta1_dot", frame.control.beta1Dot);
  addVec(control, "alpha2", frame.control.alpha2);
  addVec(control, "beta2", frame.control.beta2);
  addVec(control, "beta2_dot", frame.control.beta2Dot);
  addVec(control, "z2", frame.control.z2);
  addVec(control, "z3", frame.control.z3);
  addVec(control, "uc", frame.control.uc);
  addVec(control, "u", frame.control.u);
  addPose(control, "reference", frame.control.reference);
  addPose(control, "motion_reference", frame.control.motionReference);
  addArray(control, "pose_error", frame.control.poseError);
  if (frame.snapshot.valid) {
    JsonObject snapshot = document.createNestedObject("snapshot");
    snapshot["kind"] = "integral_v3_sgn_gyro_gate";
    snapshot["window_s"] = frame.snapshot.windowSeconds;
    addArray(snapshot, "zdot2", frame.snapshot.zdot2);
    addArray(snapshot, "y2", frame.snapshot.y2);
    addArray(snapshot, "x3", frame.snapshot.x3);
    addArray(snapshot, "zdot3", frame.snapshot.zdot3);
    addArray(snapshot, "y3", frame.snapshot.y3);
    addArray(snapshot, "x4", frame.snapshot.x4);
    addArray(snapshot, "velocity_raw", frame.snapshot.velocityRaw);
    addArray(snapshot, "current_raw", frame.snapshot.currentRaw);
  }
  sendJson(document);
}

void sendManualState(const TelemetryFrame& frame) {
  // Ordinary keyboard driving needs sensor health and pose, not the complete
  // synthesis/controller payload.  Keeping this frame small prevents routine
  // 50 Hz status traffic from filling the ESP32 TCP send buffer.
  StaticJsonDocument<1024> document;
  document["v"] = PROTOCOL_VERSION;
  document["type"] = "state";
  document["seq"] = frame.sequence;
  document["t_us"] = frame.timestampUs;
  document["mode"] = modeName(frame.mode);
  document["fault"] = faultName(frame.fault);
  document["armed"] = frame.armed;
  document["dropped"] = frame.droppedFrames;
  document["loop_us"] = frame.loopTimeUs;
  JsonObject root = document.as<JsonObject>();
  addPose(root, "pose", frame.sensors.pose);
  addVec(root, "vel", frame.sensors.velocity);
  addVec(root, "cur", frame.sensors.current);
  addVec(root, "u", frame.control.u);
  addArray(root, "bus_voltage", frame.sensors.busVoltage);
  document["wheel_r"] = frame.sensors.wheelRight;
  document["wheel_l"] = frame.sensors.wheelLeft;
  document["gyro_z"] = frame.sensors.gyroZ;
  document["ina_ok"] = frame.sensors.inaOk;
  document["imu_ok"] = frame.sensors.imuOk;
  document["imu_calibrated"] = frame.sensors.imuCalibrated;
  sendJson(document);
}

void latchFault(FaultCode fault) {
  if (activeFault == FaultCode::NONE) {
    activeFault = fault;
    serialPrintf("[FAULT] code=%s mode=%s armed=%d\n", faultName(fault),
                 modeName(activeMode), armed ? 1 : 0);
  }
  armed = false;
  activeMode = RunMode::IDLE;
  clearAutonomousStopDeadline();
  vehicle.stopMotors();
}

void enterMode(RunMode requested, const RuntimeConfig& config) {
  vehicle.stopMotors();
  clearAutonomousStopDeadline();
  armed = false;
  activeMode = RunMode::IDLE;
  collectedSamples = 0;
  const bool dataDrivenTheory =
      requested == RunMode::THEORY && config.comparisonCase < 2.5f;
  if (dataDrivenTheory && !controller.weightsValid()) {
    latchFault(FaultCode::INVALID_WEIGHTS);
    return;
  }
  const bool sensorsRequired = requested == RunMode::COLLECT ||
      requested == RunMode::THEORY ||
      (requested == RunMode::MANUAL && config.manualSnapshots);
  // IMU 机制已取消：不再要求陀螺仪校准有效（ω 全部来自编码器）。
  controller.configure(config);
  controller.reset(vehicle.state());
  modeStartMs = millis();
  activeMode = requested;
  armed = requested == RunMode::MANUAL || requested == RunMode::COLLECT ||
          requested == RunMode::THEORY;
  if (armed && requested != RunMode::MANUAL) {
    armAutonomousStopDeadline(config.runDuration + config.holdDuration);
  }
  serialPrintf("[MODE] mode=%s armed=%d\n", modeName(activeMode), armed ? 1 : 0);
}

void controlTask(void*) {
  RuntimeConfig config;
  Vec2 manualVoltage{};
  uint32_t lastTelemetryUs = 0;
  uint32_t lastStateUs = 0;
  IntegralSnapshotAccumulator integralSnapshots;
  uint32_t lastUsbDiagnosticMs = 0;
  uint8_t overCurrentCount = 0;
  uint8_t sensorFailureCount = 0;
  uint8_t sensorReadyCount = 0;
  bool sensorGateOpen = true;
  uint16_t rightEncoderFailureCount = 0;
  uint16_t leftEncoderFailureCount = 0;
  uint8_t overrunCount = 0;
  uint32_t overrunPeakUs = 0;
  uint8_t theoryDivergenceCount = 0;
  TickType_t lastWake = xTaskGetTickCount();
  const TickType_t periodTicks = pdMS_TO_TICKS(UserConfig::CONTROL_PERIOD_US / 1000);
  for (;;) {
    const uint32_t loopStartUs = micros();
    while (Serial.available()) {
      const char value = static_cast<char>(Serial.read());
      if (value == '\n') {
        if (!usbReceiveLine.isEmpty()) handleCommand(usbReceiveLine, true);
        usbReceiveLine = "";
      } else if (value != '\r') {
        if (usbReceiveLine.length() < JSON_CAPACITY - 1) usbReceiveLine += value;
        else {
          usbReceiveLine = "";
          serialPrintf("[USB_CMD] ERROR command_too_long\n");
        }
      }
    }
    bool clientConnected;
    bool usbClientConnected;
    uint32_t lastClientRxMs;
    RunMode requestedMode = RunMode::IDLE;
    bool modePending = false;
    bool clearFault = false;
    bool zeroPose = false;
    bool setPose = false;
    bool calibrate = false;
    bool weightsPending = false;
    bool configPending = false;
    Pose2D requestedPose{};
    ControllerWeights weights{};
    xSemaphoreTake(sharedMutex, portMAX_DELAY);
    config = shared.config;
    usbClientConnected = shared.usbClientConnected &&
        millis() - shared.lastUsbRxMs <= NETWORK_TIMEOUT_MS;
    if (shared.usbClientConnected && !usbClientConnected) {
      shared.usbClientConnected = false;
    }
    clientConnected = shared.clientConnected || usbClientConnected;
    lastClientRxMs = shared.lastClientRxMs;
    if (shared.usbClientConnected &&
        (!shared.clientConnected || shared.lastUsbRxMs > lastClientRxMs)) {
      lastClientRxMs = shared.lastUsbRxMs;
    }
    if (shared.modePending) {
      requestedMode = shared.requestedMode;
      modePending = true;
      shared.modePending = false;
    }
    if (shared.manualPending) {
      manualVoltage = shared.manualVoltage;
      shared.manualPending = false;
    }
    clearFault = shared.clearFaultPending;
    shared.clearFaultPending = false;
    zeroPose = shared.zeroPosePending;
    shared.zeroPosePending = false;
    setPose = shared.setPosePending;
    requestedPose = shared.requestedPose;
    shared.setPosePending = false;
    calibrate = shared.calibratePending;
    shared.calibratePending = false;
    weightsPending = shared.weightsPending;
    weights = shared.weights;
    shared.weightsPending = false;
    configPending = shared.configPending;
    shared.configPending = false;
    xSemaphoreGive(sharedMutex);

    if (consumeHardStop()) {
      vehicle.stopMotors();
      armed = false;
      activeMode = RunMode::IDLE;
    }
    if (clearFault) activeFault = FaultCode::NONE;
    if (configPending) {
      vehicle.configure(config);
      controller.configure(config);
    }
    if (weightsPending && !controller.setWeights(weights)) activeFault = FaultCode::INVALID_WEIGHTS;
    if (zeroPose && !armed) vehicle.zeroPose();
    if (setPose && !armed) vehicle.setPose(requestedPose);
    if (calibrate && !armed) vehicle.calibrateGyro();

    vehicle.sample(config.dtControl);
    const SensorState sensors = vehicle.state();
    const uint32_t nowMs = millis();
    if (nowMs - lastUsbDiagnosticMs >= 1000) {
      serialPrintf(
          "[USB] INA=%s IMU=%s CH1_LEFT=%.4fA CH2_RIGHT=%.4fA "
          "CH3_LOGIC=%.4fA V1=%.3fV V2=%.3fV V3=%.3fV "
          "TICKS_L=%lld TICKS_R=%lld W_L=%.3f W_R=%.3f\n",
          sensors.inaOk ? "OK" : "FAIL", sensors.imuOk ? "OK" : "FAIL",
          sensors.channelCurrent[UserConfig::INA_CHANNEL_LEFT],
          sensors.channelCurrent[UserConfig::INA_CHANNEL_RIGHT],
          sensors.channelCurrent[UserConfig::INA_CHANNEL_LOGIC],
          sensors.busVoltage[UserConfig::INA_CHANNEL_LEFT],
          sensors.busVoltage[UserConfig::INA_CHANNEL_RIGHT],
          sensors.busVoltage[UserConfig::INA_CHANNEL_LOGIC],
          static_cast<long long>(sensors.ticksLeft),
          static_cast<long long>(sensors.ticksRight), sensors.wheelLeft,
          sensors.wheelRight);
      lastUsbDiagnosticMs = nowMs;
    }
    if (modePending) {
      if (requestedMode != RunMode::IDLE && activeFault != FaultCode::NONE) vehicle.stopMotors();
      else enterMode(requestedMode, config);
      sensorFailureCount = 0;
      sensorReadyCount = 0;
      const bool requestedSensorsRequired = armed &&
          (activeMode == RunMode::COLLECT || activeMode == RunMode::THEORY ||
           (activeMode == RunMode::MANUAL && config.manualSnapshots));
      sensorGateOpen = !requestedSensorsRequired;
      xSemaphoreTake(sharedMutex, portMAX_DELAY);
      usbClientConnected = shared.usbClientConnected &&
          millis() - shared.lastUsbRxMs <= NETWORK_TIMEOUT_MS;
      if (shared.usbClientConnected && !usbClientConnected) {
        shared.usbClientConnected = false;
      }
      clientConnected = shared.clientConnected || usbClientConnected;
      lastClientRxMs = shared.lastClientRxMs;
      if (shared.usbClientConnected &&
          (!shared.clientConnected || shared.lastUsbRxMs > lastClientRxMs)) {
        lastClientRxMs = shared.lastUsbRxMs;
      }
      xSemaphoreGive(sharedMutex);
    }
    if (armed && (!clientConnected || millis() - lastClientRxMs > NETWORK_TIMEOUT_MS)) {
      serialPrintf("[DBG] NETLOST cc=%d lastRx=%lu now=%lu diff=%lu\n",
                    clientConnected ? 1 : 0,
                    static_cast<unsigned long>(lastClientRxMs),
                    static_cast<unsigned long>(millis()),
                    static_cast<unsigned long>(millis() - lastClientRxMs));
      latchFault(FaultCode::NETWORK_LOST);
    }
    if (armed && activeMode != RunMode::MANUAL &&
        millis() - modeStartMs > static_cast<uint32_t>((config.runDuration + config.holdDuration) * 1000.0f)) {
      requestHardStop();
      vehicle.stopMotors();
      armed = false;
      activeMode = RunMode::IDLE;
    }
    const bool pidCollectionMode =
        activeMode == RunMode::MANUAL && config.manualSnapshots;
    const bool sensorsRequired = activeMode == RunMode::COLLECT ||
        activeMode == RunMode::THEORY || pidCollectionMode;
    bool sensorGateOpenedThisCycle = false;
    if (armed && sensorsRequired) {
      const bool inaHealthy = sensors.inaOk;
      if (!inaHealthy) {
        sensorGateOpen = false;
        sensorReadyCount = 0;
        if (++sensorFailureCount >= INA_FAILURE_TRIP_COUNT) {
          latchFault(FaultCode::SENSOR_FAILURE);
        }
      } else {
        sensorFailureCount = 0;
      }
      if (armed && inaHealthy) {
        if (sensorReadyCount < SENSOR_READY_REQUIRED_COUNT) ++sensorReadyCount;
        if (!sensorGateOpen &&
            sensorReadyCount >= SENSOR_READY_REQUIRED_COUNT) {
          sensorGateOpen = true;
          sensorGateOpenedThisCycle = true;
          integralSnapshots.reset();
          serialPrintf("[SENSOR_GATE] ready after %u consecutive samples\n",
                       static_cast<unsigned>(sensorReadyCount));
        }
      }
    } else {
      sensorFailureCount = 0;
      sensorReadyCount = 0;
      sensorGateOpen = true;
    }
    const float maxMotorCurrent =
        fmaxf(fabsf(sensors.channelCurrent[UserConfig::INA_CHANNEL_LEFT]),
              fabsf(sensors.channelCurrent[UserConfig::INA_CHANNEL_RIGHT]));
    if (armed && maxMotorCurrent > config.currentLimit) {
      if (++overCurrentCount >= 3) latchFault(FaultCode::OVER_CURRENT);
    } else overCurrentCount = 0;
    float measuredBus = sensors.busVoltage[UserConfig::INA_CHANNEL_LOGIC];
    if (measuredBus < 1.0f) {
      measuredBus = fmaxf(sensors.busVoltage[UserConfig::INA_CHANNEL_LEFT],
                          sensors.busVoltage[UserConfig::INA_CHANNEL_RIGHT]);
    }
    if (armed && measuredBus > 1.0f && measuredBus < config.batteryMin) {
      latchFault(FaultCode::UNDER_VOLTAGE);
    }

    Vec2 command{};
    SnapshotFrame snapshot{};
    const float runTime = (millis() - modeStartMs) * 0.001f;
    if (sensorGateOpenedThisCycle && activeMode == RunMode::THEORY && armed) {
      controller.reset(sensors, runTime);
    }
    if (activeMode == RunMode::MANUAL && armed &&
        (!pidCollectionMode || sensorGateOpen)) {
      command = {clampf(manualVoltage.x, -config.uMax, config.uMax),
                 clampf(manualVoltage.y, -config.uMax, config.uMax)};
    } else if (activeMode == RunMode::COLLECT && armed && sensorGateOpen) {
      command = controller.collectionExcitation(runTime);
    } else if (activeMode == RunMode::THEORY && armed && sensorGateOpen) {
      command = controller.step(sensors, runTime, config.dtControl);
      const ControlState& control = controller.state();
      const float positionError = std::sqrt(
          control.poseError[0] * control.poseError[0] +
          control.poseError[1] * control.poseError[1]);
      const float motionTargetError = std::sqrt(
          control.motionPoseError[0] * control.motionPoseError[0] +
          control.motionPoseError[1] * control.motionPoseError[1]);
      const float z2Norm = std::sqrt(
          control.z2.x * control.z2.x + control.z2.y * control.z2.y);
      const float z3Norm = std::sqrt(
          control.z3.x * control.z3.x + control.z3.y * control.z3.y);
      const bool outsideSafetyEnvelope =
          runTime >= config.theorySafetyGrace &&
          (positionError > config.theoryMaxPositionError ||
           motionTargetError > config.motionMaxTargetError ||
           std::fabs(control.poseError[2]) > config.theoryMaxHeadingError ||
           z2Norm > config.theoryMaxZ2 || z3Norm > config.theoryMaxZ3);
      if (outsideSafetyEnvelope) {
        if (++theoryDivergenceCount >= 3) {
          serialPrintf(
              "[THEORY_SAFETY] e_motion=%.4f e_target=%.4f "
              "e_th=%.4f z2=%.4f z3=%.4f\n",
              positionError, motionTargetError,
              std::fabs(control.poseError[2]), z2Norm, z3Norm);
          latchFault(FaultCode::TRACKING_DIVERGENCE);
          command = {};
        }
      } else {
        theoryDivergenceCount = 0;
      }
    } else {
      theoryDivergenceCount = 0;
    }

    // PID collection deliberately runs through MANUAL voltage commands.  A
    // missing encoder must therefore be checked explicitly: otherwise one
    // zero wheel combined with encoder-yaw fallback creates a mathematically
    // perfect but physically false circle of radius TRACK_WIDTH_M / 2.
    auto updateEncoderMonitor = [](bool suspicious, uint16_t& count) {
      if (suspicious) {
        if (count < ENCODER_MONITOR_TRIP_COUNT) ++count;
      } else if (count > 0) {
        --count;
      }
    };
    if (armed && pidCollectionMode && sensorGateOpen) {
      const bool rightSuspicious =
          std::fabs(command.x) >= ENCODER_MONITOR_COMMAND_V &&
          std::fabs(sensors.wheelRight) < ENCODER_MONITOR_MIN_SPEED_MPS;
      const bool leftSuspicious =
          std::fabs(command.y) >= ENCODER_MONITOR_COMMAND_V &&
          std::fabs(sensors.wheelLeft) < ENCODER_MONITOR_MIN_SPEED_MPS;
      updateEncoderMonitor(rightSuspicious, rightEncoderFailureCount);
      updateEncoderMonitor(leftSuspicious, leftEncoderFailureCount);
      if (rightEncoderFailureCount >= ENCODER_MONITOR_TRIP_COUNT) {
        latchFault(FaultCode::RIGHT_ENCODER_FAILURE);
        command = {};
      } else if (leftEncoderFailureCount >= ENCODER_MONITOR_TRIP_COUNT) {
        latchFault(FaultCode::LEFT_ENCODER_FAILURE);
        command = {};
      }
    } else {
      rightEncoderFailureCount = 0;
      leftEncoderFailureCount = 0;
    }

    // USB needs live pose/velocity in MANUAL too because the PC-side PID
    // collector deliberately drives through the safe manual-voltage protocol.
    if (usbClientConnected && armed &&
        (activeMode == RunMode::MANUAL || activeMode == RunMode::THEORY) &&
        loopStartUs - lastStateUs >= 40000u) {
      const ControlState& cs = controller.state();
      UsbFrame frame{};
      frame.type = UsbFrameType::STATE;
      frame.tUs = sensors.timestampUs;
      frame.mode = activeMode;
      frame.fault = activeFault;
      frame.armed = armed;
      float* const d = frame.data;
      d[0] = sensors.pose.x;       d[1] = sensors.pose.y;
      d[2] = sensors.pose.theta;   d[3] = sensors.velocity.x;
      d[4] = sensors.velocity.y;   d[5] = sensors.current.x;
      d[6] = sensors.current.y;    d[7] = cs.alpha1.x;
      d[8] = cs.alpha1.y;          d[9] = cs.beta1.x;
      d[10] = cs.beta1.y;          d[11] = cs.beta2.x;
      d[12] = cs.beta2.y;          d[13] = cs.z2.x;
      d[14] = cs.z2.y;             d[15] = cs.z3.x;
      d[16] = cs.z3.y;             d[17] = cs.uc.x;
      const Vec2 appliedVoltage = vehicle.lastMotorVoltage();
      d[18] = cs.uc.y;             d[19] = appliedVoltage.x;
      d[20] = appliedVoltage.y;    d[21] = cs.poseError[0];
      d[22] = cs.poseError[1];     d[23] = cs.poseError[2];
      float mh2 = 0.0f, mh3 = 0.0f;
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 6; ++col) {
          mh2 += cs.mhat2[row][col] * cs.mhat2[row][col];
          mh3 += cs.mhat3[row][col] * cs.mhat3[row][col];
        }
      }
      d[24] = std::sqrt(mh2);      d[25] = std::sqrt(mh3);
      d[26] = sensors.wheelRight;  d[27] = sensors.wheelLeft;
      d[28] = cs.reference.x;      d[29] = cs.reference.y;
      d[30] = cs.reference.theta;
      d[31] = cs.motionReference.x; d[32] = cs.motionReference.y;
      d[33] = cs.motionReference.theta;
      d[34] = cs.beta1Dot.x;       d[35] = cs.beta1Dot.y;
      d[36] = cs.alpha2.x;         d[37] = cs.alpha2.y;
      d[38] = cs.beta2Dot.x;       d[39] = cs.beta2Dot.y;
      frame.imuOk = sensors.imuOk;
      frame.imuCalibrated = sensors.imuCalibrated;
      frame.inaOk = sensors.inaOk;
      xQueueSend(usbFrameQueue, &frame, 0);
      lastStateUs = loopStartUs;
    }

    // At 200 Hz, integrate one complete non-overlapping window.  The endpoint
    // differences and the averaged regressors therefore describe the same
    // interval and require no numerical differentiation on the PC.
    const bool snapshotMode = armed && sensorGateOpen &&
        (activeMode == RunMode::COLLECT ||
         (activeMode == RunMode::MANUAL && config.manualSnapshots));
    const uint32_t snapshotWarmupMs = activeMode == RunMode::MANUAL
        ? MANUAL_SNAPSHOT_WARMUP_MS : COLLECTION_WARMUP_MS;
    if (snapshotMode && millis() - modeStartMs >= snapshotWarmupMs) {
      const CollectionPoint point = controller.collectionPoint(
          sensors, vehicle.lastMotorVoltage(), runTime);
      snapshot = integralSnapshots.update(
          point, sensors.timestampUs, config.integralWindow);
      if (snapshot.valid) {
        if (usbClientConnected) {
          UsbFrame frame{};
          frame.type = UsbFrameType::SNAPSHOT;
          frame.tUs = sensors.timestampUs;
          frame.windowSeconds = snapshot.windowSeconds;
          memcpy(&frame.data[0], snapshot.zdot2, sizeof(snapshot.zdot2));
          memcpy(&frame.data[2], snapshot.y2, sizeof(snapshot.y2));
          memcpy(&frame.data[10], snapshot.x3, sizeof(snapshot.x3));
          memcpy(&frame.data[12], snapshot.zdot3, sizeof(snapshot.zdot3));
          memcpy(&frame.data[14], snapshot.y3, sizeof(snapshot.y3));
          memcpy(&frame.data[22], snapshot.x4, sizeof(snapshot.x4));
          memcpy(&frame.data[24], snapshot.velocityRaw,
                 sizeof(snapshot.velocityRaw));
          memcpy(&frame.data[26], snapshot.currentRaw,
                 sizeof(snapshot.currentRaw));
          xQueueSend(usbFrameQueue, &frame, 0);
        }
        if (activeMode == RunMode::COLLECT &&
            ++collectedSamples >= config.collectSamples) {
          vehicle.stopMotors();
          armed = false;
          activeMode = RunMode::IDLE;
        }
      }
    } else {
      integralSnapshots.reset();
    }
    if (!finite2(command)) latchFault(FaultCode::NONFINITE_CONTROL);
    const bool motorOutputAllowed = armed && !hardStopPending() &&
        (!sensorsRequired || sensorGateOpen);
    if (motorOutputAllowed) {
      vehicle.setMotorVoltages(command);
      if (hardStopPending()) vehicle.stopMotors();
    } else {
      vehicle.stopMotors();
    }

    const uint32_t loopTimeUs = micros() - loopStartUs;
    if (armed && loopTimeUs > CONTROL_OVERRUN_LIMIT_US) {
      overrunPeakUs = std::max(overrunPeakUs, loopTimeUs);
      if (overrunCount < UINT8_MAX) ++overrunCount;
      if (overrunCount >= CONTROL_OVERRUN_TRIP_COUNT) {
        serialPrintf("[OVERRUN] sustained peak_us=%lu limit_us=%lu count=%u\n",
                      static_cast<unsigned long>(overrunPeakUs),
                      static_cast<unsigned long>(CONTROL_OVERRUN_LIMIT_US),
                      static_cast<unsigned>(overrunCount));
        latchFault(FaultCode::CONTROL_OVERRUN);
      }
    } else if (overrunCount > 0) {
      --overrunCount;
      if (overrunCount == 0) overrunPeakUs = 0;
    } else if (!armed) {
      overrunPeakUs = 0;
    }
    const bool tcpActive = tcpClientFd >= 0;
    const bool telemetryDue = snapshot.valid ||
        loopStartUs - lastTelemetryUs >= static_cast<uint32_t>(config.telemetryPeriod * 1.0e6f);
    if (telemetryDue && tcpActive) {
      TelemetryFrame frame{};
      frame.timestampUs = sensors.timestampUs;
      frame.sequence = ++sequenceNumber;
      frame.mode = activeMode;
      frame.fault = activeFault;
      frame.armed = armed;
      frame.weightsValid = controller.weightsValid();
      frame.droppedFrames = droppedFrames;
      frame.loopTimeUs = loopTimeUs;
      frame.sensors = sensors;
      frame.control = controller.state();
      frame.control.u = vehicle.lastMotorVoltage();
      frame.snapshot = snapshot;
      if (xQueueSend(telemetryQueue, &frame, 0) != pdTRUE) ++droppedFrames;
      lastTelemetryUs = loopStartUs;
    }
    xTaskDelayUntil(&lastWake, periodTicks);
  }
}

void connectWifi() {
  WiFi.setSleep(false);
  if (strlen(UserConfig::WIFI_STA_SSID) > 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(UserConfig::WIFI_STA_SSID, UserConfig::WIFI_STA_PASSWORD);
    const uint32_t started = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - started < 12000) delay(100);
    if (WiFi.status() == WL_CONNECTED) return;
    WiFi.disconnect(true);
  }
  WiFi.mode(WIFI_AP);
  WiFi.softAP(UserConfig::WIFI_AP_SSID, UserConfig::WIFI_AP_PASSWORD);
  const esp_err_t setResult =
      esp_wifi_set_protocol(WIFI_IF_AP,
                            WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G);
  uint8_t actualProtocol = 0;
  const esp_err_t getResult =
      esp_wifi_get_protocol(WIFI_IF_AP, &actualProtocol);
  Serial.printf("[WIFI] set_protocol=%d get_protocol=%d actual=0x%02X\n",
                static_cast<int>(setResult),
                static_cast<int>(getResult),
                actualProtocol);
}

void printI2cScan(const char* label) {
  uint8_t found = 0;
  Serial.printf("[I2C_LEVEL %s] SDA=%d SCL=%d\n", label,
                digitalRead(UserConfig::PIN_I2C_SDA),
                digitalRead(UserConfig::PIN_I2C_SCL));
  Wire.beginTransmission(UserConfig::INA3221_ADDRESS);
  const uint8_t inaError = Wire.endTransmission(true);
  Wire.beginTransmission(UserConfig::MPU6050_ADDRESS);
  const uint8_t mpuError = Wire.endTransmission(true);
  Serial.printf("[I2C_PROBE %s] 0x40_ERR=%u 0x68_ERR=%u\n", label,
                inaError, mpuError);
  Serial.printf("[I2C_SCAN %s]", label);
  if (inaError == 5 || mpuError == 5) {
    Serial.println(" BUS_ERROR");
    return;
  }
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission(true);
    if (error == 0) {
      Serial.printf(" 0x%02X", address);
      ++found;
    }
  }
  if (found == 0) Serial.print(" NONE");
  Serial.println();
}

void diagnoseI2cMappings() {
  Wire.begin(UserConfig::PIN_I2C_SDA, UserConfig::PIN_I2C_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(20);
  printI2cScan("SDA21_SCL22");
  Wire.end();

  Wire.begin(UserConfig::PIN_I2C_SCL, UserConfig::PIN_I2C_SDA);
  Wire.setClock(100000);
  Wire.setTimeOut(20);
  printI2cScan("SDA22_SCL21");
  Wire.end();
}

}  // namespace

void setup() {
  bootResetReason = esp_reset_reason();
  bootId = esp_random();
  Serial.setRxBufferSize(4096);
  Serial.begin(460800);
  Serial.println();
  Serial.println("[BOOT] MIMO car firmware starting");
  Serial.printf("[BOOT] id=%08lX reset=%s\n",
                static_cast<unsigned long>(bootId),
                resetReasonName(bootResetReason));
  if (I2C_MINIMAL_SCANNER_MODE) {
    delay(500);
    Wire.begin(UserConfig::PIN_I2C_SDA, UserConfig::PIN_I2C_SCL);
    Wire.setClock(10000);
    Wire.setTimeOut(50);
    delay(500);
    Serial.println("[I2C_MINIMAL] ACTIVE");
    return;
  }
  if (I2C_LINE_TEST_MODE) {
    pinMode(UserConfig::PIN_I2C_SDA, OUTPUT_OPEN_DRAIN);
    pinMode(UserConfig::PIN_I2C_SCL, OUTPUT_OPEN_DRAIN);
    digitalWrite(UserConfig::PIN_I2C_SDA, HIGH);
    digitalWrite(UserConfig::PIN_I2C_SCL, HIGH);
    gpio_set_pull_mode(static_cast<gpio_num_t>(UserConfig::PIN_I2C_SDA), GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(static_cast<gpio_num_t>(UserConfig::PIN_I2C_SCL), GPIO_PULLUP_ONLY);
    Serial.println("[I2C_LINE_TEST] ACTIVE");
    return;
  }

  sharedMutex = xSemaphoreCreateMutex();
  telemetryQueue = xQueueCreate(TELEMETRY_QUEUE_LENGTH, sizeof(TelemetryFrame));
  usbFrameQueue = xQueueCreate(USB_FRAME_QUEUE_LENGTH, sizeof(UsbFrame));
  loadPersistent();
  vehicle.configure(shared.config);
  controller.configure(shared.config);
  // Start WiFi AP FIRST, before any I2C operations that might block.
  // If an I2C device holds SDA/SCL low, vehicle.begin() can hang for a long
  // time; we still want the WiFi AP up so the user can connect and diagnose.
  connectWifi();
  diagnoseI2cMappings();
  const bool ioReady = vehicle.begin();
  Serial.printf("[BOOT] INA3221(0x%02X)=%s MPU6050(0x%02X)=%s IO=%s\n",
                UserConfig::INA3221_ADDRESS, vehicle.inaAvailable() ? "OK" : "FAIL",
                UserConfig::MPU6050_ADDRESS, vehicle.imuAvailable() ? "OK" : "FAIL",
                ioReady ? "READY" : "NOT_READY");
  if (TheoryController::validate(shared.weights)) controller.setWeights(shared.weights);
  tcpListenFd = socket(AF_INET, SOCK_STREAM, 0);
  if (tcpListenFd >= 0) {
    int enable = 1;
    setsockopt(tcpListenFd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(UserConfig::TCP_PORT);
    if (bind(tcpListenFd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(tcpListenFd, 4) < 0) {
      Serial.printf("[TCP] setup bind/listen fail errno=%d\n", errno);
      close(tcpListenFd);
      tcpListenFd = -1;
    } else {
      fcntl(tcpListenFd, F_SETFL, O_NONBLOCK);
      Serial.printf("[TCP] listening on %u (single native lwIP)\n",
                    UserConfig::TCP_PORT);
    }
  }
  udpDiag.begin(9999);
  xTaskCreatePinnedToCore(safetyTask, "motor-safety", 2048, nullptr, 4,
                          nullptr, 0);
  xTaskCreatePinnedToCore(controlTask, "control", 8192, nullptr, 3, nullptr, 1);
}

void loop() {
  if (I2C_MINIMAL_SCANNER_MODE) {
    printI2cScan("MINIMAL");
    delay(2000);
    return;
  }
  if (I2C_LINE_TEST_MODE) {
    pinMode(UserConfig::PIN_I2C_SDA, OUTPUT_OPEN_DRAIN);
    pinMode(UserConfig::PIN_I2C_SCL, OUTPUT_OPEN_DRAIN);
    digitalWrite(UserConfig::PIN_I2C_SDA, HIGH);
    digitalWrite(UserConfig::PIN_I2C_SCL, HIGH);
    gpio_set_pull_mode(static_cast<gpio_num_t>(UserConfig::PIN_I2C_SDA), GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(static_cast<gpio_num_t>(UserConfig::PIN_I2C_SCL), GPIO_PULLUP_ONLY);
    delay(100);
    const int idleSda = digitalRead(UserConfig::PIN_I2C_SDA);
    const int idleScl = digitalRead(UserConfig::PIN_I2C_SCL);
    digitalWrite(UserConfig::PIN_I2C_SDA, LOW);
    delay(100);
    const int lowSda = digitalRead(UserConfig::PIN_I2C_SDA);
    digitalWrite(UserConfig::PIN_I2C_SDA, HIGH);
    delay(100);
    digitalWrite(UserConfig::PIN_I2C_SCL, LOW);
    delay(100);
    const int lowScl = digitalRead(UserConfig::PIN_I2C_SCL);
    digitalWrite(UserConfig::PIN_I2C_SCL, HIGH);
    Serial.printf("[I2C_LINE_TEST] idle SDA=%d SCL=%d | SDA->low reads %d | SCL->low reads %d\n",
                  idleSda, idleScl, lowSda, lowScl);
    delay(3000);
    return;
  }
  {
    static uint32_t lastLoopDiagMs = 0;
    const uint32_t loopNow = millis();
    if (loopNow - lastLoopDiagMs >= 1000) {
      serialPrintf("[LOOP] alive ap_ip=%s sta=%u\n",
                   WiFi.softAPIP().toString().c_str(),
                   WiFi.softAPgetStationNum());
      lastLoopDiagMs = loopNow;
    }
  }
  {
    static uint32_t lastUdpDiagMs = 0;
    const int packetSize = udpDiag.parsePacket();
    if (packetSize > 0) {
      char buf[64];
      const int r = udpDiag.read(buf, sizeof(buf));
      serialPrintf("[UDP] rx=%d from %s:%u\n", r,
                   udpDiag.remoteIP().toString().c_str(),
                   udpDiag.remotePort());
      char resp[128];
      const int n = snprintf(resp, sizeof(resp), "STA=%u",
                             WiFi.softAPgetStationNum());
      udpDiag.beginPacket(udpDiag.remoteIP(), udpDiag.remotePort());
      udpDiag.write(reinterpret_cast<const uint8_t*>(buf), r);
      udpDiag.write(reinterpret_cast<const uint8_t*>("|"), 1);
      udpDiag.write(reinterpret_cast<const uint8_t*>(resp), n);
      udpDiag.endPacket();
      lastUdpDiagMs = millis();
    }
  }
  if (tcpListenFd >= 0 && tcpClientFd < 0) {
    struct sockaddr_in from{};
    socklen_t fromLen = sizeof(from);
    const int incoming = lwip_accept(tcpListenFd, (struct sockaddr*)&from, &fromLen);
    if (incoming >= 0) {
      tcpClientFd = incoming;
      int val = 1;
      setsockopt(tcpClientFd, IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val));
      linger sl{};
      sl.l_onoff = 1;
      sl.l_linger = 0;
      setsockopt(tcpClientFd, SOL_SOCKET, SO_LINGER, &sl, sizeof(sl));
      timeval snd{};
      snd.tv_sec = 0;
      snd.tv_usec = TCP_SEND_TIMEOUT_US;
      setsockopt(tcpClientFd, SOL_SOCKET, SO_SNDTIMEO, &snd, sizeof(snd));
      receiveLine = "";
      xQueueReset(telemetryQueue);
      xSemaphoreTake(sharedMutex, portMAX_DELAY);
      shared.clientConnected = true;
      shared.lastClientRxMs = millis();
      xSemaphoreGive(sharedMutex);
      tcpTxFailCount = 0;
    }
  }

  if (tcpClientFd >= 0) {
    char buf[192];
    const ssize_t r = recv(tcpClientFd, buf, sizeof(buf), MSG_DONTWAIT);
    if (r > 0) {
      for (ssize_t i = 0; i < r; ++i) {
        const char value = buf[i];
        if (value == '\n') {
          if (!receiveLine.isEmpty()) handleCommand(receiveLine, false);
          receiveLine = "";
        } else if (value != '\r') {
          if (receiveLine.length() < JSON_CAPACITY - 1) receiveLine += value;
          else {
            receiveLine = "";
            sendReply(0, false, "command_too_long");
          }
        }
      }
    } else if (r == 0 || (errno != EWOULDBLOCK && errno != EAGAIN)) {
      close(tcpClientFd);
      tcpClientFd = -1;
      xSemaphoreTake(sharedMutex, portMAX_DELAY);
      shared.clientConnected = false;
      xSemaphoreGive(sharedMutex);
      tcpTxFailCount = 0;
    }
  }

  UsbFrame usbFrame;
  int usbFramesWritten = 0;
  while (usbFramesWritten < 6 && xQueueReceive(usbFrameQueue, &usbFrame, 0) == pdTRUE) {
    StaticJsonDocument<2304> document;
    document["v"] = PROTOCOL_VERSION;
    document["type"] = usbFrame.type == UsbFrameType::SNAPSHOT ? "snapshot" : "state";
    document["t_us"] = usbFrame.tUs;
    auto addRange = [&document, &usbFrame](const char* key, int offset, int count) {
      JsonArray array = document.createNestedArray(key);
      for (int i = 0; i < count; ++i) array.add(usbFrame.data[offset + i]);
    };
    if (usbFrame.type == UsbFrameType::SNAPSHOT) {
      document["kind"] = "integral_v3_sgn_gyro_gate";
      document["window_s"] = usbFrame.windowSeconds;
      addRange("zdot2", 0, 2);   addRange("y2", 2, 8);
      addRange("x3", 10, 2);     addRange("zdot3", 12, 2);
      addRange("y3", 14, 8);     addRange("x4", 22, 2);
      addRange("velocity_raw", 24, 2);
      addRange("current_raw", 26, 2);
    } else {
      document["mode"] = modeName(usbFrame.mode);
      document["fault"] = faultName(usbFrame.fault);
      document["armed"] = usbFrame.armed;
      addRange("pose", 0, 3);    addRange("vel", 3, 2);
      addRange("cur", 5, 2);     addRange("alpha1", 7, 2);
      addRange("beta1", 9, 2);   addRange("beta2", 11, 2);
      addRange("z2", 13, 2);     addRange("z3", 15, 2);
      addRange("uc", 17, 2);     addRange("u", 19, 2);
      addRange("e", 21, 3);      addRange("mh", 24, 2);
      addRange("reference", 28, 3);
      addRange("motion_reference", 31, 3);
      addRange("beta1_dot", 34, 2);
      addRange("alpha2", 36, 2);
      addRange("beta2_dot", 38, 2);
      document["wheel_r"] = usbFrame.data[26];
      document["wheel_l"] = usbFrame.data[27];
      document["gyro_z"] = usbFrame.data[4];
      document["imu_ok"] = usbFrame.imuOk;
      document["imu_calibrated"] = usbFrame.imuCalibrated;
      document["ina_ok"] = usbFrame.inaOk;
    }
    // USB diagnostic frames must never enter the WiFi TCP stream.  WiFi uses
    // the single newest TelemetryFrame below; duplicating this queue over TCP
    // can saturate the socket during PID snapshot collection and starve ACKs.
    sendSerialJson(document);
    ++usbFramesWritten;
  }

  TelemetryFrame frame;
  if (tcpClientFd >= 0 &&
      xQueueReceive(telemetryQueue, &frame, 0) == pdTRUE) {
    // Keep only the newest frame. Sending a backlog can starve inbound stop
    // commands for seconds when WiFi throughput falls behind telemetry.
    TelemetryFrame newer;
    while (xQueueReceive(telemetryQueue, &newer, 0) == pdTRUE) frame = newer;
    if (frame.mode == RunMode::MANUAL && !frame.snapshot.valid) {
      sendManualState(frame);
    } else {
      sendTelemetry(frame);
    }
  }
  delay(1);
}
